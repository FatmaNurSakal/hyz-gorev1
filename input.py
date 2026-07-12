import cv2
import math
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import os
import glob
import json

# =========================================================
# AYARLAR (GÜNCELLENEN KISIM)
# =========================================================
MODEL_PATH = "best.pt"
FRAMES_DIR = "2024_TUYZ_Online_Yarisma_Ana_Oturum"      # JPG'lerin bulunduğu klasör
CSV_PATH = "2024_TUYZ_Online_Yarisma_Ana_Oturum.csv"    # İhtiyaç halinde kullanılacak CSV
OUTPUT_JSON_PATH = "results.json"                       # Çıktı alınacak JSON dosyası

CONF_THRES = 0.25
IOU_THRES = 0.45

# Hareket analizi
MOTION_HISTORY = 5          # Kaç kare geriye bakılır
MOTION_PIXEL_THRESHOLD = 6  # Drone kompanzasyonlu gerçek eşik (daha hassas)

# Optik akış ayarları (kamera hareketi tahmini için)
OF_MAX_CORNERS = 200        # Arka plan için takip edilecek max köşe noktası
OF_QUALITY_LEVEL = 0.01
OF_MIN_DISTANCE = 10
OF_BLOCK_SIZE = 3
OF_LK_WIN_SIZE = (21, 21)   # Lucas-Kanade pencere boyutu
OF_LK_MAX_LEVEL = 3         # Piramit seviyesi

# UAP/UAI alan uygunluğu
AREA_OCCUPANCY_THRESHOLD = 0.10

# Basit tracker ayarları
MAX_CENTER_DIST = 50
MAX_MISSED_FRAMES = 10

CLASS_NAMES = {
    0: "tasit",
    1: "insan",
    2: "uap",
    3: "uai"
}

MOTION_LABELS = {
    0: "hareketsiz",
    1: "hareketli",
    -1: "hareket_durumu_yok"
}

LANDING_LABELS = {
    0: "uygun_degil",
    1: "uygun",
    -1: "inis_alani_degil"
}


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    x_left   = max(ax1, bx1)
    y_top    = max(ay1, by1)
    x_right  = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)

def overlap_ratio(area_box, other_box):
    a = box_area(area_box)
    if a <= 0:
        return 0.0
    return intersection_area(area_box, other_box) / a

def overlap_ratio_on_other(area_box, other_box):
    other_area = box_area(other_box)
    if other_area <= 0:
        return 0.0
    return intersection_area(area_box, other_box) / other_area

def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def color_for_class(class_id):
    if class_id == 0: return (0, 255, 255)   # Taşıt  - sarı
    if class_id == 1: return (0, 255, 0)     # İnsan  - yeşil
    if class_id == 2: return (255, 0, 0)     # UAP    - mavi
    if class_id == 3: return (255, 0, 255)   # UAI    - mor
    return (200, 200, 200)


# =========================================================
# KAMERA HAREKETİ TAHMİNCİSİ (Sparse Optik Akış)
# =========================================================
class CameraMotionEstimator:
    """
    İki ardışık kare arasındaki arka plan (kamera) hareketini
    Lucas-Kanade sparse optik akışı ile tahmin eder.

    Çalışma mantığı:
      1. Önceki karenin gri görüntüsünde Shi-Tomasi köşe noktaları bul.
      2. Bu noktaları mevcut karede Lucas-Kanade ile takip et.
      3. Başarılı eşleşmelerin (x, y) öteleme medyanını al → (cam_dx, cam_dy).
      4. Medyan seçimi, araçların kendi hareketi gibi "aykırı akışları" eler.
    """

    def __init__(self):
        self.prev_gray   = None
        self.cam_dx      = 0.0   # Son kare x ötelemesi (piksel)
        self.cam_dy      = 0.0   # Son kare y ötelemesi (piksel)

        # Shi-Tomasi parametreleri
        self.feature_params = dict(
            maxCorners   = OF_MAX_CORNERS,
            qualityLevel = OF_QUALITY_LEVEL,
            minDistance  = OF_MIN_DISTANCE,
            blockSize    = OF_BLOCK_SIZE
        )

        # Lucas-Kanade parametreleri
        self.lk_params = dict(
            winSize  = OF_LK_WIN_SIZE,
            maxLevel = OF_LK_MAX_LEVEL,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01
            )
        )

    def update(self, frame_bgr):
        """
        Yeni kareyi işle, (cam_dx, cam_dy) güncelle.
        İlk karede her ikisi de 0 döner.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.cam_dx = 0.0
            self.cam_dy = 0.0
            return 0.0, 0.0

        # Önceki karedeki köşe noktaları bul
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=None, **self.feature_params
        )

        if prev_pts is None or len(prev_pts) < 4:
            # Yeterli köşe bulunamazsa hareketi sıfır say
            self.prev_gray = gray
            self.cam_dx = 0.0
            self.cam_dy = 0.0
            return 0.0, 0.0

        # Lucas-Kanade ile yeni konumları bul
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Yalnızca başarılı takip edilen noktalar
        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 4:
            self.prev_gray = gray
            self.cam_dx = 0.0
            self.cam_dy = 0.0
            return 0.0, 0.0

        # Her noktanın ötelemesi
        deltas = good_next - good_prev          # shape: (N, 2)
        dx_vals = deltas[:, 0]
        dy_vals = deltas[:, 1]

        # Medyan: araçların kendi hareketini aykırı değer olarak eler
        self.cam_dx = float(np.median(dx_vals))
        self.cam_dy = float(np.median(dy_vals))

        self.prev_gray = gray
        return self.cam_dx, self.cam_dy


# =========================================================
# BASİT VE SAĞLAM TRACKER
# =========================================================
class SimpleTracker:
    def __init__(self, max_center_dist=50, max_missed=10):
        self.max_center_dist = max_center_dist
        self.max_missed      = max_missed
        self.next_id         = 0
        self.tracks          = {}

    def update(self, detections):
        for tid in self.tracks:
            self.tracks[tid]["missed"] += 1

        for det in detections:
            cls_id = det["class_id"]
            box    = det["box"]
            c      = center_of_box(box)

            best_tid, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if tr["class_id"] != cls_id:
                    continue
                dist = euclidean(c, tr["center"])
                if dist < best_dist and dist <= self.max_center_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "class_id": cls_id,
                    "center":   c,
                    "box":      box,
                    "missed":   0
                }
            else:
                tid = best_tid
                self.tracks[tid]["center"] = c
                self.tracks[tid]["box"]    = box
                self.tracks[tid]["missed"] = 0

            det["track_id"] = tid

        dead = [tid for tid, tr in self.tracks.items()
                if tr["missed"] > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

        return detections


# =========================================================
# DURUM HESAPLARI
# =========================================================
def get_motion_status(class_id, track_id, box,
                      track_history, cam_dx, cam_dy):
    """
    Drone hareketinden arındırılmış araç hareketi analizi.

    track_history[track_id]: deque of (raw_cx, raw_cy)
        → Ham piksel koordinatları (kamera hareketi dahil)

    Kompanzasyon:
        Deque'deki N nokta, N-1 kare boyunca birikmiş
        kamera hareketini içerir.

        Her kare: ham_konum = gercek_konum + kamera_oteleme
        N kare sonunda birikim: N * (cam_dx, cam_dy)

        comp_dx = raw_dx - cam_dx * len(history)
        comp_dy = raw_dy - cam_dy * len(history)

        "raw_dx - cam_dx * len(history)" satırı:
            raw_dx  → tarihsel penceredeki toplam ham öteleme
            cam_dx * len(history) → aynı sürede kameranın taşıdığı miktar
            Fark     → aracın gerçekten gittiği mesafe
    """
    if class_id != 0:
        return -1

    raw_cx, raw_cy = center_of_box(box)
    track_history[track_id].append((raw_cx, raw_cy))

    history = track_history[track_id]
    if len(history) < 2:
        return 0  # Yeterli veri yok → hareketsiz say

    # Ham başlangıç–son farkı (kamera etkisi dahil)
    first_x, first_y = history[0]
    last_x,  last_y  = history[-1]
    raw_dx = last_x - first_x
    raw_dy = last_y - first_y

    # ---- KRİTİK SATIR ----
    # Kamera hareketini çıkar:
    # cam_dx/cam_dy = son karede ölçülen ÖTELEMEDİR (piksel/kare).
    # history penceresi boyunca kameranın toplam etkisi:
    #   cam_dx * len(history) piksel
    # Bunu ham farki çıkarınca gerçek araç hareketi kalır.
    n = len(history)
    comp_dx = raw_dx - cam_dx * n   # ← "kamera hareketini çıkar"
    comp_dy = raw_dy - cam_dy * n

    compensated_dist = math.hypot(comp_dx, comp_dy)
    return 1 if compensated_dist >= MOTION_PIXEL_THRESHOLD else 0


def get_landing_status(class_id, area_box, all_detections):
    if class_id not in [2, 3]:
        return -1

    for det in all_detections:
        other_cls = det["class_id"]
        other_box = det["box"]

        if other_cls in [0, 1]:
            other_center = center_of_box(other_box)
            if point_in_box(other_center, area_box):
                return 0
            ratio_on_object = overlap_ratio_on_other(area_box, other_box)
            if ratio_on_object >= 0.30:
                return 0

    return 1


# =========================================================
# ANA
# =========================================================

def main():
    model = YOLO(MODEL_PATH)
    
    # Klasördeki JPG dosyalarını isimlerine göre sıralı şekilde al
    # (Eğer frame sırası CSV'den okunacaksa pandas ile CSV okunup yollar oradan çekilebilir)
    image_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    
    if not image_paths:
        raise RuntimeError(f"Belirtilen klasörde hiç JPG bulunamadı: {FRAMES_DIR}")

    total_images = len(image_paths) # Toplam resim sayısını alıyoruz

    tracker       = SimpleTracker(MAX_CENTER_DIST, MAX_MISSED_FRAMES)
    cam_estimator = CameraMotionEstimator()

    track_history = defaultdict(lambda: deque(maxlen=MOTION_HISTORY))
    
    json_results = []
    global_id = 22246  # JSON yapısındaki başlangıç ID'niz (istediğiniz gibi dinamik yapabilirsiniz)
    frame_idx = 0

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        frame_idx += 1

        # İlerleme yüzdesini hesapla ve aynı satıra yazdır
        progress = (frame_idx / total_images) * 100
        print(f"\rİşlem durumu: %{progress:.2f} tamamlandı ({frame_idx}/{total_images} kare)", end="")

        # 1) Kamera hareketini tahmin et (optik akış)
        cam_dx, cam_dy = cam_estimator.update(frame)

        # 2) Nesne tespiti
        results = model.predict(
            source=frame, 
            conf=CONF_THRES,
            iou=IOU_THRES, 
            verbose=False,
            device=0
        )
        result     = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes    = result.boxes.cls.cpu().numpy().astype(int)
            confs      = result.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes_xyxy, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue

                detections.append({
                    "frame":      frame_idx,
                    "class_id":   int(cls_id),
                    "class_name": CLASS_NAMES.get(int(cls_id), f"Sinif_{cls_id}"),
                    "conf":       float(conf),
                    "box":        [x1, y1, x2, y2]
                })

        # 3) Takip
        detections = tracker.update(detections)

        # 4 & 5) JSON Çıktısı İçin Liste Hazırlıkları
        detected_objects = []
        detected_undefined_objects = [] # Eğer tanımlanamayan nesne mantığınız varsa buraya ekleyin

        for det in detections:
            # Hareket ve iniş durumlarını hesapla
            motion = get_motion_status(det["class_id"], det["track_id"], det["box"], track_history, cam_dx, cam_dy)
            landing = get_landing_status(det["class_id"], det["box"], detections)
            
            x1, y1, x2, y2 = det["box"]
            cls_id = det["class_id"]

            detected_objects.append({
                "cls": f"http://localhost/classes/{cls_id}/",
                "landing_status": str(landing),
                "motion_status": str(motion),
                "top_left_x": round(float(x1), 2),
                "top_left_y": round(float(y1), 2),
                "bottom_right_x": round(float(x2), 2),
                "bottom_right_y": round(float(y2), 2)
            })

        # 6) Her Kare (Frame) İçin JSON Formatını Oluştur
        frame_data = {
            "id": global_id,
            "user": "http://localhost/users/4/",
            "frame": f"http://localhost/frames/{frame_idx}/",  # Gerçek resim ismi istenirse os.path.basename(img_path) kullanılabilir
            "detected_objects": detected_objects,
            "detected_translations": [
                {
                    "translation_x": round(float(cam_dx), 2),
                    "translation_y": round(float(cam_dy), 2),
                    "translation_z": 0.0  # Z ekseni kameranızda hesaplanmıyorsa varsayılan değer eklendi
                }
            ],
            "detected_undefined_objects": detected_undefined_objects
        }
        
        json_results.append(frame_data)
        global_id += 1

    # Tüm döngü bittikten sonra JSON Dosyasına Yazma İşlemi
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(json_results, json_file, indent=4, ensure_ascii=False)

    print(f"[OK] İşlem tamamlandı. Sonuç JSON: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()