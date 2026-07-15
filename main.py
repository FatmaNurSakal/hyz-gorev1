import cv2
import math
import numpy as np
import json
import os
import glob
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# AYARLAR
# =========================================================
MODEL_PATH = "best.pt"
INPUT_FOLDER = "frames"
OUTPUT_JSON_PATH = "results.json"

CONF_THRES = 0.25
IOU_THRES = 0.45

MOTION_HISTORY = 5
MOTION_PIXEL_THRESHOLD = 6

OF_MAX_CORNERS = 200
OF_QUALITY_LEVEL = 0.01
OF_MIN_DISTANCE = 10
OF_BLOCK_SIZE = 3
OF_LK_WIN_SIZE = (21, 21)
OF_LK_MAX_LEVEL = 3

MAX_CENTER_DIST = 50
MAX_MISSED_FRAMES = 10

CLASS_NAMES = {0: "tasit", 1: "insan", 2: "uap", 3: "uai"}

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top: return 0
    return (x_right - x_left) * (y_bottom - y_top)

def overlap_ratio_on_other(area_box, other_box):
    other_area = box_area(other_box)
    if other_area <= 0: return 0.0
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

# =========================================================
# SINIFLAR
# =========================================================
class CameraMotionEstimator:
    def __init__(self):
        self.prev_gray = None
        self.cam_dx, self.cam_dy = 0.0, 0.0
        self.feature_params = dict(maxCorners=OF_MAX_CORNERS, qualityLevel=OF_QUALITY_LEVEL, minDistance=OF_MIN_DISTANCE, blockSize=OF_BLOCK_SIZE)
        self.lk_params = dict(winSize=OF_LK_WIN_SIZE, maxLevel=OF_LK_MAX_LEVEL, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray
            return 0.0, 0.0
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None, **self.lk_params)
        good_prev, good_next = prev_pts[status == 1], next_pts[status == 1]
        if len(good_prev) < 4:
            self.prev_gray = gray
            return 0.0, 0.0
        deltas = good_next - good_prev
        self.cam_dx, self.cam_dy = float(np.median(deltas[:, 0])), float(np.median(deltas[:, 1]))
        self.prev_gray = gray
        return self.cam_dx, self.cam_dy

class SimpleTracker:
    def __init__(self, max_center_dist=50, max_missed=10):
        self.max_center_dist = max_center_dist
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        for tid in self.tracks: self.tracks[tid]["missed"] += 1
        for det in detections:
            cls_id, box = det["class_id"], det["box"]
            c = center_of_box(box)
            best_tid, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if tr["class_id"] == cls_id:
                    dist = euclidean(c, tr["center"])
                    if dist < best_dist and dist <= self.max_center_dist:
                        best_dist, best_tid = dist, tid
            if best_tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"class_id": cls_id, "center": c, "box": box, "missed": 0}
            else:
                tid = best_tid
                self.tracks[tid].update({"center": c, "box": box, "missed": 0})
            det["track_id"] = tid
        for tid in [tid for tid, tr in self.tracks.items() if tr["missed"] > self.max_missed]:
            del self.tracks[tid]
        return detections

# =========================================================
# ANA MANTIKSAL FONKSİYONLAR
# =========================================================
def get_motion_status(class_id, track_id, box, track_history, cam_dx, cam_dy):
    # Tespit edilen nesne taşıt (0) değilse hareket durumu "Taşıt Değil" (-1) olmalıdır.
    if class_id != 0: 
        return "-1"
    
    raw_cx, raw_cy = center_of_box(box)
    track_history[track_id].append((raw_cx, raw_cy))
    history = track_history[track_id]
    if len(history) < 2: 
        return "0"  # Veri yetersizse varsayılan hareketsiz
    
    raw_dx = history[-1][0] - history[0][0]
    raw_dy = history[-1][1] - history[0][1]
    comp_dx = raw_dx - cam_dx * len(history)
    comp_dy = raw_dy - cam_dy * len(history)
    
    # Piksel eşik değerine göre hareketli (1) veya hareketsiz (0)
    if math.hypot(comp_dx, comp_dy) >= MOTION_PIXEL_THRESHOLD:
        return "1"
    else:
        return "0"

def get_landing_status(class_id, area_box, all_detections):
    # Tespit edilen nesne UAP (2) veya UAİ (3) alanı değilse iniş durumu "İniş Alanı Değil" (-1) olmalıdır.
    if class_id not in [2, 3]: 
        return "-1"
    
    for det in all_detections:
        # Alan üzerinde insan (1) veya taşıt (0) varsa inişe uygun değil (0)
        if det["class_id"] in [0, 1] and point_in_box(center_of_box(det["box"]), area_box): 
            return "0"
        if det["class_id"] in [0, 1] and overlap_ratio_on_other(area_box, det["box"]) >= 0.30: 
            return "0"
            
    # Alan temiz ise inişe uygun (1)
    return "1"

# =========================================================
# ANA İŞLEM DÖNGÜSÜ
# =========================================================
def main():
    model = YOLO(MODEL_PATH)
    tracker = SimpleTracker(MAX_CENTER_DIST, MAX_MISSED_FRAMES)
    cam_estimator = CameraMotionEstimator()
    track_history = defaultdict(lambda: deque(maxlen=MOTION_HISTORY))
    
    image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    
    final_results = {} 

    for idx, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        frame_idx = idx + 1
        
        # Frame için tespit anahtarını oluştur
        frame_key = f"frame_{frame_idx}"
        final_results[frame_key] = {
            "detected_objects": []
        }
        
        cam_dx, cam_dy = cam_estimator.update(frame)
        results = model.predict(source=frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False, device=0)
        
        detections = []
        if results[0].boxes is not None:
            for box, cls_id, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                detections.append({
                    "class_id": int(cls_id), 
                    "class_name": CLASS_NAMES.get(int(cls_id)), 
                    "conf": float(conf), 
                    "box": [float(b) for b in box] # Koordinatları float tipinde tutuyoruz
                })
        
        detections = tracker.update(detections)
        
        for det in detections:
            # Durum ID tespitlerini doğrudan teknik şartnameye uygun formatta string olarak hesaplıyoruz
            motion_status_str = get_motion_status(det["class_id"], det["track_id"], det["box"], track_history, cam_dx, cam_dy)
            landing_status_str = get_landing_status(det["class_id"], det["box"], detections)
            
            # Nesne sınıf adresini oluştur (örn: "http://localhost/classes/0/")
            cls_url = f"http://localhost/classes/{det['class_id']}/"
            
            # Koordinat sınır kutuları
            x1, y1, x2, y2 = det["box"]
            
            # Şartnamede belirtilen formata uygun JSON objesi
            formatted_obj = {
                "cls": cls_url,
                "landing_status": landing_status_str,
                "motion_status": motion_status_str,
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2
            }
            
            # Frame altındaki detected_objects dizisine ekle
            final_results[frame_key]["detected_objects"].append(formatted_obj)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"[OK] İşlem tamamlandı. Sonuçlar kaydedildi: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()