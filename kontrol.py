import cv2
import json
import os

# ==========================================
# AYARLAR
# ==========================================
JSON_PATH = "results.json"
FRAMES_DIR = "frames"
OUTPUT_DIR = "Gorsel_Test_Sonuclari"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASS_INFO = {
    0: {"name": "Tasit", "color": (0, 255, 255)},
    1: {"name": "Insan", "color": (0, 255, 0)},
    2: {"name": "UAP",   "color": (255, 0, 0)},
    3: {"name": "UAI",   "color": (255, 0, 255)}
}

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Hata: {JSON_PATH} bulunamadı!")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Toplam {len(data)} frame işleniyor...")

    for frame_key, detections in data.items():
        # JSON'daki "frame_1" ifadesinden "1" sayısını ayıkla
        try:
            frame_num = int(frame_key.split('_')[1])
        except (IndexError, ValueError):
            print(f"[HATA] Beklenmeyen format: {frame_key}")
            continue
        
        # 6 haneli format: frame_000001.jpg, frame_000002.jpg ...
        file_name = f"frame_{frame_num:06d}.jpg"
        img_path = os.path.join(FRAMES_DIR, file_name)

        if not os.path.exists(img_path):
            print(f"[UYARI] Resim bulunamadı, atlanıyor: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None: continue

        # Bu karedeki tüm tespitleri çiz
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            cls_id = item["class_id"]
            
            cls_name = CLASS_INFO.get(cls_id, {}).get("name", f"Sinif_{cls_id}")
            color = CLASS_INFO.get(cls_id, {}).get("color", (255, 255, 255))
            landing = item.get("landing_status_name", "N/A")
            motion = item.get("motion_status_name", "N/A")

            # Kutuyu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Etiketi hazırla
            label = f"{cls_name} | {landing} | {motion}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # İşlenmiş kareyi kaydet
        out_path = os.path.join(OUTPUT_DIR, f"test_{file_name}")
        cv2.imwrite(out_path, img)
        print(f"Kaydedildi: {out_path}")

    print(f"\n[OK] İşlem bitti. Sonuçlar '{OUTPUT_DIR}' klasöründe.")

if __name__ == "__main__":
    main()