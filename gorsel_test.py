import cv2
import json
import os

# ==========================================
# AYARLAR
# ==========================================
JSON_PATH = "results.json"
FRAMES_DIR = "2024_TUYZ_Online_Yarisma_Ana_Oturum"
OUTPUT_DIR = "Gorsel_Test_Sonuclari"

# Çıktı klasörü yoksa oluştur
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Sınıf İsimleri ve Renkleri (BGR formatında)
CLASS_INFO = {
    "0": {"name": "Tasit", "color": (0, 255, 255)},   # Sarı
    "1": {"name": "Insan", "color": (0, 255, 0)},     # Yeşil
    "2": {"name": "UAP",   "color": (255, 0, 0)},     # Mavi
    "3": {"name": "UAI",   "color": (255, 0, 255)}    # Mor
}

def main():
    # JSON dosyasını oku
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_frames = len(data)
    print(f"Toplam {total_frames} kare test edilecek...")


    for idx, item in enumerate(data):
        # 1. JSON'dan frame numarasını al (Örn: "1")
        frame_url = item.get("frame", "")
        # Eğer frame URL formatındaysa rakamı al, değilse doğrudan sayıyı al
        frame_raw_num = frame_url.strip("/").split("/")[-1]
        
        # 2. Rakamı 6 haneli, başına sıfır eklenen formata dönüştür (Örn: "1" -> "000001")
        # Yarışma formatında 4'er artış varsa frame numarası ile dosya ismi eşleşmelidir.
        # Eğer frame numarası 1, 2, 3 ise ve dosyalar 000000, 000004 ise:
        # frame_int = int(frame_raw_num) * 4 
        # file_num = f"{frame_int:06d}"
        
        # Basitçe 6 haneli formata çevirmek için:
        file_num = f"{int(frame_raw_num):06d}" 
        
        # 3. Resim yolunu oluştur (Örn: "frame_000001.jpg")
        img_path = os.path.join(FRAMES_DIR, f"frame_{file_num}.jpg")
        
        # Hata ayıklama için (resim bulunamazsa yolu terminale yazar)
        if not os.path.exists(img_path):
            # Eğer resimler 000000, 000004 şeklinde gidiyorsa yukarıdaki 
            # frame_int hesaplamasını aktif etmelisin.
            print(f"\n[UYARI] Resim bulunamadı, bu yol kontrol edilsin: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        detected_objects = item.get("detected_objects", [])
        
        # Her bir tespit edilen nesneyi resim üzerine çiz
        for obj in detected_objects:
            # Sınıf ID'sini URL'den çek ("http://localhost/classes/3/" -> "3")
            cls_url = obj.get("cls", "")
            cls_id = cls_url.strip("/").split("/")[-1]
            
            cls_name = CLASS_INFO.get(cls_id, {}).get("name", f"Sinif_{cls_id}")
            color = CLASS_INFO.get(cls_id, {}).get("color", (255, 255, 255))

            landing = obj.get("landing_status", "-1")
            motion = obj.get("motion_status", "-1")

            # Koordinatları al
            x1 = int(float(obj.get("top_left_x", 0)))
            y1 = int(float(obj.get("top_left_y", 0)))
            x2 = int(float(obj.get("bottom_right_x", 0)))
            y2 = int(float(obj.get("bottom_right_y", 0)))

            # Kutuyu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Sınıf, iniş ve hareket bilgilerini metin olarak ekle
            label = f"{cls_name} | Inis:{landing} | Hrk:{motion}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # İşlenmiş resmi yeni klasöre kaydet
        out_path = os.path.join(OUTPUT_DIR, f"test_{file_num}.jpg")
        cv2.imwrite(out_path, img)

        # Konsolda ilerleme yüzdesini göster
        progress = ((idx + 1) / total_frames) * 100
        print(f"\rİşlem durumu: %{progress:.2f} tamamlandı...", end="")

    print(f"\n[OK] İşlem bitti. Görsel test sonuçlarını '{OUTPUT_DIR}' klasöründen inceleyebilirsin.")

if __name__ == "__main__":
    main()