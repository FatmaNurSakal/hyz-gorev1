# hyz-gorev1 - .venv, requirements.txt ve pytorch kurulumu

## 1. Sanal ortam kur
```powershell
.\.venv\Scripts\activate

python -m venv .venv

.venv\Scripts\activate

python -m pip install --upgrade pip
```

## 2. CUDA destekli PyTorch kur

### Senin sistem CUDA 12.8 gösteriyor, bu yüzden bunu kur:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### PyTorch GPU görüyor mu test et:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK')"
```

### Beklenen çıktı buna benzer olmalıdır:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 3. Python interpreter kontrol et ve seç
```text
Ctrl + Shift + P
Python: Select Interpreter
Python 3.14.3 (.venv) .\.venv\Scripts\python.exe
```

## 4. Proje paketlerini kur

```powershell
pip install -r requirements.txt
```

# Gorev 1 - hyz1
## Terminalden Çalıştırma Komutu:
```powershell
python main.py
```
## results.json
```text
{
    "frame_1": {
        "detected_objects": [
            {
                "cls": "http://localhost/classes/0/",
                "landing_status": "-1",
                "motion_status": "0",
                "top_left_x": 1128.161376953125,
                "top_left_y": 839.4052734375,
                "bottom_right_x": 1204.84033203125,
                "bottom_right_y": 980.6632690429688
            },
            {
                "cls": "http://localhost/classes/0/",
                "landing_status": "-1",
                "motion_status": "0",
                "top_left_x": 1124.207763671875,
                "top_left_y": 213.76455688476562,
                "bottom_right_x": 1224.097412109375,
                "bottom_right_y": 582.9495849609375
            },
            {
                "cls": "http://localhost/classes/0/",
                "landing_status": "-1",
                "motion_status": "0",
                "top_left_x": 978.47705078125,
                "top_left_y": 766.6078491210938,
                "bottom_right_x": 1075.1190185546875,
                "bottom_right_y": 984.9823608398438
            },
            {
                "cls": "http://localhost/classes/0/",
                "landing_status": "-1",
                "motion_status": "0",
                "top_left_x": 1017.3572387695312,
                "top_left_y": 0.0,
                "bottom_right_x": 1095.1878662109375,
                "bottom_right_y": 140.82260131835938
            },
            {
                "cls": "http://localhost/classes/1/",
                "landing_status": "-1",
                "motion_status": "-1",
                "top_left_x": 208.99749755859375,
                "top_left_y": 336.7648620605469,
                "bottom_right_x": 238.78591918945312,
                "bottom_right_y": 363.656982421875
            }
        ]
    },
    "frame_2"...
}
```
# Gorev 1 - hyz - Kontrol (İsteğe bağlı)
## Terminalden Çalıştırma Komutu:
```powershell
python kontrol.py
```
## Çıktı
<img width="1920" height="1080" alt="test_frame_000014" src="https://github.com/user-attachments/assets/fb712487-c823-490d-b77f-7a2ff809c075" />
<img width="1920" height="1080" alt="test_frame_001148" src="https://github.com/user-attachments/assets/46178d95-b21a-49bb-be57-6fd7c4d4de55" />
<img width="1920" height="1080" alt="test_frame_001241" src="https://github.com/user-attachments/assets/dac51264-af20-498a-ae08-40a7ceabd9f1" />

# Not - Tüm görevler için Örnek JSON çıktı formatı
```text
Görev 1: Nesne Tespiti (detected_objects)
Görev 2: Pozisyon Tespiti / Kestirimi (detected_translations)
Görev 3: Görüntü Eşleme / Referans Obje Tespiti (detected_undefined_objects)

Şartnamede paylaşılan örnek JSON çıktı formatı şu şekildedir:
[
  {
    "id": 22246,
    "user": "http://localhost/users/4/",
    "frame": "http://localhost/frames/4000/",
    "detected_objects": [
      {
        "cls": "http://localhost/classes/1/",
        "landing_status": "-1",
        "motion_status": "-1",
        "top_left_x": 262.87,
        "top_left_y": 734.47,
        "bottom_right_x": 405.2,
        "bottom_right_y": 847.3
      }
    ],
    "detected_translations": [
      {
        "translation_x": 0.02,
        "translation_y": 0.01,
        "translation_z": 0.03
      }
    ],
    "detected_undefined_objects": [
      {
        "object_id": 1,
        "top_left_x": 262.87,
        "top_left_y": 734.47,
        "bottom_right_x": 405.2,
        "bottom_right_y": 847.3
      }
    ]
  }
]
```


