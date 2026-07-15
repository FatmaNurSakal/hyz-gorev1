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
## Çıktı
```text
results.json
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


