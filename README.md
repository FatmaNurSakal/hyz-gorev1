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

## 3. Python interpreter kontrol et
```text
Ctrl + Shift + P
Python: Select Interpreter
Python 3.14.3 (pytorch) 
.\.venv\Scripts\python.exe
```

## 4. Proje paketlerini kur

```powershell
pip install -r requirements.txt
```

# Gorev 1 - hyz1
## Terminalden Çalıştırma Komutu:
```powershell
python input.py
```
## Çıktı
```text
results.json
```
## Terminalden Çalıştırma Komutu:
```powershell
python gorsel_test.py
```
## Çıktı
<img width="1920" height="1080" alt="test_000004" src="https://github.com/user-attachments/assets/6ef4c7e2-ce00-40bd-9912-a32a79bc949d" />

