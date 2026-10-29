# Meme Röntgeni (Mamografi) Sınıflandırma

Mamografi görüntülerini yapay zeka ile **3 sınıfa** ayıran proje:

| Sınıf | Anlamı |
|---|---|
| `normal` | Hastalık yok |
| `benign` | İyi huylu |
| `malignant` | Kötü huylu |

Proje iki aşamadan oluşur:
1. **Model eğitimi**: ImageNet'te önceden eğitilmiş bir CNN (EfficientNet-B0 / ResNet50 / DenseNet121) transfer öğrenme ile ince ayarlanır.
2. **Uygulama**: Streamlit arayüzüne bir röntgen görüntüsü yüklenir; uygulama sınıfı, olasılıkları ve modelin odaklandığı bölgeleri (Grad-CAM ısı haritası) gösterir.

> ⚠️ Bu proje araştırma/eğitim amaçlıdır ve **tıbbi teşhis aracı değildir**. Klinik kullanım için onaylı (ör. CE/FDA) bir süreç, büyük ve çeşitli veri ile bağımsız doğrulama gerekir.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Veri seti

Önerilen açık veri setleri:

| Veri seti | Tür | Sınıflar | Not |
|---|---|---|---|
| [MIAS](https://www.kaggle.com/datasets/kmader/mias-mammography) | Mamografi (röntgen) | normal / benign / malignant | 322 görüntü, doğrudan 3 sınıf. `--mias` ile otomatik hazırlanır |
| [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) | Mamografi (röntgen) | benign / malignant | Büyük; "normal" sınıfı için DDSM normal vakaları eklenmeli |
| [VinDr-Mammo](https://physionet.org/content/vindr-mammo/) | Mamografi (DICOM) | BI-RADS 1–5 | BI-RADS 1 → normal, 2–3 → benign, 4–5 → malignant eşlemesi yapılabilir |
| [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | **Ultrason** | normal / benign / malignant | Röntgen değil, ama 3 sınıf hazır; denemeler için pratik |

### Veriyi hazırlama

Kendi verinizi şu yapıya koyun:

```
data/raw/
├── normal/
├── benign/
└── malignant/
```

ve %70 / %15 / %15 eğitim / doğrulama / test olarak bölün:

```bash
python -m src.prepare_data --src data/raw --dst data/processed
```

MIAS için (Info.txt ve `mdbXXX.pgm` dosyalarının olduğu klasör):

```bash
python -m src.prepare_data --mias data/mias --dst data/processed
```

MIAS'ta aynı hastanın iki memesi aynı kümede tutulur (veri sızıntısını önlemek için).
Kendi verinizde de **aynı hastaya ait görüntülerin farklı kümelere düşmemesine** dikkat edin.

## 2. Eğitim

```bash
python -m src.train --data data/processed --arch efficientnet_b0 --epochs 30
```

- Dengesiz sınıflar için ağırlıklı kayıp fonksiyonu kullanılır.
- Veri artırma: çevirme, döndürme, kırpma, parlaklık/kontrast.
- En iyi **doğrulama makro F1** skoruna sahip model `models/best_model.pt` olarak kaydedilir; erken durdurma uygulanır.
- GPU (CUDA) veya Apple Silicon (MPS) varsa otomatik kullanılır. GPU yoksa [Google Colab](https://colab.research.google.com/) önerilir.

Önemli seçenekler: `--img-size` (varsayılan 384), `--batch-size`, `--lr`, `--patience`. Tümü için `python -m src.train -h`.

## 3. Değerlendirme

```bash
python -m src.evaluate --data data/processed/test --checkpoint models/best_model.pt
```

Sınıf bazlı precision / recall / F1, karışıklık matrisi ve ROC-AUC yazdırılır; matris `outputs/confusion_matrix.png` olarak kaydedilir.
Tıbbi problemlerde özellikle **kötü huylu sınıfın duyarlılığına (recall)** dikkat edin: kaçırılan bir kanser vakası, yanlış alarmdan çok daha maliyetlidir.

## 4. Uygulama

```bash
streamlit run app.py
```

Tarayıcıda açılan sayfada görüntüyü (PNG, JPG, PGM, TIFF veya DICOM `.dcm`) yükleyin. Uygulama:
- tahmin edilen sınıfı ve güven oranını,
- her sınıfın olasılığını,
- Grad-CAM ısı haritasını gösterir.

Farklı bir model dosyası için kenar çubuğunu veya `MODEL_PATH` ortam değişkenini kullanın.

Komut satırından tek görüntü tahmini:

```bash
python -m src.predict ornek.png --checkpoint models/best_model.pt
```

## Proje yapısı

```
├── app.py               # Streamlit arayüzü
├── src/
│   ├── config.py        # Sınıflar, görüntü boyutu vb.
│   ├── data.py          # Görüntü okuma (DICOM dahil), artırma, veri yükleyici
│   ├── model.py         # Transfer öğrenme modelleri
│   ├── prepare_data.py  # Eğitim/doğrulama/test bölme, MIAS dönüştürücü
│   ├── train.py         # Eğitim
│   ├── evaluate.py      # Test metrikleri
│   └── predict.py       # Tahmin + Grad-CAM
└── tests/               # Sentetik veriyle uçtan uca test (pytest)
```

## Test

```bash
pytest -q
```
