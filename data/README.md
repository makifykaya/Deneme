# Veri klasörü

Ham veriyi şu yapıda `data/raw/` altına koyun (klasör isimleri sınıf etiketleridir):

```
data/raw/
├── normal/      # hastalık yok
├── benign/      # iyi huylu
└── malignant/   # kötü huylu
```

Ardından eğitim/doğrulama/test bölmelerini oluşturun:

```bash
python -m src.prepare_data --src data/raw --dst data/processed
```

MIAS veri seti için `--mias` seçeneğine bakın (ana README).
