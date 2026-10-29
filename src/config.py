"""Projede ortak kullanılan sabitler."""

# Sınıf sırası modelin çıkış indekslerini belirler; değiştirmeyin.
CLASSES = ["benign", "malignant", "normal"]

CLASS_LABELS_TR = {
    "benign": "İyi huylu (benign)",
    "malignant": "Kötü huylu (malignant)",
    "normal": "Hastalık yok (normal)",
}

IMG_SIZE = 384
# ImageNet istatistikleri (önceden eğitilmiş ağırlıklarla uyumlu)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".dcm"}

DEFAULT_CHECKPOINT = "models/best_model.pt"
