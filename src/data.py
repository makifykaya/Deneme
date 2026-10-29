"""Görüntü okuma, ön işleme ve veri yükleyiciler."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLASSES, IMAGE_EXTENSIONS, IMG_SIZE, MEAN, STD


def load_image(source) -> Image.Image:
    """Dosya yolu veya dosya benzeri nesneden RGB PIL görüntüsü döndürür.

    DICOM (.dcm) dosyaları pydicom ile okunur ve 8 bite normalize edilir.
    Mamografiler genelde gri tonlamalıdır; önceden eğitilmiş ağlar 3 kanal
    beklediği için görüntü RGB'ye çevrilir.
    """
    name = str(getattr(source, "name", source)).lower()
    if name.endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(source)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
    return Image.open(source).convert("RGB")


def build_transforms(train: bool, img_size: int = IMG_SIZE):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class MammogramDataset(Dataset):
    """`root/<sınıf>/<görüntü>` yapısındaki klasörü okur."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for idx, cls in enumerate(CLASSES):
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Sınıf klasörü bulunamadı: {cls_dir}")
            for p in sorted(cls_dir.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, idx))
        if not self.samples:
            raise RuntimeError(f"{self.root} altında görüntü bulunamadı")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = [0] * len(CLASSES)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def class_weights(dataset: MammogramDataset) -> torch.Tensor:
    """Dengesiz veri için ters frekans sınıf ağırlıkları."""
    counts = torch.tensor(dataset.class_counts(), dtype=torch.float32).clamp(min=1)
    return counts.sum() / (len(counts) * counts)


def make_loader(root, train: bool, batch_size: int, img_size: int = IMG_SIZE, num_workers: int = 2):
    ds = MammogramDataset(root, build_transforms(train, img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return ds, loader
