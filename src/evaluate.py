"""Test kümesinde modeli değerlendirir: doğruluk, sınıf bazlı metrikler, karışıklık matrisi.

Örnek:
    python -m src.evaluate --data data/processed/test --checkpoint models/best_model.pt
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .config import CLASS_LABELS_TR, DEFAULT_CHECKPOINT
from .data import make_loader
from .predict import Predictor


@torch.no_grad()
def evaluate(data_dir, checkpoint, batch_size=16, save_dir=None):
    pred = Predictor(checkpoint)
    _, loader = make_loader(data_dir, False, batch_size, pred.img_size, num_workers=0)
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.softmax(pred.model(x.to(pred.device)), 1).cpu())
        targets += y.tolist()
    probs = torch.cat(probs).numpy()
    preds = probs.argmax(1)
    labels = list(range(len(pred.classes)))
    names = [CLASS_LABELS_TR[c] for c in pred.classes]

    print(classification_report(targets, preds, labels=labels, target_names=names, digits=3, zero_division=0))
    cm = confusion_matrix(targets, preds, labels=labels)
    print("Karışıklık matrisi (satır = gerçek, sütun = tahmin):")
    print(cm)
    try:
        auc = roc_auc_score(targets, probs, multi_class="ovr", labels=labels)
        print(f"Makro ROC-AUC (one-vs-rest): {auc:.3f}")
    except ValueError:
        pass  # test kümesinde her sınıf yoksa AUC hesaplanamaz

    if save_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay(cm, display_labels=pred.classes).plot(cmap="Blues")
        plt.title("Karışıklık matrisi")
        plt.savefig(Path(save_dir) / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        print(f"Grafik kaydedildi: {Path(save_dir) / 'confusion_matrix.png'}")
    return cm


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/test")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-dir", default="outputs")
    a = p.parse_args()
    evaluate(a.data, a.checkpoint, a.batch_size, a.save_dir)
