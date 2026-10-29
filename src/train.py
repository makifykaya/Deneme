"""Modeli eğitir ve en iyi doğrulama skoruna sahip ağırlıkları kaydeder.

Örnek:
    python -m src.train --data data/processed --epochs 30 --arch efficientnet_b0
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import CLASSES, DEFAULT_CHECKPOINT, IMG_SIZE
from .data import class_weights, make_loader
from .model import SUPPORTED_ARCHS, build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += logits.argmax(1).tolist()
            targets += y.tolist()
    n = len(loader.dataset)
    acc = sum(p == t for p, t in zip(preds, targets)) / n
    f1 = f1_score(targets, preds, average="macro", labels=list(range(len(CLASSES))), zero_division=0)
    return total_loss / n, acc, f1


def train(args):
    device = get_device()
    data = Path(args.data)
    train_ds, train_loader = make_loader(data / "train", True, args.batch_size, args.img_size, args.workers)
    val_ds, val_loader = make_loader(data / "val", False, args.batch_size, args.img_size, args.workers)
    print(f"Cihaz: {device} | eğitim: {len(train_ds)} | doğrulama: {len(val_ds)}")
    print("Eğitim sınıf dağılımı:", dict(zip(CLASSES, train_ds.class_counts())))

    model = build_model(args.arch, len(CLASSES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, patience_left, history = -1.0, None, args.patience, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))
        print(f"[{epoch:03d}/{args.epochs}] eğitim kayıp {tr_loss:.4f} doğ. {tr_acc:.3f} | "
              f"doğrulama kayıp {va_loss:.4f} doğ. {va_acc:.3f} F1 {va_f1:.3f} ({time.time() - t0:.0f}s)")

        if va_f1 > best_f1:
            best_f1, best_state, patience_left = va_f1, copy.deepcopy(model.state_dict()), args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Erken durdurma: doğrulama F1 iyileşmiyor.")
                break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": args.arch, "classes": CLASSES,
                "img_size": args.img_size, "val_f1": best_f1}, out)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"En iyi model kaydedildi: {out} (doğrulama makro F1 = {best_f1:.3f})")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Meme röntgeni (mamografi) 3 sınıflı sınıflandırıcı eğitimi")
    p.add_argument("--data", default="data/processed", help="train/ val/ test/ içeren klasör")
    p.add_argument("--arch", default="efficientnet_b0", choices=SUPPORTED_ARCHS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--patience", type=int, default=7, help="Erken durdurma sabrı (epoch)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="ImageNet ağırlıklarını kullanma")
    p.add_argument("--output", default=DEFAULT_CHECKPOINT)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
