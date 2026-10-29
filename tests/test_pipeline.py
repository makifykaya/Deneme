"""Sentetik verilerle uçtan uca duman testi: hazırlık → eğitim → tahmin → değerlendirme."""

import numpy as np
from PIL import Image

from src.config import CLASSES
from src.evaluate import evaluate
from src.predict import Predictor, overlay_heatmap
from src.prepare_data import main as prepare
from src.train import parse_args, train


def _make_raw(root, n=8):
    rng = np.random.default_rng(0)
    for i, cls in enumerate(CLASSES):
        (root / cls).mkdir(parents=True)
        for j in range(n):
            arr = (rng.random((96, 80)) * 60 + i * 80).astype(np.uint8)  # sınıfa göre parlaklık
            Image.fromarray(arr).save(root / cls / f"{cls}_{j}.png")


def test_end_to_end(tmp_path):
    raw, processed, ckpt = tmp_path / "raw", tmp_path / "processed", tmp_path / "model.pt"
    _make_raw(raw)
    prepare(["--src", str(raw), "--dst", str(processed)])
    for subset in ("train", "val", "test"):
        for cls in CLASSES:
            assert any((processed / subset / cls).iterdir())

    train(parse_args(["--data", str(processed), "--epochs", "2", "--batch-size", "4", "--img-size", "64",
                      "--workers", "0", "--no-pretrained", "--output", str(ckpt)]))
    assert ckpt.exists()

    pred = Predictor(str(ckpt), device="cpu")
    img = Image.open(next((raw / "malignant").iterdir())).convert("RGB")
    probs = pred.predict(img)
    assert set(probs) == set(CLASSES)
    assert abs(sum(probs.values()) - 1) < 1e-4

    cam = pred.gradcam(img)
    assert cam.shape == (img.height, img.width)
    assert overlay_heatmap(img, cam).shape == (img.height, img.width, 3)

    cm = evaluate(processed / "test", str(ckpt), batch_size=4)
    assert cm.shape == (3, 3)
