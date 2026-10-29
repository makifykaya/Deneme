"""Ham veriyi eğitim / doğrulama / test kümelerine böler.

1) Genel kullanım — `src/<sınıf>/...` yapısındaki klasör:
    python -m src.prepare_data --src data/raw --dst data/processed

2) MIAS mamografi veri seti (Info.txt + mdbXXX.pgm dosyaları):
    python -m src.prepare_data --mias data/mias --dst data/processed
   MIAS'ta ardışık görüntüler (mdb001/mdb002 ...) aynı hastanın sol/sağ memesidir;
   veri sızıntısını önlemek için bu çiftler aynı kümede tutulur.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .config import CLASSES, IMAGE_EXTENSIONS

MIAS_SEVERITY = {"B": "benign", "M": "malignant"}


def collect_folder(src):
    """{sınıf: [(grup_anahtarı, dosya), ...]} — her dosya kendi grubudur."""
    items = {}
    for cls in CLASSES:
        files = sorted(p for p in (Path(src) / cls).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"{Path(src) / cls} altında görüntü yok")
        items[cls] = [(str(p), p) for p in files]
    return items


def collect_mias(mias_dir):
    mias_dir = Path(mias_dir)
    info = next((p for p in mias_dir.rglob("*") if p.name.lower() in {"info.txt", "info"}), None)
    if info is None:
        raise FileNotFoundError("MIAS Info.txt bulunamadı")
    labels = {}
    for line in info.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].startswith("mdb"):
            continue
        ref = parts[0]
        if parts[2] == "NORM":
            cls = "normal"
        elif len(parts) >= 4 and parts[3] in MIAS_SEVERITY:
            cls = MIAS_SEVERITY[parts[3]]
        else:
            continue
        # Aynı görüntüde birden fazla bulgu olabilir; kötü huylu öncelikli
        if labels.get(ref) != "malignant":
            labels[ref] = cls

    images = {p.stem: p for p in mias_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    items = defaultdict(list)
    for ref, cls in labels.items():
        if ref in images:
            patient = (int(ref[3:]) + 1) // 2
            items[cls].append((f"patient{patient}", images[ref]))
    missing = len(labels) - sum(len(v) for v in items.values())
    if missing:
        print(f"Uyarı: {missing} etiketli görüntü dosyası bulunamadı")
    return dict(items)


def split(items, val_ratio, test_ratio, seed):
    """Grupları (hasta) kümelere dağıtır; her grup tek bir kümede kalır."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for cls, entries in items.items():
        for key, path in entries:
            groups[key].append((cls, path))
    # Grupları baskın sınıfına göre katmanlara ayırarak dengeli bölme
    by_cls = defaultdict(list)
    for key, entries in groups.items():
        cls = "malignant" if any(c == "malignant" for c, _ in entries) else entries[0][0]
        by_cls[cls].append(key)

    assignment = {}
    for cls, keys in by_cls.items():
        keys.sort()
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * test_ratio))
        n_val = max(1, round(len(keys) * val_ratio))
        for i, key in enumerate(keys):
            assignment[key] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    result = defaultdict(list)
    for key, entries in groups.items():
        for cls, path in entries:
            result[assignment[key]].append((cls, path))
    return result


def write_split(result, dst):
    dst = Path(dst)
    for subset, entries in result.items():
        for cls in CLASSES:
            (dst / subset / cls).mkdir(parents=True, exist_ok=True)
        for cls, path in entries:
            target = dst / subset / cls / path.name
            if target.exists():
                target = target.with_name(f"{path.parent.name}_{path.name}")
            shutil.copy2(path, target)
    for subset in ("train", "val", "test"):
        counts = {c: len(list((dst / subset / c).iterdir())) for c in CLASSES}
        print(f"{subset:<5} {counts}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="<sınıf>/ alt klasörlerini içeren ham veri klasörü")
    g.add_argument("--mias", help="MIAS veri seti klasörü")
    p.add_argument("--dst", default="data/processed")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    if Path(a.dst).exists() and any(Path(a.dst).iterdir()):
        raise SystemExit(f"{a.dst} boş değil; önce silin veya başka bir --dst seçin.")
    items = collect_mias(a.mias) if a.mias else collect_folder(a.src)
    write_split(split(items, a.val, a.test, a.seed), a.dst)


if __name__ == "__main__":
    main()
