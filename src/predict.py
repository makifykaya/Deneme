"""Eğitilmiş modelle tek görüntü tahmini ve Grad-CAM ısı haritası.

Örnek:
    python -m src.predict ornek.png --checkpoint models/best_model.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_LABELS_TR, DEFAULT_CHECKPOINT
from .data import build_transforms, load_image
from .model import build_model, gradcam_target_layer


class Predictor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.arch = ckpt["arch"]
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.model = build_model(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_transforms(train=False, img_size=self.img_size)

    def _tensor(self, image):
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image):
        """PIL görüntüsü için {sınıf: olasılık} sözlüğü döndürür."""
        probs = F.softmax(self.model(self._tensor(image)), dim=1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(self.classes, probs)}

    def gradcam(self, image, class_idx=None):
        """Modelin karar verirken odaklandığı bölgeleri gösteren [0,1] aralığında harita."""
        layer = gradcam_target_layer(self.model, self.arch)
        acts, grads = {}, {}

        def hook(module, inputs, output):
            acts["a"] = output
            output.register_hook(lambda g: grads.__setitem__("g", g))

        handle = layer.register_forward_hook(hook)
        try:
            logits = self.model(self._tensor(image))
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            self.model.zero_grad()
            logits[0, class_idx].backward()
        finally:
            handle.remove()
        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam


def overlay_heatmap(image, cam, alpha=0.45):
    """Grad-CAM haritasını görüntünün üzerine bindirir (RGB numpy dizisi)."""
    import matplotlib

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def main():
    p = argparse.ArgumentParser(description="Tek görüntü için tahmin")
    p.add_argument("image")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = p.parse_args()

    probs = Predictor(args.checkpoint).predict(load_image(args.image))
    best = max(probs, key=probs.get)
    print(f"Sonuç: {CLASS_LABELS_TR[best]} (%{probs[best] * 100:.1f})")
    for c, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {CLASS_LABELS_TR[c]:<25} %{v * 100:.1f}")


if __name__ == "__main__":
    main()
