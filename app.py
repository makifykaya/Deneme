"""Meme röntgeni (mamografi) sınıflandırma arayüzü.

Çalıştırma:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

from src.config import CLASS_LABELS_TR, DEFAULT_CHECKPOINT
from src.data import load_image
from src.predict import Predictor, overlay_heatmap

st.set_page_config(page_title="Mamografi Sınıflandırma", page_icon="🩺", layout="wide")

RESULT_STYLE = {"normal": st.success, "benign": st.warning, "malignant": st.error}


@st.cache_resource
def get_predictor(path):
    return Predictor(path)


st.title("🩺 Meme Röntgeni Sınıflandırma")
st.caption("Görüntüyü yükleyin; model **hastalık yok**, **iyi huylu** veya **kötü huylu** olarak sınıflandırsın.")
st.info("⚠️ Bu uygulama bir araştırma/eğitim prototipidir. **Tıbbi teşhis yerine geçmez**; "
        "sonuçlar mutlaka bir radyolog tarafından değerlendirilmelidir.")

with st.sidebar:
    st.header("Ayarlar")
    ckpt = st.text_input("Model dosyası", os.environ.get("MODEL_PATH", DEFAULT_CHECKPOINT))
    show_cam = st.checkbox("Isı haritası (Grad-CAM) göster", value=True)

if not Path(ckpt).exists():
    st.warning(f"Model dosyası bulunamadı: `{ckpt}`. Önce modeli eğitin: "
               "`python -m src.train --data data/processed`")
    st.stop()

predictor = get_predictor(ckpt)
uploaded = st.file_uploader("Röntgen görüntüsü seçin",
                            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "pgm", "dcm"])
if uploaded is None:
    st.stop()

try:
    image = load_image(uploaded)
except Exception as e:  # bozuk / desteklenmeyen dosya
    st.error(f"Görüntü okunamadı: {e}")
    st.stop()

with st.spinner("Analiz ediliyor..."):
    probs = predictor.predict(image)
    best = max(probs, key=probs.get)

left, right = st.columns(2)
with left:
    st.subheader("Yüklenen görüntü")
    st.image(image, use_container_width=True)
with right:
    if show_cam:
        st.subheader("Modelin odaklandığı bölgeler")
        st.image(overlay_heatmap(image, predictor.gradcam(image, predictor.classes.index(best))),
                 use_container_width=True, clamp=True)

st.subheader("Sonuç")
RESULT_STYLE[best](f"**{CLASS_LABELS_TR[best]}** — güven: %{probs[best] * 100:.1f}")
for c in sorted(probs, key=probs.get, reverse=True):
    st.write(CLASS_LABELS_TR[c])
    st.progress(probs[c], text=f"%{probs[c] * 100:.1f}")
