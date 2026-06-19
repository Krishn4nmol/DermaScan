"""
DermaScan — Streamlit Web App
Skin Lesion Detection using Lightweight Deep Learning
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import matplotlib.cm as cm
from pathlib import Path

from models.backbones import build_model
from utils.preprocess import preprocess_image
from utils.dataset import get_val_transforms

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Melanocytic Nevi", "Melanoma", "Benign Keratosis",
    "Basal Cell Carcinoma", "Actinic Keratoses",
    "Vascular Lesions", "Dermatofibroma"
]

CLASS_INFO = {
    "Melanocytic Nevi":      {"risk": "Low",    "color": "🟢", "desc": "Common benign mole. Usually harmless but monitor for changes."},
    "Melanoma":              {"risk": "High",   "color": "🔴", "desc": "Malignant skin cancer. Requires immediate medical attention."},
    "Benign Keratosis":      {"risk": "Low",    "color": "🟢", "desc": "Non-cancerous skin growth. Generally harmless."},
    "Basal Cell Carcinoma":  {"risk": "Medium", "color": "🟡", "desc": "Most common skin cancer. Rarely spreads, but needs treatment."},
    "Actinic Keratoses":     {"risk": "Medium", "color": "🟡", "desc": "Pre-cancerous lesion caused by sun damage. Needs monitoring."},
    "Vascular Lesions":      {"risk": "Low",    "color": "🟢", "desc": "Blood vessel abnormality. Usually benign."},
    "Dermatofibroma":        {"risk": "Low",    "color": "🟢", "desc": "Benign fibrous nodule. Harmless skin growth."},
}

CHECKPOINTS = {
    "EfficientNet-B0 (Best Accuracy)":  "checkpoints/efficientnet_best.pth",
    "MobileNetV3 (Balanced)":           "checkpoints/mobilenetv3_best.pth",
    "ShuffleNetV2 (Fastest)":           "checkpoints/shufflenetv2_best.pth",
}

MODEL_KEYS = {
    "EfficientNet-B0 (Best Accuracy)":  "efficientnet",
    "MobileNetV3 (Balanced)":           "mobilenetv3",
    "ShuffleNetV2 (Fastest)":           "shufflenetv2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name: str, ckpt_path: str):
    model = build_model(model_name, num_classes=7, pretrained=False,
                        dropout=0.4, use_cbam=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model  = model
        self.grads  = None
        self.acts   = None
        self._hooks = []
        self._hooks.append(target_layer.register_forward_hook(self._save_act))
        self._hooks.append(target_layer.register_backward_hook(self._save_grad))

    def _save_act(self, m, i, o):  self.acts = o.detach()
    def _save_grad(self, m, i, o): self.grads = o[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        for h in self._hooks: h.remove()


def get_target_layer(model, model_name):
    if model_name == "mobilenetv3":  return model.features[-1]
    if model_name == "efficientnet": return model.features[-1]
    if model_name == "shufflenetv2": return model.conv5


def make_overlay(image_np, cam, alpha=0.5):
    h, w = image_np.shape[:2]
    if CV2_AVAILABLE:
        cam_r = cv2.resize(cam, (w, h))
    else:
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_r = np.array(cam_pil.resize((w, h))) / 255.0
    heatmap = (cm.jet(cam_r)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, model_name, pil_image, use_preprocess=True):
    if use_preprocess:
        pil_image = preprocess_image(pil_image)

    img_np    = np.array(pil_image)
    transform = get_val_transforms(224)
    tensor    = transform(image=img_np)["image"].unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    conf      = float(probs[pred_idx])

    # Grad-CAM
    target_layer = get_target_layer(model, model_name)
    gc  = GradCAM(model, target_layer)
    cam = gc(tensor, pred_idx)
    gc.remove()

    img_224  = np.array(pil_image.resize((224, 224)))
    overlay  = make_overlay(img_224, cam)

    return pred_name, conf, probs, cam, overlay


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="DermaScan — Skin Lesion Detection",
        page_icon="🔬",
        layout="wide"
    )

    # Header
    st.markdown("""
    <h1 style='text-align:center; color:#2C7BE5;'>🔬 DermaScan</h1>
    <h4 style='text-align:center; color:gray;'>Lightweight Deep Learning for Skin Lesion Detection</h4>
    <hr>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.warning("⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist.")

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(CHECKPOINTS.keys()),
        index=0
    )
    use_preprocess = st.sidebar.checkbox(
        "Apply Preprocessing",
        value=True
    )
    show_all_probs = st.sidebar.checkbox(
        "Show All Class Probabilities",
        value=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    model_info = {
        "EfficientNet-B0 (Best Accuracy)":  "4.22M params | AUC 0.9426",
        "MobileNetV3 (Balanced)":           "3.09M params | AUC 0.9322",
        "ShuffleNetV2 (Fastest)":           "1.39M params | AUC 0.9352",
    }
    st.sidebar.info(model_info[model_choice])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Made by:** Anmol Krishna")
    st.sidebar.markdown("**Institute:** KIIT University")
    st.sidebar.markdown("[GitHub](https://github.com/Krishn4nmol/DermaScan)")

    # Load model
    model_key = MODEL_KEYS[model_choice]
    ckpt_path = CHECKPOINTS[model_choice]

    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found: {ckpt_path}\nPlease train the model first.")
        return

    model = load_model(model_key, ckpt_path)

    # Upload
    st.markdown("### 📤 Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "Choose a dermoscopic image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("**Uploaded Image**")
            st.image(pil_image, use_column_width=True)

        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                pred_name, conf, probs, cam, overlay = predict(
                    model, model_key, pil_image, use_preprocess
                )

            info = CLASS_INFO[pred_name]

            st.markdown("---")
            st.markdown("### 🎯 Result")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Predicted Class", pred_name)
            with r2:
                st.metric("Confidence", f"{conf*100:.1f}%")
            with r3:
                st.metric("Risk Level", f"{info['color']} {info['risk']}")

            st.info(f"ℹ️ **About {pred_name}:** {info['desc']}")

            st.markdown("### 🔥 Grad-CAM Visualization")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(pil_image.resize((224, 224)), use_column_width=True)
            with c2:
                st.markdown("**Attention Heatmap**")
                cam_colored = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
                st.image(cam_colored, use_column_width=True)
            with c3:
                st.markdown("**Overlay**")
                st.image(overlay, use_column_width=True)

            if show_all_probs:
                st.markdown("### 📊 All Class Probabilities")
                for name, prob in zip(CLASS_NAMES, probs):
                    info_i = CLASS_INFO[name]
                    st.progress(float(prob),
                                text=f"{info_i['color']} {name}: {prob*100:.1f}%")

    else:
        st.markdown("### 📋 Detectable Conditions")
        cols = st.columns(4)
        for i, (name, info) in enumerate(CLASS_INFO.items()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style='padding:10px; border-radius:8px;
                border:1px solid #ddd; margin:5px;'>
                <b>{info['color']} {name}</b><br>
                <small>Risk: {info['risk']}</small>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='text-align:center; color:gray; font-size:12px;'>
    DermaScan | Anmol Krishna | KIIT University |
    Trained on HAM10000 Dataset |
    <a href='https://github.com/Krishn4nmol/DermaScan'>GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()