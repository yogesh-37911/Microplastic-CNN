"""Streamlit screening interface backed by the project's PyTorch inference service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.inference import get_inference_service


st.set_page_config(page_title="Microplastic Vision", page_icon="🔬", layout="wide")
st.title("🔬 Microplastic Vision")
st.caption("Microscopy image screening · experimental classifier")

service = get_inference_service()
if not service.is_loaded:
    st.error("No provenance-checked model checkpoint is available. Classification is paused.")
elif not service.is_ready:
    st.warning("The loaded model has not met the validation threshold. Class predictions are withheld.")
else:
    st.info("Model is ready according to its held-out score. Results are still screening suggestions, not chemical identification.")

st.markdown(
    "**Training scope:** the available labeled dataset covers microfibres in microscope filter crops. "
    "It has not been validated for stock photos, illustrations, product images, or every microplastic morphology."
)

uploads = st.file_uploader(
    "Upload microscopy images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)
if uploads and st.button("Analyze images", type="primary"):
    for uploaded in uploads:
        image = Image.open(uploaded).convert("RGB")
        result = service.predict(image)
        st.session_state.setdefault("analysis_history", []).insert(
            0,
            {
                "name": uploaded.name,
                "image": image.copy(),
                "result": result,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

history = st.session_state.setdefault("analysis_history", [])
st.subheader(f"Recent analysis ({len(history)})")
if history:
    if st.button("Clear analysis history"):
        st.session_state.analysis_history = []
        st.rerun()
    for index, item in enumerate(history):
        with st.container(border=True):
            col_image, col_result = st.columns([1, 3])
            col_image.image(item["image"], use_container_width=True)
            result = item["result"]
            with col_result:
                st.markdown(f"**{item['name']}** · {item['time']}")
                if result["predicted_class"]:
                    st.subheader(result["predicted_class"].replace("_", " ").title())
                    if result["confidence"] is not None:
                        st.metric("Model score", f"{result['confidence']:.1%}")
                else:
                    labels = {
                        "model_requires_validation": "Model needs validation — no class predicted",
                        "model_unavailable": "Model unavailable — no class predicted",
                        "out_of_domain": "Outside training image scope — no class predicted",
                        "uncertain": "Uncertain — no class predicted",
                    }
                    st.warning(labels.get(result["status"], "No class prediction"))
                for warning in result["warnings"]:
                    st.caption(f"⚠️ {warning}")
                if st.button("Delete", key=f"delete-{index}"):
                    history.pop(index)
                    st.rerun()
else:
    st.caption("Your analyzed images will appear here for this session.")
