import streamlit as st
import numpy as np
from PIL import Image

from detector.model import load_model
from detector.inference import detect_objects
from analysis.features import extract_features
from analysis.scoring import compute_tidiness_score
from utils.visualization import draw_visuals

st.set_page_config("Room Tidiness Analyzer", layout="wide")
st.title("🧹 Room Tidiness Analyzer")

@st.cache_resource
def load_yolo():
    return load_model()

model = load_yolo()

uploaded = st.file_uploader("Upload image", ["jpg", "png", "jpeg"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_np = np.array(pil_img)

    detections = detect_objects(model, img_np)
    features = extract_features(detections, img_np)
    result = compute_tidiness_score(features)

    vis_img = draw_visuals(pil_img.copy(), detections)

    st.image(vis_img, use_column_width=True)
    st.metric("Skor Kerapian", result["score"], result["label"])
    st.write(result["explanation"])
    st.json(features)
