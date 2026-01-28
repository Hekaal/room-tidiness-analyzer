import sys
import streamlit as st

st.write("Python version:", sys.version)
import cv2, numpy as np

from detector.model import load_model
from detector.inference import detect_objects
from analysis.features import extract_features
from analysis.scoring import compute_tidiness_score
from utils.visualization import draw_detections, draw_zones

st.set_page_config("Room Tidiness Analyzer", layout="wide")
st.title("🧹 Room Tidiness Analyzer")

@st.cache_resource
def load_yolo():
    return load_model()

model = load_yolo()

mode = st.sidebar.radio("Mode", ["Image", "Video (Webcam)"])

# ================= IMAGE MODE =================
if mode == "Image":
    file = st.file_uploader("Upload image", ["jpg","png","jpeg"])
    if file:
        data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        detections = detect_objects(model, image)
        features = extract_features(detections, image)
        result = compute_tidiness_score(features)

        vis = draw_zones(image.copy())
        vis = draw_detections(vis, detections)

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.metric("Skor Kerapian", result["score"], result["label"])
        st.write(result["explanation"])
        st.json(features)

# ================= VIDEO MODE =================
if mode == "Video (Webcam)":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(model, frame)
        features = extract_features(detections, frame)
        result = compute_tidiness_score(features)

        frame = draw_zones(frame)
        frame = draw_detections(frame, detections)

        cv2.putText(frame, f"{result['label']} ({result['score']})",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
