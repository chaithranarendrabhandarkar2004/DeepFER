import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.predict import load_emotion_model, predict_emotion
from src.face_detector import detect_faces
from src.config import CLASS_NAMES
from utils.visualization import plot_emotion_probabilities


st.set_page_config(
    page_title="DeepFER - Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 DeepFER - Facial Emotion Recognition")
st.markdown("### Powered by Deep Convolutional Neural Networks & OpenCV")
st.write(
    "Upload a facial image below to perform real-time facial emotion detection and analysis."
)


@st.cache_resource
def get_model():
    return load_emotion_model()


try:
    model = get_model()
except Exception as e:
    st.error("Emotion model could not be loaded.")
    st.code(str(e))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a face image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    faces = detect_faces(image_bgr)

    image_with_boxes = image_array.copy()
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_with_boxes, caption="Detected Face Bounding Box(es)", use_container_width=True)

    with col2:
        st.subheader("Emotion Analysis Results")
        if len(faces) == 0:
            st.warning(
                "Haar Cascade did not detect a distinct face box. Analyzing full image..."
            )
            h_img, w_img = image_bgr.shape[:2]
            faces = [(0, 0, w_img, h_img)]
        else:
            st.success(f"Successfully detected {len(faces)} face(s).")

        for i, (x, y, w, h) in enumerate(faces):
            face = image_bgr[y:y+h, x:x+w]
            emotion, confidence, predictions = predict_emotion(model, face)

            st.markdown(f"#### Face #{i + 1}")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Predicted Emotion", emotion)
            with m_col2:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.markdown("**Emotion Probability Distribution:**")
            fig = plot_emotion_probabilities(predictions, CLASS_NAMES)
            st.pyplot(fig)