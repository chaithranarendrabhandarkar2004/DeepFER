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
from src.config import CLASS_NAMES, CONFUSION_MATRIX_PATH, MODEL_PATH
from utils.visualization import plot_emotion_probabilities


st.set_page_config(
    page_title="DeepFER - Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 DeepFER - Facial Emotion Recognition System")
st.markdown("### Deep Learning for Computer Vision: Real-Time FER & Transfer Learning")

# Load model
@st.cache_resource
def get_model():
    return load_emotion_model()


try:
    model = get_model()
except Exception as e:
    st.error("Emotion recognition model could not be loaded. Please ensure the model file exists.")
    st.code(str(e))
    st.stop()


# Helper function to process and display prediction results
def process_and_display_image(image_input):
    image_array = np.array(image_input)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    faces = detect_faces(image_bgr)
    image_with_boxes = image_array.copy()

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Facial Input & Face Detection")
        st.image(image_with_boxes, caption="Detected Bounding Box(es)", use_container_width=True)

    with col2:
        st.subheader("Emotion Recognition Results")
        if len(faces) == 0:
            st.warning("Haar Cascade face detector did not isolate a face. Analyzing complete image frame...")
            h_img, w_img = image_bgr.shape[:2]
            faces = [(0, 0, w_img, h_img)]
        else:
            st.success(f"Detected {len(faces)} face region(s).")

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = image_bgr[y:y+h, x:x+w]
            emotion, confidence, predictions = predict_emotion(model, face_crop)

            if len(faces) > 1:
                st.markdown(f"#### Face #{i + 1}")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Predicted Emotion", emotion)
            with m_col2:
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            st.markdown("**Emotion Confidence Distribution:**")
            fig = plot_emotion_probabilities(predictions, CLASS_NAMES)
            st.pyplot(fig)


# Interface Tabs
tab1, tab2, tab3 = st.tabs([
    "📁 Upload Image",
    "📷 Real-Time Camera Stream",
    "📊 Model Architecture & Evaluation Metrics"
])

with tab1:
    st.subheader("Upload Image for Facial Emotion Detection")
    uploaded_file = st.file_uploader("Select an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        process_and_display_image(image)

with tab2:
    st.subheader("Real-Time Camera / Webcam Input")
    st.write("Capture a image from your camera to perform instant real-time emotion recognition.")
    camera_file = st.camera_input("Capture Face Image from Camera", key="camera")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
        process_and_display_image(image)

with tab3:
    st.subheader("Model Architecture & Technical Specifications")
    
    st.markdown("""
    #### 🧠 Implemented Deep Learning Methods:
    - **CNN & Transfer Learning Architectures**: Supports custom multi-block CNN and pre-trained **MobileNetV2** backbones.
    - **Data Augmentation**: Real-time augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`).
    - **Evaluation Metrics**: Full **Precision**, **Recall**, and **F1-Score** evaluation across all 7 emotion categories.
    - **Face Detection**: Multi-pass OpenCV Haar Cascade pipeline.
    """)

    st.divider()

    st.subheader("Model Confusion Matrix Heatmap")
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Saved Model Confusion Matrix", use_container_width=True)
    else:
        st.info("No saved Confusion Matrix plot found. Run `uv run python src/evaluate.py` or `uv run python src/train.py` to generate evaluation plots.")