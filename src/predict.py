import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    MODEL_PATH
)


def load_emotion_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Warning: Failed to load existing model ({e}). Rebuilding new model...")

    from src.model import build_model
    model = build_model()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    return model


def predict_emotion(model, face):
    if face is None or face.size == 0:
        return CLASS_NAMES[0], 0.0, np.zeros(len(CLASS_NAMES))

    if len(face.shape) == 3 and face.shape[2] == 3:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    elif len(face.shape) == 3 and face.shape[2] == 1:
        gray = face[:, :, 0]
    else:
        gray = face

    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    predictions = model.predict(gray, verbose=0)[0]
    index = int(np.argmax(predictions))
    emotion = CLASS_NAMES[index]
    confidence = float(predictions[index])

    return emotion, confidence, predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepFER Emotion Prediction CLI")
    parser.add_argument("image_path", type=str, nargs="?", help="Path to image file for prediction")
    args = parser.parse_args()

    print("Loading emotion recognition model...", flush=True)
    model = load_emotion_model()

    if args.image_path:
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"Error: File not found at '{img_path}'", flush=True)
            sys.exit(1)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Failed to read image file '{img_path}'", flush=True)
            sys.exit(1)

        from src.face_detector import detect_faces
        faces = detect_faces(image)

        if len(faces) == 0:
            print("No distinct face bounding box detected. Analyzing whole image...", flush=True)
            h, w = image.shape[:2]
            faces = [(0, 0, w, h)]
        else:
            print(f"Detected {len(faces)} face(s) in image.", flush=True)

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = image[y:y+h, x:x+w]
            emotion, confidence, probabilities = predict_emotion(model, face_crop)
            print(f"\n--- Result for Face {i + 1} ---", flush=True)
            print(f"Predicted Emotion : {emotion}", flush=True)
            print(f"Confidence Level  : {confidence * 100:.2f}%", flush=True)
            print("Class Probabilities:", flush=True)
            for name, prob in zip(CLASS_NAMES, probabilities):
                print(f"  - {name:<10}: {prob * 100:.2f}%", flush=True)
    else:
        print("Model standard check successful! Pass an image path to predict emotion.", flush=True)


if __name__ == "__main__":
    main()