import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODEL_PATH, CONFUSION_MATRIX_PATH, CLASS_NAMES
from src.train import load_dataset_fast
from utils.visualization import plot_confusion_matrix


def evaluate_model():
    print("=" * 60, flush=True)
    print("      DeepFER - Model Evaluation & Metrics Report", flush=True)
    print("=" * 60, flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}. Please train a model first using `python src/train.py`.")

    print(f"Loading trained Keras model from: {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading evaluation dataset...", flush=True)
    _, _, X_val, y_val = load_dataset_fast()

    print(f"Evaluating model on {len(X_val)} validation samples...", flush=True)
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n-> Overall Validation Loss:     {test_loss:.4f}", flush=True)
    print(f"-> Overall Validation Accuracy: {test_acc * 100:.2f}%\n", flush=True)

    # Generate Predictions
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Classification Report (Precision, Recall, F1-score)
    print("-" * 60, flush=True)
    print("Classification Report (Precision, Recall, F1-Score per Emotion Class):", flush=True)
    print("-" * 60, flush=True)
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    print(report, flush=True)

    # Confusion Matrix
    print("-" * 60, flush=True)
    print("Confusion Matrix:", flush=True)
    print("-" * 60, flush=True)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm, flush=True)

    # Save Confusion Matrix Plot
    print(f"\nGenerating and saving Confusion Matrix plot to: {CONFUSION_MATRIX_PATH}", flush=True)
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=CONFUSION_MATRIX_PATH)
    print("Evaluation completed successfully!", flush=True)
    print("=" * 60, flush=True)


    return test_loss, test_acc, report, cm


if __name__ == "__main__":
    evaluate_model()
