import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model import build_model
from src.config import (
    MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    RAW_DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    DEFAULT_MODEL_TYPE,
    USE_DATA_AUGMENTATION,
)
from utils.visualization import plot_confusion_matrix


def load_dataset_fast():
    train_dir = RAW_DATA_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    class_map = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}

    def process_file(file_info):
        img_path, label_idx = file_info
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img, label_idx

    file_list = []
    for class_folder in train_dir.iterdir():
        if not class_folder.is_dir():
            continue
        folder_name = class_folder.name.lower()
        if folder_name not in class_map:
            continue
        label_idx = class_map[folder_name]
        for img_path in class_folder.glob("*.jpg"):
            if img_path.name.startswith("._"):
                continue
            file_list.append((img_path, label_idx))

    if not file_list:
        raise ValueError(
            f"No .jpg files found in {train_dir}. Please place dataset subfolders inside {train_dir}."
        )

    print(f"Parallel loading {len(file_list)} valid face images into memory...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_file, file_list))

    images, labels = [], []
    for img, lbl in results:
        if img is not None:
            images.append(img)
            labels.append(lbl)

    if not images:
        raise ValueError("Failed to load valid image data from dataset.")

    X = np.array(images, dtype=np.float32).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"Dataset ready: {len(X_train)} training samples, {len(X_val)} validation samples.", flush=True)
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description="DeepFER Model Training Pipeline")
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["custom", "transfer"],
        help="Architecture type: 'custom' (CNN) or 'transfer' (MobileNetV2)",
    )
    parser.add_argument(
        "--no_aug",
        action="store_true",
        help="Disable real-time Data Augmentation",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Enable fine-tuning for Transfer Learning backbones",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )

    args = parser.parse_args()
    use_aug = not args.no_aug

    print(f"\n--- Initializing DeepFER Model Training ---")
    print(f"Model Type:        {args.model_type.upper()}")
    print(f"Data Augmentation: {use_aug}")
    print(f"Fine Tuning:       {args.fine_tune}\n")

    X_train, y_train, X_val, y_val = load_dataset_fast()

    model = build_model(
        model_type=args.model_type,
        use_augmentation=use_aug,
        fine_tune=args.fine_tune
    )
    model.summary()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("\nStarting model training...", flush=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=128,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks,
    )

    print("\n--- Evaluating Model & Computing Full Metrics ---", flush=True)
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss:     {test_loss:.4f}", flush=True)
    print(f"Validation Accuracy: {test_accuracy * 100:.2f}%\n", flush=True)

    # Classification Report & Confusion Matrix
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("=" * 60)
    print("Classification Report (Precision, Recall, F1-Score):")
    print("=" * 60)
    report = classification_report(
        y_true_classes, y_pred_classes, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=CONFUSION_MATRIX_PATH)
    print(f"Confusion Matrix plot saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()