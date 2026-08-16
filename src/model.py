import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf

from src.config import (
    IMAGE_SIZE,
    CHANNELS,
    NUM_CLASSES,
    USE_DATA_AUGMENTATION,
    AUG_ROTATION_FACTOR,
    AUG_ZOOM_FACTOR,
    AUG_TRANSLATION_FACTOR,
    DEFAULT_MODEL_TYPE,
)


def build_data_augmentation():
    """Creates a Keras Sequential layer for real-time Data Augmentation (Rotation, Flipping, Zooming, Translation)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(AUG_ROTATION_FACTOR),
            tf.keras.layers.RandomZoom(AUG_ZOOM_FACTOR),
            tf.keras.layers.RandomTranslation(
                AUG_TRANSLATION_FACTOR, AUG_TRANSLATION_FACTOR
            ),
        ],
        name="data_augmentation",
    )


def build_custom_cnn(use_augmentation=True):
    """Builds a deep Custom CNN architecture with optional Data Augmentation, BatchNorm, and Dropout."""
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    x = inputs
    if use_augmentation:
        augmentation = build_data_augmentation()
        x = augmentation(x)

    # Conv Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Conv Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Conv Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Dense Classification Block
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DeepFER_Custom_CNN")


def build_transfer_learning_model(use_augmentation=True, fine_tune=False):
    """Builds a Transfer Learning architecture using MobileNetV2 pre-trained on ImageNet with fine-tuning support."""
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    x = inputs
    if use_augmentation:
        augmentation = build_data_augmentation()
        x = augmentation(x)

    # Convert 1-channel grayscale to 3-channel RGB for MobileNetV2
    if CHANNELS == 1:
        x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])

    # Resize input to minimum compatible size for MobileNetV2 (96x96)
    x = tf.keras.layers.Resizing(96, 96)(x)

    # Pre-trained MobileNetV2 Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model weights unless fine-tuning is enabled
    base_model.trainable = fine_tune

    x = base_model(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DeepFER_MobileNetV2_Transfer")


def build_model(model_type=DEFAULT_MODEL_TYPE, use_augmentation=USE_DATA_AUGMENTATION, fine_tune=False):
    """Factory function to build and compile the requested emotion recognition model architecture."""
    if model_type.lower() == "transfer":
        model = build_transfer_learning_model(use_augmentation=use_augmentation, fine_tune=fine_tune)
        lr = 1e-4 if fine_tune else 1e-3
    else:
        model = build_custom_cnn(use_augmentation=use_augmentation)
        lr = 1e-3

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model