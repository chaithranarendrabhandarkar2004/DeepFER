from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "emotion_model.keras"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

IMAGE_SIZE = 48
CHANNELS = 1

NUM_CLASSES = 7

# Model Architecture Config: 'custom' or 'transfer' (MobileNetV2)
DEFAULT_MODEL_TYPE = "custom"

# Data Augmentation parameters
USE_DATA_AUGMENTATION = True
AUG_ROTATION_FACTOR = 0.15
AUG_ZOOM_FACTOR = 0.15
AUG_TRANSLATION_FACTOR = 0.1

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]