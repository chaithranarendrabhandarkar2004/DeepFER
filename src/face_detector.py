import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2

CASCADES = [
    cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ),
    cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    ),
    cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    ),
]


def detect_faces(image):
    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_eq = cv2.equalizeHist(gray)

    for cascade in CASCADES:
        if cascade.empty():
            continue

        faces = cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return faces

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return faces

    return []