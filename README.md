# DeepFER - Deep Facial Emotion Recognition

DeepFER is a deep learning-based Facial Emotion Recognition system built with **TensorFlow/Keras**, **OpenCV**, and **Streamlit**. It processes face images and classifies facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## 📁 Project Architecture & Structure

```
DeepFER/
├── app/
│   └── streamlit_app.py      # Streamlit Web UI Dashboard
├── data/
│   ├── raw/                  # Dataset directory (train/ and test/ folders)
│   └── processed/            # Processed dataset directory
├── models/
│   └── emotion_model.keras   # Saved Keras model binary
├── src/
│   ├── __init__.py           # Package init with root sys.path resolution
│   ├── config.py             # Global project configuration & constants
│   ├── face_detector.py      # OpenCV Haar Cascade face detection pipeline
│   ├── model.py              # CNN Architecture (Conv2D + BatchNorm + Dropout)
│   ├── predict.py            # Prediction pipeline & CLI runner
│   └── train.py              # Multithreaded fast model training pipeline
├── utils/
│   ├── __init__.py           # Utils package initialization
│   └── visualization.py      # Charting & probability visualization helpers
├── pyproject.toml            # Project configuration for uv package manager
├── requirements.txt          # Python dependencies list
└── README.md                 # Project documentation & execution guide
```

---

## ⚡ Fast Setup & Execution using `uv` in VS Code

### 1. Install & Sync Virtual Environment with `uv`
Run the following command in your VS Code Terminal (`Ctrl + ~`):

```bash
uv sync
```

This will automatically create a virtual environment (`.venv`) and install all required dependencies specified in `pyproject.toml`.

---

### 2. Configure Python Interpreter in VS Code
1. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) to open the Command Palette.
2. Search and select: **Python: Select Interpreter**.
3. Choose the interpreter located inside the `.venv` directory:
   `.\.venv\Scripts\python.exe` (Windows) or `./.venv/bin/python` (Linux/macOS).

---

### 3. Run the Applications

#### Option A: Launch Streamlit Web App UI
```bash
uv run streamlit run app/streamlit_app.py
```

#### Option B: Perform Image Emotion Prediction (CLI)
```bash
uv run python src/predict.py data/raw/test/happy/im0.jpg
```

#### Option C: Train / Re-train the CNN Model
```bash
uv run python src/train.py
```

---

## 🧠 Model Architecture

The DeepFER CNN model consists of:
- **3 Convolutional Blocks**: Dual `Conv2D(3x3)` layers per block with `ReLU` activations, `BatchNormalization`, `MaxPooling2D(2x2)`, and `Dropout(0.25)`.
- **Dense Classifier**: Flatten layer, `Dense(256)` with `BatchNormalization` and `Dropout(0.5)`, followed by a 7-class `Softmax` output.
- **Optimizer**: Adam with learning rate reduction on plateau and early stopping callbacks.
