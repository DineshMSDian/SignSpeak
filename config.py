"""
Central Configuration Module
Real-Time Continuous Hybrid Sign Language Translation System
"""

import os

# Suppress TensorFlow info/warning logs globally
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── MediaPipe Hand Landmarks ──────────────────────────────────
NUM_LANDMARKS = 21          # Landmarks per hand
NUM_HANDS = 2               # Max hands to track
NUM_COORDS = 3              # x, y, z per landmark
FEATURE_DIM = NUM_LANDMARKS * NUM_HANDS * NUM_COORDS  # 126

# ─── Sequence Parameters ───────────────────────────────────────
SEQUENCE_LENGTH = 60        # Frames per sequence

# ─── Model Parameters ──────────────────────────────────────────
LSTM_UNITS_1 = 128          # First LSTM layer units
LSTM_UNITS_2 = 64           # Second LSTM layer units
DENSE_UNITS = 64            # Dense layer units
DROPOUT_RATE = 0.3          # Dropout rate
LEARNING_RATE = 0.001       # Adam optimizer LR
EPOCHS = 100                # Max training epochs
BATCH_SIZE = 32             # Training batch size
EARLY_STOP_PATIENCE = 10   # Early stopping patience
LR_REDUCE_PATIENCE = 5     # ReduceLROnPlateau patience
LR_REDUCE_FACTOR = 0.5     # LR reduction factor

# ─── Prediction Parameters ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for prediction
DEBOUNCE_FRAMES = 15        # Frames to skip after a prediction

# ─── MediaPipe Detection ───────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ─── Webcam ─────────────────────────────────────────────────────
CAMERA_INDEX = 0            # Default webcam

# ─── Paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
ASL_RAW_DIR = os.path.join(RAW_DATA_DIR, "ASL")
ISL_RAW_DIR = os.path.join(RAW_DATA_DIR, "ISL")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "sign_lstm_best.keras")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "label_map.json")

# ─── Split Model Paths (ASL / ISL) ────────────────────────────
ASL_MODEL_PATH = os.path.join(MODELS_DIR, "sign_lstm_asl.keras")
ASL_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map_asl.json")
ISL_MODEL_PATH = os.path.join(MODELS_DIR, "sign_lstm_isl.keras")
ISL_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map_isl.json")

# ─── ASL Letter Labels ────────────────────────────────────────
ASL_LETTERS = set("abcdefghijklmnopqrstuvwxyz")

# ─── TTS ────────────────────────────────────────────────────────
TTS_RATE = 150              # Words per minute
TTS_ENABLED_DEFAULT = True  # TTS on by default

# ─── Supported Languages ───────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "ISL": "Indian Sign Language",
    "ASL": "American Sign Language",
}
DEFAULT_LANGUAGE = "ISL"

# ─── Create directories on import ──────────────────────────────
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, DOCS_DIR]:
    os.makedirs(_dir, exist_ok=True)
