"""
Central Configuration Module
Real-Time Continuous Hybrid Sign Language Translation System
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_LANDMARKS = 21
NUM_HANDS = 2
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_HANDS * NUM_COORDS

SEQUENCE_LENGTH = 60

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5

CONFIDENCE_THRESHOLD = 0.3
DEBOUNCE_FRAMES = 15

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

CAMERA_INDEX = 0

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

ASL_MODEL_PATH = os.path.join(MODELS_DIR, "sign_lstm_asl.keras")
ASL_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map_asl.json")
ISL_MODEL_PATH = os.path.join(MODELS_DIR, "sign_lstm_isl.keras")
ISL_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map_isl.json")

ASL_LETTERS = set("abcdefghijklmnopqrstuvwxyz")

TTS_RATE = 150
TTS_ENABLED_DEFAULT = True

SUPPORTED_LANGUAGES = {
    "ISL": "Indian Sign Language",
    "ASL": "American Sign Language",
}
DEFAULT_LANGUAGE = "ISL"

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, DOCS_DIR]:
    os.makedirs(_dir, exist_ok=True)
