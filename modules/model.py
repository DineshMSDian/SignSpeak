"""
Module 4: LSTM Temporal Model
─────────────────────────────
Multi-layer LSTM architecture for temporal gesture classification.
Input: (sequence_length, feature_dim) = (60, 126)
Output: softmax probability over gesture classes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    num_classes: int,
    sequence_length: int = config.SEQUENCE_LENGTH,
    feature_dim: int = config.FEATURE_DIM,
    lstm_units_1: int = config.LSTM_UNITS_1,
    lstm_units_2: int = config.LSTM_UNITS_2,
    dense_units: int = config.DENSE_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile a multi-layer LSTM model for gesture classification.

    Architecture:
        Input(60, 126)
        → LSTM(128, return_sequences=True) → Dropout(0.3)
        → LSTM(64) → Dropout(0.3)
        → Dense(64, ReLU) → Dropout(0.3)
        → Dense(num_classes, Softmax)

    Args:
        num_classes: Number of gesture classes to classify.
        sequence_length: Number of frames per sequence (default 60).
        feature_dim: Dimension of each frame vector (default 126).

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    model = Sequential([
        Input(shape=(sequence_length, feature_dim)),

        # First LSTM layer — returns sequences for stacking
        LSTM(lstm_units_1, return_sequences=True),
        Dropout(dropout_rate),

        # Second LSTM layer — returns final hidden state
        LSTM(lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),

        # Dense classification head
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),

        # Output layer
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_trained_model(model_path: str = config.MODEL_SAVE_PATH):
    """
    Load a previously trained model from disk.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded tf.keras.Model, or None if file doesn't exist.
    """
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found at {model_path}")
        return None
    model = keras_load_model(model_path)
    print(f"[INFO] Model loaded from {model_path}")
    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Return model summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def get_training_callbacks(
    model_save_path: str = config.MODEL_SAVE_PATH,
    log_dir: str = config.LOGS_DIR,
) -> list:
    """
    Get standard training callbacks.

    Returns:
        List of Keras callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return callbacks


# ── Standalone Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Building LSTM Model")
    model = build_lstm_model(num_classes=10)
    print(get_model_summary(model))
    print(f"\n[INFO] Input shape:  {model.input_shape}")
    print(f"[INFO] Output shape: {model.output_shape}")
    print(f"[INFO] Parameters:   {model.count_params():,}")
    print("[PASS] Model built successfully!")
