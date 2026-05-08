"""
Training Script — Split ASL / ISL Models
─────────────────────────────────────────
Trains two separate LSTM models from the reorganized data:
  data/raw/ASL/a, b, c, ... z     → models/sign_lstm_asl.keras
  data/raw/ISL/hello, bye, ...    → models/sign_lstm_isl.keras

Usage:
  python scripts/train_split.py
  python scripts/train_split.py --asl
  python scripts/train_split.py --isl
"""

import os
import sys
import json
import argparse
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from modules.model import build_lstm_model

def load_data_from_dir(data_dir):
    """Load all sequences from a directory. Each subfolder = one class."""
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not labels:
        raise ValueError(f"No gesture folders found in {data_dir}")

    label_map = {label: idx for idx, label in enumerate(labels)}
    X_all, y_all = [], []

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        files = sorted(glob.glob(os.path.join(label_dir, "*.npy")))

        for filepath in files:
            seq = np.load(filepath)
            if seq.shape == (config.SEQUENCE_LENGTH, config.FEATURE_DIM):
                X_all.append(seq)
                y_all.append(label_map[label])
            else:
                print(f"  [WARN] Skipping {filepath}: shape {seq.shape}")

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int32), label_map

def train_model(mode):
    """Train a single model for the given mode ('asl' or 'isl')."""
    mode_upper = mode.upper()

    if mode == "asl":
        data_dir = config.ASL_RAW_DIR
        model_path = config.ASL_MODEL_PATH
        label_map_path = config.ASL_LABEL_MAP_PATH
    else:
        data_dir = config.ISL_RAW_DIR
        model_path = config.ISL_MODEL_PATH
        label_map_path = config.ISL_LABEL_MAP_PATH

    print(f"\n{'=' * 60}")
    print(f"  Training {mode_upper} Model")
    print(f"  Data directory: {data_dir}")
    print(f"{'=' * 60}")

    X_all, y_all, label_map = load_data_from_dir(data_dir)
    num_classes = len(label_map)
    print(f"\n[INFO] Labels ({num_classes}): {list(label_map.keys())}")
    print(f"[INFO] Total samples: {len(X_all)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    model = build_lstm_model(num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE, min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\n[INFO] Training for max {config.EPOCHS} epochs...\n")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  {mode_upper} Results:")
    print(f"  Train Accuracy: {train_acc:.4f}  |  Loss: {train_loss:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}  |  Loss: {test_loss:.4f}")
    print(f"{'=' * 60}")

    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] Model saved → {model_path}")
    print(f"[INFO] Label map saved → {label_map_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train split ASL/ISL models")
    parser.add_argument("--asl", action="store_true", help="Train ASL model only")
    parser.add_argument("--isl", action="store_true", help="Train ISL model only")
    args = parser.parse_args()

    if not args.asl and not args.isl:
        args.asl = True
        args.isl = True

    if args.asl:
        train_model("asl")
    if args.isl:
        train_model("isl")

    print("\n[DONE] Training complete!")
