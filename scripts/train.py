"""
Training Script
───────────────
Loads processed dataset, one-hot encodes labels,
trains the LSTM model with callbacks, and saves results.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.utils import to_categorical
from modules.dataset_manager import DatasetManager
from modules.model import build_lstm_model, get_training_callbacks, get_model_summary

def plot_training_history(history, save_dir: str = config.LOGS_DIR):
    """Save training/validation accuracy and loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Training history plot saved → {path}")

def train():
    """Main training pipeline."""
    print("=" * 60)
    print("  SignSpeak — LSTM Model Training")
    print("=" * 60)

    dm = DatasetManager()

    processed_files = [
        os.path.join(config.PROCESSED_DATA_DIR, f)
        for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    ]

    if all(os.path.exists(f) for f in processed_files):
        print("[INFO] Loading existing processed dataset...")
        X_train, X_test, y_train, y_test, label_map = dm.load_processed()
    else:
        print("[INFO] Building dataset from raw data...")
        X_train, X_test, y_train, y_test, label_map = dm.build_dataset()

    num_classes = len(label_map)
    print(f"\n[INFO] Dataset Summary:")
    print(f"  Training samples:   {X_train.shape[0]}")
    print(f"  Testing samples:    {X_test.shape[0]}")
    print(f"  Sequence length:    {X_train.shape[1]}")
    print(f"  Feature dimension:  {X_train.shape[2]}")
    print(f"  Number of classes:  {num_classes}")
    print(f"  Classes: {list(label_map.keys())}")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print(f"\n[INFO] Building LSTM model...")
    model = build_lstm_model(num_classes=num_classes)
    print(get_model_summary(model))

    print(f"\n[INFO] Training for max {config.EPOCHS} epochs...")
    print(f"  Batch size:         {config.BATCH_SIZE}")
    print(f"  Learning rate:      {config.LEARNING_RATE}")
    print(f"  Early stop patience: {config.EARLY_STOP_PATIENCE}")
    print()

    callbacks = get_training_callbacks()

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 60)
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Train Accuracy: {train_acc:.4f}  |  Train Loss: {train_loss:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}  |  Test Loss:  {test_loss:.4f}")
    print("=" * 60)

    plot_training_history(history)

    print(f"\n[INFO] Best model saved → {config.MODEL_SAVE_PATH}")
    print("[DONE] Training complete!")

if __name__ == "__main__":
    train()
