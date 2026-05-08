"""
Evaluation Script
─────────────────
Computes accuracy, precision, recall, F1-score, confusion matrix,
and inference latency for the trained LSTM model.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import time
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def evaluate():
    """Run full model evaluation pipeline."""
    print("=" * 60)
    print("  SignSpeak — Model Evaluation")
    print("=" * 60)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[ERROR] No trained model found at {config.MODEL_SAVE_PATH}")
        print("[INFO] Run 'python scripts/train.py' first!")
        return

    print(f"[INFO] Loading model from {config.MODEL_SAVE_PATH}...")
    model = load_model(config.MODEL_SAVE_PATH)
    print("[INFO] Model loaded successfully.")

    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, "y_test.npy"))

    with open(config.LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    reverse_map = {int(v): k for k, v in label_map.items()}
    class_names = [reverse_map[i] for i in range(len(reverse_map))]

    print(f"\n[INFO] Test samples: {X_test.shape[0]}")
    print(f"[INFO] Classes: {class_names}")

    print("\n[INFO] Running predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "─" * 40)
    print("  EVALUATION METRICS")
    print("─" * 40)
    print(f"  Accuracy:  {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("─" * 40)

    print("\n  Per-Class Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.7)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — SignSpeak LSTM", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(config.LOGS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    print("\n[INFO] Measuring inference latency (100 runs)...")
    sample = X_test[:1]
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_latency = np.mean(times)
    p95_latency = np.percentile(times, 95)
    p99_latency = np.percentile(times, 99)

    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  P95 latency:     {p95_latency:.2f} ms")
    print(f"  P99 latency:     {p99_latency:.2f} ms")
    print(f"  Max latency:     {max(times):.2f} ms")

    report = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "p99_latency_ms": float(p99_latency),
        "num_test_samples": int(X_test.shape[0]),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    report_path = os.path.join(config.LOGS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Evaluation report saved → {report_path}")

    print("\n[DONE] Evaluation complete!")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
