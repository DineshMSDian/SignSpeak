"""
Data Collection Script
──────────────────────
Interactive CLI tool for recording gesture sequences.
Captures 60-frame landmark sequences via webcam + MediaPipe,
normalizes them, and saves via DatasetManager.
"""

import os
import sys
import cv2
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from modules.capture_engine import HandCapture
from modules.normalization import normalize_landmarks, SequenceBuffer
from modules.dataset_manager import DatasetManager

def collect_data():
    """
    Interactive gesture data collection.

    Controls:
        s — Start / stop recording a sequence
        q — Quit the current gesture and return to menu
    """
    dm = DatasetManager()
    hc = HandCapture()

    print("=" * 60)
    print("  SignSpeak — Gesture Data Collection Tool")
    print("=" * 60)

    while True:
        print("\n── Menu ──────────────────────────────────────")
        print("  Current gestures:", dm.get_gesture_labels() or "(none)")
        print("  Distribution:", dm.get_class_distribution() or "(empty)")
        print()

        gesture = input("Enter gesture name (or 'quit' to exit): ").strip().lower()
        if gesture in ("quit", "q", "exit"):
            break
        if not gesture:
            print("[WARN] Gesture name cannot be empty!")
            continue

        gesture = gesture.replace(" ", "_")
        num_samples = input(f"How many samples for '{gesture}'? [30]: ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 30

        print(f"\n[INFO] Recording '{gesture}' — {num_samples} samples")
        print("[INFO] Press 's' to start/stop recording each sample")
        print("[INFO] Press 'q' to quit this gesture\n")

        if not hc.start():
            print("[ERROR] Failed to start webcam!")
            continue

        collected = 0
        buf = SequenceBuffer()
        recording = False

        try:
            while collected < num_samples:
                landmarks, annotated, num_hands = hc.capture_and_extract()
                if annotated is None:
                    continue

                status = "RECORDING" if recording else "READY"
                color = (0, 0, 255) if recording else (0, 200, 0)
                cv2.putText(annotated, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated, f"Status: {status}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(annotated, f"Samples: {collected}/{num_samples}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated, f"Hands: {num_hands}", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if recording:
                    progress = buf.current_length / config.SEQUENCE_LENGTH
                    bar_w = 300
                    cv2.rectangle(annotated, (10, 155), (10 + bar_w, 175), (50, 50, 50), -1)
                    cv2.rectangle(annotated, (10, 155), (10 + int(bar_w * progress), 175), color, -1)
                    cv2.putText(annotated, f"{buf.current_length}/{config.SEQUENCE_LENGTH}",
                                (320, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    normalized = normalize_landmarks(landmarks)
                    buf.push(normalized)

                    if buf.is_ready():
                        seq = buf.get_sequence()
                        path = dm.save_sequence(gesture, seq)
                        collected += 1
                        print(f"  ✓ Sample {collected}/{num_samples} saved → {os.path.basename(path)}")
                        buf.reset()
                        recording = False
                        time.sleep(0.3)

                cv2.imshow("SignSpeak — Data Collection", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    if not recording:
                        buf.reset()
                        recording = True
                        print(f"  ● Recording sample {collected + 1}...")
                    else:
                        recording = False
                        buf.reset()
                        print("  ○ Recording cancelled")
                elif key == ord("q"):
                    break

        finally:
            hc.stop()
            cv2.destroyAllWindows()

        print(f"\n[INFO] Collected {collected} samples for '{gesture}'")

    print("\n── Final Distribution ─────────────────────────")
    for label, count in dm.get_class_distribution().items():
        print(f"  {label}: {count} samples")
    print("\n[DONE] Data collection complete!")

if __name__ == "__main__":
    collect_data()
