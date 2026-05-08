"""
SignSpeak — Local Test Script
─────────────────────────────
Standalone OpenCV-based test for ASL or ISL models.
No Streamlit needed — runs with just OpenCV + TTS.

Usage:
  python test_local.py --mode asl
  python test_local.py --mode isl
  python test_local.py

Controls:
  a  = Add current prediction to sentence
  s  = Speak the full sentence
  c  = Clear sentence
  u  = Undo last word
  q  = Quit
"""

import os
import sys
import json
import argparse
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.capture_engine import HandCapture
from modules.normalization import normalize_landmarks, SequenceBuffer
from modules.predictor import ContinuousPredictor
from modules.tts_engine import TTSEngine
from modules.model import load_trained_model

def main():
    parser = argparse.ArgumentParser(description="SignSpeak Local Test")
    parser.add_argument("--mode", choices=["asl", "isl"], default="asl",
                        help="Mode: asl (A-Z letters) or isl (gestures)")
    args = parser.parse_args()

    mode = args.mode.upper()

    if mode == "ASL":
        model_path = config.ASL_MODEL_PATH
        label_path = config.ASL_LABEL_MAP_PATH
    else:
        model_path = config.ISL_MODEL_PATH
        label_path = config.ISL_LABEL_MAP_PATH

    print(f"\n{'=' * 50}")
    print(f"  SignSpeak Local Test — {mode} Mode")
    print(f"{'=' * 50}")

    model = load_trained_model(model_path)
    if model is None:
        print(f"[ERROR] Model not found: {model_path}")
        print(f"  Run: python scripts/train_split.py --{mode.lower()}")
        return

    with open(label_path, "r") as f:
        label_map = json.load(f)
    print(f"[INFO] Loaded {len(label_map)} classes: {', '.join(sorted(label_map.keys()))}")

    predictor = ContinuousPredictor(
        model=model,
        label_map=label_map,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )
    tts = TTSEngine(enabled=True)
    buf = SequenceBuffer()
    hc = HandCapture()

    sentence_words = []
    current_prediction = "—"
    current_confidence = 0.0

    if not hc.start():
        print("[ERROR] Failed to open webcam!")
        return

    print("\n[CONTROLS]")
    print("  a = Add to sentence  |  s = Speak sentence")
    print("  c = Clear sentence   |  u = Undo last word")
    print("  q = Quit")
    print()

    frame_count = 0

    try:
        while True:
            landmarks, annotated, num_hands = hc.capture_and_extract()
            if annotated is None:
                continue

            frame_count += 1

            normalized = normalize_landmarks(landmarks)
            buf.push(normalized)

            cv2.putText(annotated, f"Mode: {mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Buffer: {buf.current_length}/{config.SEQUENCE_LENGTH}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if buf.is_ready() and frame_count % 2 == 0:
                seq = buf.get_sequence()
                label, conf, raw_label = predictor.predict(seq)

                current_confidence = conf

                if label is not None:
                    current_prediction = label
                    display_name = label.replace("_", " ")
                    print(f"  >> {display_name.upper()} ({conf:.0%})")

                    tts.speak(display_name)

                    cv2.putText(annotated, f">> {display_name.upper()} ({conf:.0%})",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(annotated, f"({raw_label} {conf:.0%})",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

            sentence_str = " ".join(sentence_words) if sentence_words else "No sentence yet"
            cv2.putText(annotated, f"Sentence: {sentence_str}", (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

            cv2.putText(annotated, f"Prediction: {current_prediction}",
                        (10, annotated.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            cv2.imshow(f"SignSpeak — {mode} Mode", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if current_prediction != "—":
                    sentence_words.append(current_prediction)
                    display = current_prediction.replace("_", " ")
                    print(f"  [+] Added '{display}' → Sentence: {' '.join(sentence_words)}")
            elif key == ord('s'):
                if sentence_words:
                    full = " ".join(w.replace("_", " ") for w in sentence_words)
                    print(f"  [🔊] Speaking: {full}")
                    tts.speak(full)
            elif key == ord('c'):
                sentence_words.clear()
                current_prediction = "—"
                print("  [✓] Sentence cleared")
            elif key == ord('u'):
                if sentence_words:
                    removed = sentence_words.pop()
                    print(f"  [↩] Removed: {removed}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        hc.stop()
        cv2.destroyAllWindows()

    if sentence_words:
        final = " ".join(w.replace("_", " ") for w in sentence_words)
        print(f"\n[FINAL SENTENCE] {final}")

if __name__ == "__main__":
    main()
