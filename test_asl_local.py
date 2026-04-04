"""
SignSpeak — ASL Alphabet Local Tester
─────────────────────────────────────
Standalone real-time ASL alphabet recognition using webcam.
No Streamlit required. Just run: python test_asl_local.py

Requirements:
    pip install opencv-python mediapipe tensorflow numpy

Controls:
    Q / ESC  — Quit
    R        — Reset prediction buffer
    SPACE    — Toggle pause/resume
"""

import os
import sys
import cv2
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import mediapipe as mp

# ─── Configuration ──────────────────────────────────────────────
NUM_LANDMARKS = 21
NUM_HANDS = 2
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_HANDS * NUM_COORDS  # 126
SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.7

# Paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sign_lstm_best.keras")
LABEL_MAP_PATH = os.path.join(SCRIPT_DIR, "models", "label_map.json")


# ─── Normalization (self-contained) ────────────────────────────
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Wrist-relative normalization: subtract wrist from all landmarks per hand."""
    normalized = np.zeros_like(landmarks, dtype=np.float32)
    for hand_idx in range(NUM_HANDS):
        start = hand_idx * NUM_LANDMARKS
        end = start + NUM_LANDMARKS
        hand = landmarks[start:end]
        wrist = hand[0].copy()
        normalized[start:end] = hand - wrist
    return normalized.flatten()


class SequenceBuffer:
    """Rolling buffer for 60-frame sequences."""

    def __init__(self):
        self._buffer = []

    def push(self, frame_vector: np.ndarray):
        self._buffer.append(frame_vector.copy())
        if len(self._buffer) > SEQUENCE_LENGTH:
            self._buffer.pop(0)

    def is_ready(self) -> bool:
        return len(self._buffer) >= SEQUENCE_LENGTH

    def get_sequence(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        arr = np.array(self._buffer, dtype=np.float32)
        if len(arr) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(arr), FEATURE_DIM), dtype=np.float32)
            arr = np.vstack([padding, arr])
        return arr

    def reset(self):
        self._buffer.clear()

    @property
    def length(self):
        return len(self._buffer)


# ─── Main ───────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SignSpeak — ASL Alphabet Local Tester")
    print("=" * 55)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load label map
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[ERROR] Label map not found: {LABEL_MAP_PATH}")
        sys.exit(1)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    reverse_map = {v: k for k, v in label_map.items()}
    print(f"[INFO] Loaded {len(label_map)} classes: {list(label_map.keys())}")

    # Init MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=NUM_HANDS,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Init webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buf = SequenceBuffer()
    current_prediction = ""
    current_confidence = 0.0
    paused = False
    frame_count = 0
    predict_every = 3  # predict every N frames for speed

    print("\n[INFO] Webcam started!")
    print("[INFO] Controls: Q/ESC=Quit | R=Reset | SPACE=Pause")
    print("-" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        display = frame.copy()

        if not paused:
            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            # Build landmarks array (42, 3)
            landmarks = np.zeros((NUM_HANDS * NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= NUM_HANDS:
                        break
                    # Draw hand landmarks on display
                    mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        offset = hand_idx * NUM_LANDMARKS
                        landmarks[offset + lm_idx] = [lm.x, lm.y, lm.z]

            # Normalize and buffer
            normalized = normalize_landmarks(landmarks)
            buf.push(normalized)

            # Predict
            frame_count += 1
            if buf.is_ready() and frame_count % predict_every == 0:
                seq = buf.get_sequence()
                seq_input = np.expand_dims(seq, axis=0)
                pred = model.predict(seq_input, verbose=0)[0]
                class_idx = np.argmax(pred)
                confidence = pred[class_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_prediction = reverse_map.get(class_idx, "?").upper()
                    current_confidence = confidence

        # ── Draw UI overlay ─────────────────────────────────────
        h, w = display.shape[:2]

        # Dark header bar
        cv2.rectangle(display, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(display, "SignSpeak | ASL Tester", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Buffer progress bar
        progress = buf.length / SEQUENCE_LENGTH
        bar_x, bar_y, bar_w, bar_h = 10, 42, 200, 18
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (0, 200, 100), -1)
        cv2.putText(display, f"Buffer: {buf.length}/{SEQUENCE_LENGTH}",
                    (bar_x + bar_w + 10, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Status
        if paused:
            cv2.putText(display, "PAUSED", (w - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Prediction box (bottom)
        if current_prediction:
            # Background box
            box_h = 80
            cv2.rectangle(display, (0, h - box_h), (w, h), (30, 30, 30), -1)

            # Big letter
            color = (0, 255, 100) if current_confidence > 0.9 else (0, 200, 255)
            cv2.putText(display, current_prediction, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

            # Confidence bar
            conf_bar_x = 100
            conf_bar_w = 200
            cv2.rectangle(display, (conf_bar_x, h - 40), (conf_bar_x + conf_bar_w, h - 22),
                          (60, 60, 60), -1)
            cv2.rectangle(display, (conf_bar_x, h - 40),
                          (conf_bar_x + int(conf_bar_w * current_confidence), h - 22),
                          color, -1)
            cv2.putText(display, f"{current_confidence:.0%}",
                        (conf_bar_x + conf_bar_w + 10, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("SignSpeak - ASL Tester", display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key == ord("r"):
            buf.reset()
            current_prediction = ""
            current_confidence = 0.0
            print("[INFO] Buffer reset")
        elif key == ord(" "):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\n[DONE] Session ended.")


if __name__ == "__main__":
    main()
