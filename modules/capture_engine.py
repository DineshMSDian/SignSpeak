"""
Module 1: Real-Time Capture Engine
───────────────────────────────────
OpenCV webcam streaming + MediaPipe hand landmark extraction.
Handles 0, 1, or 2 hands gracefully with zero-fill for missing data.
"""

import cv2
import numpy as np
import mediapipe as mp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class HandCapture:
    """
    Real-time hand landmark capture using OpenCV and MediaPipe.

    Provides lifecycle management for webcam streaming, hand detection,
    and 21-landmark extraction for up to 2 hands.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        self.camera_index = camera_index
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.cap = None
        self.hands = None
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        """Initialize webcam and MediaPipe Hands. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera at index {self.camera_index}")
            return False

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.NUM_HANDS,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._running = True
        return True

    def stop(self):
        """Release webcam and MediaPipe resources."""
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.cap = None
        self.hands = None

    @property
    def is_running(self) -> bool:
        return self._running and self.cap is not None and self.cap.isOpened()

    # ── Frame Capture ──────────────────────────────────────────

    def get_frame(self):
        """
        Read a single frame from the webcam.

        Returns:
            frame (np.ndarray | None): BGR frame, or None if read failed.
        """
        if not self.is_running:
            return None
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror horizontally
        return frame if ret else None

    # ── Landmark Extraction ────────────────────────────────────

    def extract_landmarks(self, frame: np.ndarray):
        """
        Detect hands and extract landmarks from a BGR frame.

        Returns:
            landmarks (np.ndarray): Shape (42, 3) — 21 landmarks × 2 hands × 3 coords.
                                    Missing hands are zero-filled.
            annotated_frame (np.ndarray): Frame with landmark overlay drawn.
            num_hands (int): Number of hands actually detected (0, 1, or 2).
        """
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        # Initialize zero-filled landmarks for both hands
        all_landmarks = np.zeros(
            (config.NUM_HANDS * config.NUM_LANDMARKS, config.NUM_COORDS),
            dtype=np.float32,
        )

        num_hands = 0

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= config.NUM_HANDS:
                    break

                # Draw landmarks on the annotated frame
                self.mp_draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

                # Extract (x, y, z) for each of the 21 landmarks
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    offset = hand_idx * config.NUM_LANDMARKS
                    all_landmarks[offset + lm_idx] = [lm.x, lm.y, lm.z]

        return all_landmarks, annotated, num_hands

    # ── Convenience ────────────────────────────────────────────

    def capture_and_extract(self):
        """
        Combined: read frame + extract landmarks in one call.

        Returns:
            landmarks (np.ndarray | None): (42, 3) array or None if no frame.
            annotated_frame (np.ndarray | None): Annotated frame or None.
            num_hands (int): Detected hand count.
        """
        frame = self.get_frame()
        if frame is None:
            return None, None, 0
        landmarks, annotated, num_hands = self.extract_landmarks(frame)
        return landmarks, annotated, num_hands

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Standalone Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Hand Capture Engine — press 'q' to quit")
    with HandCapture() as hc:
        if not hc.is_running:
            print("[ERROR] Failed to start capture engine")
            exit(1)

        while True:
            landmarks, annotated, num_hands = hc.capture_and_extract()
            if annotated is None:
                continue

            cv2.putText(
                annotated,
                f"Hands: {num_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SignSpeak — Capture Engine", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("[INFO] Capture engine stopped")
