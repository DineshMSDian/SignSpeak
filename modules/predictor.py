"""
Module 5: Continuous Prediction Engine
──────────────────────────────────────
Sliding window prediction with confidence thresholding,
debounce logic, and sentence accumulation.
"""

import json
import time
import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ContinuousPredictor:
    """
    Real-time continuous gesture prediction engine.

    Applies the trained LSTM model to incoming sequences,
    filters by confidence, debounces repeated predictions,
    and accumulates words into sentences.
    """

    def __init__(
        self,
        model=None,
        label_map: dict = None,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        debounce_frames: int = config.DEBOUNCE_FRAMES,
        allowed_labels: set = None,
    ):
        """
        Args:
            model: Trained tf.keras.Model (can be set later).
            label_map: Dict mapping label names to integer indices.
            confidence_threshold: Minimum confidence for accepting a prediction.
            debounce_frames: Number of frames to wait before accepting the same prediction.
            allowed_labels: If set, only predictions in this set are accepted.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames
        self.allowed_labels = allowed_labels  # None = accept all

        # Build reverse label map: index → label name
        self.label_map = label_map or {}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Debounce state
        self._last_prediction = None
        self._frames_since_prediction = 0

        # Sentence builder
        self.sentence_builder = SentenceBuilder()

        # Performance tracking
        self._last_inference_time = 0.0

    def set_model(self, model):
        """Set or update the prediction model."""
        self.model = model

    def set_label_map(self, label_map: dict):
        """Set or update the label map."""
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}

    def set_allowed_labels(self, allowed_labels: set):
        """Set the allowed label filter. None = accept all."""
        self.allowed_labels = allowed_labels

    def load_label_map(self, path: str = config.LABEL_MAP_PATH):
        """Load label map from JSON file."""
        with open(path, "r") as f:
            self.label_map = json.load(f)
        self.reverse_label_map = {int(v): k for k, v in self.label_map.items()}

    def predict(self, sequence: np.ndarray):
        """
        Run prediction on a single sequence.

        Args:
            sequence: Shape (60, 126) — one gesture sequence.

        Returns:
            label (str | None): Predicted gesture label, or None if filtered.
            confidence (float): Prediction confidence score.
            raw_label (str): The raw prediction label regardless of filtering.
        """
        if self.model is None:
            return None, 0.0, "no_model"

        # Add batch dimension: (1, 60, 126)
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

        start = time.perf_counter()
        predictions = self.model.predict(input_data, verbose=0)
        self._last_inference_time = (time.perf_counter() - start) * 1000  # ms

        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        raw_label = self.reverse_label_map.get(class_idx, f"class_{class_idx}")

        # ── Confidence Filter ──────────────────────────────────
        if confidence < self.confidence_threshold:
            self._frames_since_prediction += 1
            return None, confidence, raw_label

        # ── Allowed Labels Filter ─────────────────────────────
        if self.allowed_labels is not None and raw_label not in self.allowed_labels:
            self._frames_since_prediction += 1
            return None, confidence, raw_label

        # ── Debounce Logic ─────────────────────────────────────
        if raw_label == self._last_prediction:
            self._frames_since_prediction += 1
            if self._frames_since_prediction < self.debounce_frames:
                return None, confidence, raw_label

        # ── Accept Prediction ──────────────────────────────────
        self._last_prediction = raw_label
        self._frames_since_prediction = 0

        return raw_label, confidence, raw_label

    def reset(self):
        """Reset predictor state and sentence."""
        self._last_prediction = None
        self._frames_since_prediction = 0
        self.sentence_builder.reset()

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._last_inference_time

    @property
    def current_sentence(self) -> str:
        """Current accumulated sentence."""
        return self.sentence_builder.get_sentence()


class SentenceBuilder:
    """
    Accumulates predicted gesture labels into a readable sentence.
    """

    def __init__(self):
        self._words = []

    def add_word(self, word: str):
        """Append a word to the sentence."""
        if word and isinstance(word, str):
            self._words.append(word)

    def get_sentence(self) -> str:
        """Return the accumulated sentence as a single string."""
        return " ".join(self._words)

    def get_words(self) -> list:
        """Return the list of accumulated words."""
        return list(self._words)

    def undo(self) -> str:
        """Remove and return the last word, or empty string if empty."""
        return self._words.pop() if self._words else ""

    def reset(self):
        """Clear all accumulated words."""
        self._words.clear()

    def __len__(self):
        return len(self._words)

    def __str__(self):
        return self.get_sentence()

from collections import Counter

class SmoothedPredictor:
    """Majority-vote wrapper around ContinuousPredictor.
    Only emits a prediction when the same label wins 75% of
    the last N frames. Eliminates flicker for sentence building.
    """

    def __init__(self, predictor: ContinuousPredictor, vote_window: int = 8):
        self.predictor = predictor
        self.vote_window = vote_window
        self._recent = []
        self._last_emitted = None
        self._cooldown = 0
        self.COOLDOWN_FRAMES = 20

    def update(self, sequence: np.ndarray):
        """Returns (label, confidence) or (None, 0.0)."""
        label, conf, raw = self.predictor.predict(sequence)

        if self._cooldown > 0:
            self._cooldown -= 1
            return None, 0.0

        self._recent.append(raw)
        if len(self._recent) > self.vote_window:
            self._recent.pop(0)

        if len(self._recent) < self.vote_window:
            return None, 0.0

        most_common, count = Counter(self._recent).most_common(1)[0]

        if (count >= int(self.vote_window * 0.75)
                and most_common != self._last_emitted
                and conf >= 0.55):
            self._last_emitted = most_common
            self._cooldown = self.COOLDOWN_FRAMES
            self._recent.clear()
            return most_common, conf

        return None, 0.0

    def reset(self):
        self._recent.clear()
        self._last_emitted = None
        self._cooldown = 0
        
# ── Standalone Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Continuous Prediction Engine")

    # Test SentenceBuilder
    sb = SentenceBuilder()
    sb.add_word("hello")
    sb.add_word("world")
    assert sb.get_sentence() == "hello world"
    sb.undo()
    assert sb.get_sentence() == "hello"
    sb.reset()
    assert len(sb) == 0
    print("  ✓ SentenceBuilder works correctly")

    # Test predictor without model (should return None)
    pred = ContinuousPredictor()
    seq = np.random.rand(config.SEQUENCE_LENGTH, config.FEATURE_DIM).astype(np.float32)
    label, conf, raw = pred.predict(seq)
    assert label is None
    assert raw == "no_model"
    print("  ✓ Predictor handles missing model gracefully")

    print("[PASS] All prediction engine tests passed!")
