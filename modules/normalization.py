"""
Module 2: Skeletal Normalization Engine
───────────────────────────────────────
Wrist-relative landmark normalization and rolling sequence buffer.
Converts raw (42, 3) landmark arrays into position-invariant
126-dimensional vectors and manages 60-frame sliding windows.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply wrist-relative normalization to raw landmarks.

    For each hand (21 landmarks), subtract the wrist position (landmark 0)
    from all landmarks, making the representation position-invariant.

    Args:
        landmarks: Shape (42, 3) — 21 landmarks × 2 hands.

    Returns:
        Flattened normalized vector of shape (126,).
    """
    normalized = np.zeros_like(landmarks, dtype=np.float32)

    for hand_idx in range(config.NUM_HANDS):
        start = hand_idx * config.NUM_LANDMARKS
        end = start + config.NUM_LANDMARKS
        hand = landmarks[start:end]

        wrist = hand[0].copy()
        normalized[start:end] = hand - wrist

    return normalized.flatten()

class SequenceBuffer:
    """
    Rolling buffer that collects normalized frame vectors
    into fixed-length sequences for LSTM input.

    Maintains a buffer of shape (SEQUENCE_LENGTH, FEATURE_DIM).
    """

    def __init__(
        self,
        sequence_length: int = config.SEQUENCE_LENGTH,
        feature_dim: int = config.FEATURE_DIM,
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._buffer = []

    def push(self, frame_vector: np.ndarray):
        """
        Append a single frame vector to the buffer.

        Args:
            frame_vector: Normalized vector of shape (feature_dim,).
        """
        if frame_vector.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected shape ({self.feature_dim},), got {frame_vector.shape}"
            )

        self._buffer.append(frame_vector.copy())

        if len(self._buffer) > self.sequence_length:
            self._buffer.pop(0)

    def is_ready(self) -> bool:
        """True when the buffer has exactly `sequence_length` frames."""
        return len(self._buffer) >= self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """
        Return the current sequence as a NumPy array.

        Returns:
            np.ndarray: Shape (sequence_length, feature_dim).
                        Zero-padded if not yet full.
        """
        if not self._buffer:
            return np.zeros(
                (self.sequence_length, self.feature_dim), dtype=np.float32
            )

        arr = np.array(self._buffer, dtype=np.float32)

        if len(arr) < self.sequence_length:
            padding = np.zeros(
                (self.sequence_length - len(arr), self.feature_dim),
                dtype=np.float32,
            )
            arr = np.vstack([padding, arr])

        return arr

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def current_length(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    def __len__(self):
        return len(self._buffer)

if __name__ == "__main__":
    print("[TEST] Skeletal Normalization Engine")

    test_landmarks = np.random.rand(42, 3).astype(np.float32)
    normalized = normalize_landmarks(test_landmarks)
    print(f"  Input shape:  {test_landmarks.shape}")
    print(f"  Output shape: {normalized.shape}")
    assert normalized.shape == (config.FEATURE_DIM,), "Shape mismatch!"

    reshaped = normalized.reshape(config.NUM_HANDS, config.NUM_LANDMARKS, config.NUM_COORDS)
    for h in range(config.NUM_HANDS):
        wrist = reshaped[h, 0, :]
        assert np.allclose(wrist, 0.0), f"Hand {h} wrist not zeroed: {wrist}"
    print("  ✓ Wrist-relative normalization correct")

    buf = SequenceBuffer()
    for i in range(config.SEQUENCE_LENGTH):
        vec = np.random.rand(config.FEATURE_DIM).astype(np.float32)
        buf.push(vec)

    assert buf.is_ready(), "Buffer should be ready"
    seq = buf.get_sequence()
    assert seq.shape == (config.SEQUENCE_LENGTH, config.FEATURE_DIM)
    print(f"  ✓ Sequence buffer shape: {seq.shape}")

    buf.reset()
    assert buf.current_length == 0, "Buffer should be empty after reset"
    print("  ✓ Buffer reset works")

    print("[PASS] All normalization tests passed!")
