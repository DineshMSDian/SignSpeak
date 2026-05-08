"""
Module 3: Dataset Creation & Partitioning
─────────────────────────────────────────
Manages structured dataset storage, loading, stratified splitting,
and label mapping for gesture sequences.
"""

import os
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

class DatasetManager:
    """
    Handles creation, storage, and partitioning of gesture sequence datasets.

    Directory structure:
        data/raw/{gesture_label}/sequence_0001.npy
        data/raw/{gesture_label}/sequence_0002.npy
        ...
    """

    def __init__(
        self,
        raw_dir: str = config.RAW_DATA_DIR,
        processed_dir: str = config.PROCESSED_DATA_DIR,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def save_sequence(self, gesture_label: str, sequence: np.ndarray) -> str:
        """
        Save a single gesture sequence to the raw data directory.

        Args:
            gesture_label: Name of the gesture (e.g., "hello", "thank_you").
            sequence: NumPy array of shape (SEQUENCE_LENGTH, FEATURE_DIM).

        Returns:
            Path to the saved .npy file.
        """
        expected = (config.SEQUENCE_LENGTH, config.FEATURE_DIM)
        if sequence.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {sequence.shape}")

        label_dir = os.path.join(self.raw_dir, gesture_label)
        os.makedirs(label_dir, exist_ok=True)

        existing = glob.glob(os.path.join(label_dir, "sequence_*.npy"))
        next_idx = len(existing) + 1
        filename = f"sequence_{next_idx:04d}.npy"
        filepath = os.path.join(label_dir, filename)

        np.save(filepath, sequence)
        return filepath

    def get_gesture_labels(self) -> list:
        """Return sorted list of gesture labels from raw data directory."""
        if not os.path.exists(self.raw_dir):
            return []
        labels = [
            d for d in sorted(os.listdir(self.raw_dir))
            if os.path.isdir(os.path.join(self.raw_dir, d))
        ]
        return labels

    def get_class_distribution(self) -> dict:
        """Return dictionary of {label: sample_count} for each gesture."""
        distribution = {}
        for label in self.get_gesture_labels():
            label_dir = os.path.join(self.raw_dir, label)
            count = len(glob.glob(os.path.join(label_dir, "sequence_*.npy")))
            distribution[label] = count
        return distribution

    def build_dataset(self, test_size: float = 0.2, random_state: int = 42):
        """
        Load all raw sequences, create train/test split, and save as NumPy arrays.

        Creates:
            - X_train.npy, X_test.npy: Feature arrays
            - y_train.npy, y_test.npy: Label arrays (integer-encoded)
            - label_map.json: {label_name: integer_index}

        Args:
            test_size: Fraction of data for testing (default 0.2).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, label_map).
        """
        labels = self.get_gesture_labels()
        if not labels:
            raise ValueError("No gesture data found in raw directory!")

        label_map = {label: idx for idx, label in enumerate(labels)}

        X_all = []
        y_all = []

        for label in labels:
            label_dir = os.path.join(self.raw_dir, label)
            files = sorted(glob.glob(os.path.join(label_dir, "sequence_*.npy")))

            for filepath in files:
                seq = np.load(filepath)
                if seq.shape == (config.SEQUENCE_LENGTH, config.FEATURE_DIM):
                    X_all.append(seq)
                    y_all.append(label_map[label])
                else:
                    print(f"[WARN] Skipping {filepath}: shape {seq.shape}")

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)

        print(f"[INFO] Total samples: {len(X_all)}")
        print(f"[INFO] Classes: {len(labels)}")
        print(f"[INFO] Distribution: {self.get_class_distribution()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y_all,
        )

        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

        with open(config.LABEL_MAP_PATH, "w") as f:
            json.dump(label_map, f, indent=2)

        print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"[INFO] Saved to {self.processed_dir}")

        return X_train, X_test, y_train, y_test, label_map

    def load_processed(self):
        """
        Load previously processed dataset.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, label_map).
        """
        X_train = np.load(os.path.join(self.processed_dir, "X_train.npy"))
        X_test = np.load(os.path.join(self.processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(self.processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(self.processed_dir, "y_test.npy"))

        with open(config.LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)

        return X_train, X_test, y_train, y_test, label_map

if __name__ == "__main__":
    dm = DatasetManager()
    print("[INFO] Gesture labels:", dm.get_gesture_labels())
    print("[INFO] Class distribution:", dm.get_class_distribution())
