import os
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "datasets", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "datasets", "processed")

# Config
TEST_SPLIT   = 0.2      # 80/20 split
RANDOM_SEED  = 42
SEQ_LENGTH   = 60       # expected frames per sample
FEATURES     = 126      # 2 hands × 21 landmarks × 3 coords


def normalise_sample(sample: np.ndarray) -> np.ndarray:
    """
    Wrist-relative normalisation.
    For each frame, subtract the wrist (landmark 0) position from all
    21 landmarks — separately for each hand.  This makes the gesture
    position-invariant (works regardless of where the hand appears on
    screen).

    Input shape:  (SEQ_LENGTH, 126)
    Output shape: (SEQ_LENGTH, 126)
    """
    normalised = sample.copy()

    for hand_offset in [0, 63]:                     # left hand, right hand
        for frame_idx in range(normalised.shape[0]):
            wrist_x = normalised[frame_idx, hand_offset + 0]
            wrist_y = normalised[frame_idx, hand_offset + 1]
            wrist_z = normalised[frame_idx, hand_offset + 2]

            # If the hand was absent (all zeros) skip normalisation
            hand_slice = normalised[frame_idx, hand_offset : hand_offset + 63]
            if np.all(hand_slice == 0):
                continue

            # Subtract wrist position from every landmark
            for lm in range(21):
                base = hand_offset + lm * 3
                normalised[frame_idx, base + 0] -= wrist_x
                normalised[frame_idx, base + 1] -= wrist_y
                normalised[frame_idx, base + 2] -= wrist_z

    return normalised


def pad_or_truncate(sample: np.ndarray, target_len: int = SEQ_LENGTH) -> np.ndarray:
    """Ensure every sample is exactly target_len frames."""
    if sample.shape[0] > target_len:
        return sample[:target_len]
    elif sample.shape[0] < target_len:
        pad_width = target_len - sample.shape[0]
        return np.pad(sample, ((0, pad_width), (0, 0)), mode="constant")
    return sample


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover gesture classes
    classes = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    if not classes:
        print("No gesture folders found in", RAW_DIR)
        return

    print(f"Found {len(classes)} gesture classes: {', '.join(classes)}\n")

    # Load & preprocess every sample
    all_samples = []
    all_labels  = []
    class_counts = {}

    for label_idx, gesture in enumerate(classes):
        gesture_dir = os.path.join(RAW_DIR, gesture)
        files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
        class_counts[gesture] = len(files)

        for fname in files:
            sample = np.load(os.path.join(gesture_dir, fname))
            sample = pad_or_truncate(sample)
            sample = normalise_sample(sample)
            all_samples.append(sample)
            all_labels.append(label_idx)

    X = np.array(all_samples, dtype=np.float32)      # (N, 60, 126)
    y = np.array(all_labels,  dtype=np.int32)         # (N,)

    print("── Dataset Summary ──")
    for gesture, count in class_counts.items():
        print(f"  {gesture:15s}  {count:3d} samples")
    print(f"\n  Total: {len(X)} samples")
    print(f"  Shape: X={X.shape}  y={y.shape}\n")

    # ── Warn about tiny classes ────────────────────
    min_needed = max(2, int(1 / TEST_SPLIT))
    tiny = [g for g, c in class_counts.items() if c < min_needed]
    if tiny:
        print(f"  ⚠ Classes with < {min_needed} samples (can't stratify): {', '.join(tiny)}")

    # ── Train / Test split ───────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SPLIT,
            random_state = RANDOM_SEED,
            stratify     = y
        )
    except ValueError:
        print("  ⚠ Falling back to non-stratified split (some classes too small)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SPLIT,
            random_state = RANDOM_SEED,
        )

    print(f"\n  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples\n")

    # ── Save ─────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

    # Save class names for later label decoding
    np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(classes))

    print(f"✓ Saved processed data to {OUT_DIR}")
    print(f"  Files: X_train.npy, X_test.npy, y_train.npy, y_test.npy, classes.npy")


if __name__ == "__main__":
    main()
