"""
Kaggle Video Dataset Processor
────────────────────────────────
Processes video-based gesture datasets (e.g., WLASL, INCLUDE, ISL)
by extracting MediaPipe landmarks, normalizing, and saving as
60-frame sequences ready for LSTM training.

Expected input folder structure:
    dataset_folder/
    ├── hello/
    │   ├── video1.mp4
    │   ├── video2.avi
    │   └── ...
    ├── thank_you/
    │   ├── video1.mp4
    │   └── ...
    └── goodbye/
        └── ...

Usage:
    python scripts/process_kaggle.py --input path/to/dataset --output data/raw
    python scripts/process_kaggle.py --input path/to/dataset
"""

import os
import sys
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from modules.capture_engine import HandCapture
from modules.normalization import normalize_landmarks
from modules.dataset_manager import DatasetManager

def extract_sequences_from_video(
    video_path: str,
    hands_detector,
    sequence_length: int = config.SEQUENCE_LENGTH,
    stride: int = None,
) -> list:
    """
    Extract landmark sequences from a single video file.

    Opens the video, runs MediaPipe on each frame, normalizes landmarks,
    and slices the result into fixed-length sequences.

    Args:
        video_path: Path to the video file (.mp4, .avi, .mov, .mkv).
        hands_detector: MediaPipe Hands instance (from HandCapture).
        sequence_length: Number of frames per sequence (default 60).
        stride: Step size for sliding window. Defaults to sequence_length
                (non-overlapping). Use smaller values for more samples.

    Returns:
        List of np.ndarray, each of shape (sequence_length, 126).
    """
    if stride is None:
        stride = sequence_length

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    all_frame_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_detector.process(rgb)

        landmarks = np.zeros(
            (config.NUM_HANDS * config.NUM_LANDMARKS, config.NUM_COORDS),
            dtype=np.float32,
        )

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= config.NUM_HANDS:
                    break
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    offset = hand_idx * config.NUM_LANDMARKS
                    landmarks[offset + lm_idx] = [lm.x, lm.y, lm.z]

        normalized = normalize_landmarks(landmarks)
        all_frame_vectors.append(normalized)

    cap.release()

    total_frames = len(all_frame_vectors)
    if total_frames < sequence_length:
        padding = [np.zeros(config.FEATURE_DIM, dtype=np.float32)] * (sequence_length - total_frames)
        all_frame_vectors = padding + all_frame_vectors
        return [np.array(all_frame_vectors, dtype=np.float32)]

    sequences = []
    for start in range(0, total_frames - sequence_length + 1, stride):
        seq = np.array(
            all_frame_vectors[start : start + sequence_length],
            dtype=np.float32,
        )
        sequences.append(seq)

    return sequences

def process_dataset(
    input_dir: str,
    output_dir: str = None,
    sequence_length: int = config.SEQUENCE_LENGTH,
    stride: int = None,
    overlap: float = None,
    video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
):
    """
    Process an entire video gesture dataset.

    Args:
        input_dir: Root folder containing subfolders per gesture class.
        output_dir: Where to save sequences (default: data/raw).
        sequence_length: Frames per sequence (default 60).
        stride: Sliding window stride. Overridden by overlap if set.
        overlap: Fraction of overlap between sequences (0.0–0.9).
                 E.g., 0.5 = 50% overlap = stride of 30 frames.
        video_extensions: Supported video file extensions.
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR

    if overlap is not None:
        stride = max(1, int(sequence_length * (1 - overlap)))
    elif stride is None:
        stride = sequence_length

    dm = DatasetManager(raw_dir=output_dir)

    gesture_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gesture_dirs:
        print(f"[ERROR] No subfolders found in {input_dir}")
        print("Expected structure: input_dir/{gesture_name}/videos...")
        return

    print("=" * 60)
    print("  SignSpeak — Kaggle Video Dataset Processor")
    print("=" * 60)
    print(f"  Input:           {input_dir}")
    print(f"  Output:          {output_dir}")
    print(f"  Gestures found:  {len(gesture_dirs)}")
    print(f"  Sequence length: {sequence_length} frames")
    print(f"  Stride:          {stride} frames")
    print(f"  Overlap:         {((sequence_length - stride) / sequence_length) * 100:.0f}%")
    print("=" * 60)

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    total_sequences = 0
    stats = {}

    for gesture in gesture_dirs:
        gesture_path = os.path.join(input_dir, gesture)
        gesture_label = gesture.lower().replace(" ", "_").replace("-", "_")

        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(gesture_path, f"*{ext}")))
            videos.extend(glob.glob(os.path.join(gesture_path, f"*{ext.upper()}")))
        videos = sorted(set(videos))

        if not videos:
            print(f"\n[WARN] No videos found for '{gesture}' — skipping")
            continue

        print(f"\n─── Processing: {gesture_label} ({len(videos)} videos) ───")

        gesture_seq_count = 0

        for video_path in tqdm(videos, desc=f"  {gesture_label}", unit="video"):
            sequences = extract_sequences_from_video(
                video_path, hands, sequence_length, stride
            )

            for seq in sequences:
                if seq.shape == (sequence_length, config.FEATURE_DIM):
                    dm.save_sequence(gesture_label, seq)
                    gesture_seq_count += 1

        stats[gesture_label] = gesture_seq_count
        total_sequences += gesture_seq_count
        print(f"  → {gesture_seq_count} sequences saved")

    hands.close()

    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total sequences: {total_sequences}")
    print(f"  Gestures:        {len(stats)}")
    print()
    for label, count in sorted(stats.items()):
        print(f"    {label:20s} → {count:5d} sequences")
    print()
    print(f"  Output saved to: {output_dir}")
    print()
    print("  Next steps:")
    print("    1. python scripts/train.py      # Train the LSTM model")
    print("    2. python scripts/evaluate.py    # Evaluate performance")
    print("    3. streamlit run app.py          # Launch the translator")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Process a Kaggle video gesture dataset for SignSpeak LSTM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_kaggle.py --input ./kaggle_dataset
  python scripts/process_kaggle.py --input ./WLASL/videos --overlap 0.5
  python scripts/process_kaggle.py --input ./ISL_dataset --stride 30
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to dataset root (subfolders = gesture classes, containing videos)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=f"Output directory for sequences (default: {config.RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--sequence-length", "-l", type=int, default=config.SEQUENCE_LENGTH,
        help=f"Frames per sequence (default: {config.SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--stride", "-s", type=int, default=None,
        help="Sliding window stride in frames (default: sequence_length = no overlap)",
    )
    parser.add_argument(
        "--overlap", type=float, default=None,
        help="Overlap fraction between sequences, 0.0–0.9 (overrides --stride)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        sys.exit(1)

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        sequence_length=args.sequence_length,
        stride=args.stride,
        overlap=args.overlap,
    )

if __name__ == "__main__":
    main()
