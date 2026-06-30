"""
Load landmark CSV files and build temporal sequences.

Output:
  X_train, y_train  — CNN: (samples, 63)   | GRU: (samples, 30, 63)
  X_val,   y_val    — same shapes
  X_test,  y_test   — same shapes
"""

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    LANDMARKS_DIR, SEQ_LEN, SEQ_STRIDE, FEATURE_DIM,
    TRAIN_SPLIT, VAL_SPLIT, DATASET_DIR,
)


def load_landmark_csv(csv_path):
    """Load a single landmark CSV. Returns (frames, 63) array and gesture_id."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        gesture_id = None
        for row in reader:
            features = [float(row[f"{c}{i}"]) for i in range(21) for c in ["x", "y", "z"]]
            gid = int(row["gesture_id"])
            if gesture_id is None:
                gesture_id = gid
            rows.append(features)

    return np.array(rows, dtype=np.float32), gesture_id


def build_sequences_from_video(frames, gesture_id):
    """
    Build sliding window sequences from one video's landmark data.

    Returns:
      X: (num_sequences, SEQ_LEN, FEATURE_DIM) — for GRU
      y: (num_sequences,) — gesture_id for each sequence
      X_cnn: (num_sequences, FEATURE_DIM) — single frames for CNN (one per seq)
    """
    n = len(frames)
    if n < SEQ_LEN:
        # Pad by repeating first frame if too short
        pad_len = SEQ_LEN - n
        padding = np.repeat(frames[:1], pad_len, axis=0)
        frames = np.vstack([padding, frames])
        n = len(frames)

    sequences = []
    labels = []
    frames_cnn = []

    for start in range(0, n - SEQ_LEN + 1, SEQ_STRIDE):
        seq = frames[start:start + SEQ_LEN]
        sequences.append(seq)
        labels.append(gesture_id)
        frames_cnn.append(frames[start + SEQ_LEN - 1])  # last frame of window

    return np.array(sequences), np.array(labels, dtype=np.int32), np.array(frames_cnn)


def load_dataset():
    """
    Main loader. Returns:
      (X_cnn_train, y_train), (X_cnn_val, y_val), (X_cnn_test, y_test)
      (X_gru_train, y_train), (X_gru_val, y_val), (X_gru_test, y_test)
    """
    all_sequences = []
    all_frames_cnn = []
    all_labels = []

    gesture_dirs = sorted([
        d for d in os.listdir(LANDMARKS_DIR)
        if os.path.isdir(os.path.join(LANDMARKS_DIR, d))
    ])

    if not gesture_dirs:
        print(f"[WARN] No landmark data found in {LANDMARKS_DIR}")
        print("Run extract_landmarks.py first.")
        return None

    for gesture_dir in gesture_dirs:
        src_dir = os.path.join(LANDMARKS_DIR, gesture_dir)
        csv_files = sorted([
            f for f in os.listdir(src_dir)
            if f.endswith(".csv")
        ])

        for csv_file in csv_files:
            csv_path = os.path.join(src_dir, csv_file)
            frames, gesture_id = load_landmark_csv(csv_path)

            seqs, labels, cnn_frames = build_sequences_from_video(frames, gesture_id)
            if len(seqs) == 0:
                continue

            all_sequences.append(seqs)
            all_frames_cnn.append(cnn_frames)
            all_labels.append(labels)

    if not all_sequences:
        print("[ERROR] No sequences built. Check landmark data.")
        return None

    X_gru = np.vstack(all_sequences)       # (N, 30, 63)
    X_cnn = np.vstack(all_frames_cnn)      # (N, 63)
    y = np.concatenate(all_labels)          # (N,)

    # Stratified split
    tmp_split = TRAIN_SPLIT / (TRAIN_SPLIT + VAL_SPLIT) if (TRAIN_SPLIT + VAL_SPLIT) > 0 else 0.8

    X_gru_train, X_gru_tmp, X_cnn_train, X_cnn_tmp, y_train, y_tmp = train_test_split(
        X_gru, X_cnn, y, test_size=(1 - TRAIN_SPLIT), stratify=y, random_state=42
    )

    val_ratio = VAL_SPLIT / (1 - TRAIN_SPLIT) if (1 - TRAIN_SPLIT) > 0 else 0.5

    X_gru_val, X_gru_test, X_cnn_val, X_cnn_test, y_val, y_test = train_test_split(
        X_gru_tmp, X_cnn_tmp, y_tmp, test_size=(1 - val_ratio), stratify=y_tmp, random_state=42
    )

    print(f"Dataset loaded:")
    print(f"  CNN  — Train: {X_cnn_train.shape}, Val: {X_cnn_val.shape}, Test: {X_cnn_test.shape}")
    print(f"  GRU  — Train: {X_gru_train.shape}, Val: {X_gru_val.shape}, Test: {X_gru_test.shape}")

    return (X_cnn_train, y_train), (X_cnn_val, y_val), (X_cnn_test, y_test), \
           (X_gru_train, y_train), (X_gru_val, y_val), (X_gru_test, y_test)


def get_class_weights(y_train):
    """Compute balanced class weights for imbalanced datasets."""
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    weights = {c: n_samples / (n_classes * np.sum(y_train == c)) for c in classes}
    return weights


if __name__ == "__main__":
    data = load_dataset()
    if data is not None:
        (X_cnn_train, y_train), (X_cnn_val, y_val), (X_cnn_test, y_test), \
        (X_gru_train, _), (X_gru_val, _), (X_gru_test, _) = data

        print(f"\nClass distribution (train):")
        for cid in range(11):
            count = np.sum(y_train == cid)
            if count > 0:
                print(f"  {cid}: {count}")
