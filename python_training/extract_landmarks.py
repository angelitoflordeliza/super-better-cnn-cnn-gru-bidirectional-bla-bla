"""
Extract MediaPipe hand landmarks from extracted frames.

Uses mp.tasks.vision.HandLandmarker (MediaPipe v0.10+ API).

Input:  _frames/gesture_N_name/video_name/*.jpg
Output: _landmarks/gesture_N_name/video_name.csv

Each CSV row: frame_idx, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, gesture_id
"""

import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import FRAMES_DIR, LANDMARKS_DIR, GESTURE_LABELS


def extract_landmarks_from_frames(frames_dir, gesture_id):
    """Process frames using mp.tasks.vision.HandLandmarker."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    rows = []

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        # Download the standard hand landmarker model if not present
        print(f"  [INFO] Downloading hand_landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"  [INFO] Downloaded to {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            image = cv2.imread(frame_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            frame_idx = int(os.path.splitext(frame_file)[0].split("_")[1])

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand = result.hand_landmarks[0]
                row = [frame_idx]
                for lm in hand:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(gesture_id)
                rows.append(row)
            else:
                # No hand detected — record zeros + Unknown label
                row = [frame_idx] + [0.0] * 63 + [gesture_id]
                rows.append(row)

    return rows


def main():
    print("=" * 60)
    print("MUTED — Extract MediaPipe Hand Landmarks")
    print("=" * 60)

    total_rows = 0

    gesture_dirs = sorted([
        d for d in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, d))
    ])

    if not gesture_dirs:
        print(f"\nNo gesture directories in {FRAMES_DIR}")
        print("Run extract_frames.py first.\n")
        return

    for gesture_dir in gesture_dirs:
        gesture_id = None
        for gid, label in enumerate(GESTURE_LABELS):
            if label.lower().replace("_", "") in gesture_dir.lower().replace("_", ""):
                gesture_id = gid
                break

        if gesture_id is None:
            gesture_id = 10

        src_dir = os.path.join(FRAMES_DIR, gesture_dir)
        dst_dir = os.path.join(LANDMARKS_DIR, gesture_dir)
        os.makedirs(dst_dir, exist_ok=True)

        video_dirs = sorted([
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
        ])

        if not video_dirs:
            print(f"  [EMPTY] {gesture_dir}/ — no frame subdirectories")
            continue

        print(f"\n  Processing: {gesture_dir}/ (gesture_id={gesture_id}, "
              f"label={GESTURE_LABELS[gesture_id]}, {len(video_dirs)} videos)")

        for video_dir in tqdm(video_dirs, desc=f"  {gesture_dir}"):
            frames_path = os.path.join(src_dir, video_dir)
            csv_path = os.path.join(dst_dir, f"{video_dir}.csv")

            rows = extract_landmarks_from_frames(frames_path, gesture_id)
            if not rows:
                continue

            header = ["frame_idx"] + [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]] + ["gesture_id"]

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            total_rows += len(rows)

    print("\n" + "=" * 60)
    print(f"Done. Extracted {total_rows} landmark rows.")
    print(f"Output: {LANDMARKS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
