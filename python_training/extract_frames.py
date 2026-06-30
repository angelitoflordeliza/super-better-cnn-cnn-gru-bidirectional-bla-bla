"""
Extract frames from MP4 gesture videos at 30 FPS.

Reads from:   datasets/raw/sessionNNN/GestureName/*.mp4
Outputs to:   _frames/gesture_N_name/video_name/*.jpg

Auto-flattens all sessions into gesture-labeled frame directories.
"""

import cv2
import os
import re
from tqdm import tqdm

from config import RAW_DIR, FRAMES_DIR, FPS, GESTURE_FOLDER_MAP


def normalize_gesture_folder_name(name):
    """Convert folder name like 'Two_Finger_V' → 'two_finger_v' for lookup."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def extract_frames_from_video(video_path, output_dir, target_fps=FPS):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = target_fps

    frame_interval = max(1, int(round(original_fps / target_fps)))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def discover_raw_videos(raw_dir):
    """
    Walk raw_dir looking for sessionNNN/GestureName/*.mp4.
    Returns list of (video_path, gesture_id, gesture_name).
    """
    found = []
    if not os.path.isdir(raw_dir):
        print(f"  [ERROR] Raw directory not found: {raw_dir}")
        return found

    session_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    for session_dir in session_dirs:
        session_path = os.path.join(raw_dir, session_dir)
        gesture_dirs = [
            d for d in os.listdir(session_path)
            if os.path.isdir(os.path.join(session_path, d))
        ]

        for gdir in gesture_dirs:
            # Skip log files / non-gesture folders
            gdir_lower = normalize_gesture_folder_name(gdir)
            if gdir_lower.startswith("session_log") or gdir_lower.endswith(".txt"):
                continue

            gesture_id = GESTURE_FOLDER_MAP.get(gdir_lower, None)
            if gesture_id is None:
                print(f"  [SKIP] Unknown gesture folder: {session_dir}/{gdir}")
                continue

            gesture_path = os.path.join(session_path, gdir)
            video_files = sorted([
                f for f in os.listdir(gesture_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ])

            for vf in video_files:
                video_path = os.path.join(gesture_path, vf)
                found.append((video_path, gesture_id, gdir))

    return found


def main():
    print("=" * 60)
    print("MUTED — Extract Frames from Raw Session Videos")
    print("=" * 60)

    videos = discover_raw_videos(RAW_DIR)
    if not videos:
        print(f"\nNo MP4 videos found in {RAW_DIR}")
        print("Expected: datasets/raw/sessionNNN/GestureName/*.mp4")
        return

    print(f"\nFound {len(videos)} videos across {RAW_DIR}")
    print(f"Output: {FRAMES_DIR}/\n")

    total_frames = 0

    for video_path, gesture_id, gesture_name in tqdm(videos, desc="Extracting frames"):
        # Output: _frames/gesture_{id}_{name}/{video_name}/
        gesture_dir_name = f"gesture_{gesture_id}_{gesture_name}"
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Sanitize video name
        video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)

        frame_out_dir = os.path.join(FRAMES_DIR, gesture_dir_name, video_name)
        saved = extract_frames_from_video(video_path, frame_out_dir)
        total_frames += saved

    print("\n" + "=" * 60)
    print(f"Done. Extracted {total_frames} frames from {len(videos)} videos.")
    print("=" * 60)


if __name__ == "__main__":
    main()
