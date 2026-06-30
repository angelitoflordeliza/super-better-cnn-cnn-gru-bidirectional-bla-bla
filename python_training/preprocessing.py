"""
Preprocessing and augmentation for hand landmark data.

Landmarks from MediaPipe are already normalized [0, 1] range for x/y.
Z is relative depth (can be negative).

Augmentation techniques applied to 63-dim vectors:
  1. Gaussian noise (simulates tracking jitter)
  2. Random 2D rotation (simulates hand rotation in camera plane)
  3. Random scaling (simulates distance changes)
  4. Random translation (simulates positional jitter)
"""

import numpy as np

from config import (
    FEATURE_DIM, AUGMENT, AUG_NOISE_STD,
    AUG_ROTATION_DEG, AUG_SCALE_RANGE,
)


def rotation_matrix_2d(angle_deg):
    """2D rotation matrix in the XY plane."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def augment_landmarks(landmarks_63, noise_std=AUG_NOISE_STD,
                      rot_deg=AUG_ROTATION_DEG, scale_range=AUG_SCALE_RANGE):
    """
    Augment a single 63-dim landmark vector.

    Args:
      landmarks_63: (63,) array. Layout: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]

    Returns:
      aug_landmarks_63: (63,) array
    """
    aug = landmarks_63.copy().reshape(21, 3)

    # 1. Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, aug.shape)
        aug += noise

    # 2. Random rotation (XY plane only — z unchanged)
    if rot_deg > 0:
        angle = np.random.uniform(-rot_deg, rot_deg)
        R = rotation_matrix_2d(angle)
        aug[:, :2] = aug[:, :2] @ R.T

    # 3. Random scaling (isotropic)
    if scale_range is not None:
        scale = np.random.uniform(*scale_range)
        aug[:, :2] *= scale  # scale x,y only (z is depth)

    # 4. Random translation
    trans_xy = np.random.uniform(-0.02, 0.02, 2)
    aug[:, :2] += trans_xy

    return aug.flatten()


def augment_sequence(seq, noise_std=AUG_NOISE_STD,
                     rot_deg=AUG_ROTATION_DEG, scale_range=AUG_SCALE_RANGE):
    """
    Augment a full 30-frame sequence.
    Same rotation/scale applied to all frames for temporal consistency.

    Args:
      seq: (30, 63) array

    Returns:
      aug_seq: (30, 63) array
    """
    seq_3d = seq.reshape(30, 21, 3)

    angle = np.random.uniform(-rot_deg, rot_deg) if rot_deg > 0 else 0
    scale = np.random.uniform(*scale_range) if scale_range is not None else 1.0
    trans_xy = np.random.uniform(-0.02, 0.02, 2)

    R = rotation_matrix_2d(angle)

    seq_3d[:, :, :2] = seq_3d[:, :, :2] @ R.T
    seq_3d[:, :, :2] *= scale
    seq_3d[:, :, :2] += trans_xy

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, seq_3d.shape)
        seq_3d += noise

    return seq_3d.reshape(30, 63)


def normalize_landmarks(landmarks_63):
    """Z-score normalize each landmark coordinate independently.
    Optional — MediaPipe already returns normalized values.
    Included for datasets that may need additional normalization."""
    arr = landmarks_63.reshape(21, 3)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 1e-8
    arr = (arr - mean) / std
    return arr.flatten()
