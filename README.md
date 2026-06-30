# MUTED — Hand Gesture Recognition

10-class gesture recognition using MediaPipe hand landmarks with CNN baseline and CNN+GRU (bidirectional/unidirectional) models. ONNX runtime for real-time inference.

Gestures: `Open_Palm`, `Fist`, `Pinch`, `Point`, `Two_Finger_V`, `Thumbs_Up`, `Swipe`, `Push_Down`, `Twist_Left`, `Twist_Right`, `Unknown`.

---

## Quick Start — Live Test Only (models pre-trained)

If models already exist in `python_training/models/`, skip training and go straight to live webcam test.

### Prerequisites

- Python 3.10+
- Webcam

### 1. Clone & enter

```bash
git clone <repo-url>
cd super-better-cnn-cnn-gru-bidirectional-bla-bla
```

### 2. Install dependencies

Minimum for live test:

```bash
pip install opencv-python mediapipe onnxruntime numpy
```

Optional extras:

- `psutil` — enables CPU usage meter in the metrics panel
- `onnxruntime-gpu` (DirectML on Windows) — GPU-accelerated inference. The runtime tries `DmlExecutionProvider` first, then falls back to CPU.

### 3. Verify models

Ensure these files exist:

```
python_training/models/cnn_model.onnx
python_training/models/cnn_gru_unidirectional.onnx
```

If missing, run the full pipeline (see below).

### 4. Run live test

```bash
cd python_training
python live_test.py
```

The script auto-downloads `hand_landmarker.task` from MediaPipe on first run if not present.

### Controls

| Key | Action       |
|-----|-------------|
| Q   | Quit + save session log |
| R   | Reset stats counters |

### Output: Session Logs

Every live session writes a JSONL log to:

```
python_training/logs/session_YYYYMMDD_HHMMSS.jsonl
```

Format: one JSON object per prediction line. Includes timestamps, model name, predicted gesture, confidence, inference/latency ms, jitter flags, and false activation flags. A `metrics_summary` entry is appended on quit.

The log path is printed on exit:

```
Session log saved: python_training/logs/session_20260630_104439.jsonl
```

Evaluation logs from `evaluate.py`:

```
python_training/logs/predictions_cnn.csv
python_training/logs/predictions_cnn_gru.csv
```

---

## Full Pipeline (Train from Scratch)

Run these in order from the `python_training/` directory.

### 1. Install full dependencies

```bash
pip install -r python_training/requirements.txt
```

### 2. Extract frames from raw videos

```bash
python extract_frames.py
```

Reads `datasets/raw/sessionNNN/GestureName/*.mp4`, outputs `_frames/gesture_N_name/video_name/*.jpg`.

### 3. Extract hand landmarks

```bash
python extract_landmarks.py
```

Processes frames through MediaPipe HandLandmarker, outputs `_landmarks/gesture_N_name/video_name.csv` (21 landmarks × 3 coords per row).

### 4. Train CNN baseline

```bash
python train_cnn.py
```

Architecture: `Conv1D(64) → Conv1D(128) → GlobalAvgPool → Dense(64) → Softmax(11)`. Exports `models/cnn_model.onnx`.

### 5. Train CNN+GRU (bidirectional)

```bash
python train_cnn_gru.py
```

Architecture: `Conv1D(64) → Conv1D(128) → BiGRU(128) → Dense(64) → Softmax(11)`. Exports `models/cnn_gru_model.onnx`.

### 6. Export both GRU variants (bidirectional + unidirectional)

```bash
python export_models.py
```

Trains or re-exports from existing `.h5` weights. Outputs:

- `models/cnn_gru_bidirectional.onnx`
- `models/cnn_gru_unidirectional.onnx`
- `models/cnn_gru_bidirectional.weights.h5`
- `models/cnn_gru_unidirectional.weights.h5`

Set `FORCE_RETRAIN=1` to force retraining.

### 7. Evaluate models

```bash
python evaluate.py
```

Reports accuracy, per-class precision/recall/F1, false activation rate, confidence calibration, confusion matrices, latency benchmark.

---

## Project Structure

```
python_training/
├── config.py                  # Paths, hyperparameters
├── dataset_loader.py          # Load landmarks, build sequences
├── preprocessing.py           # Augmentation (noise, rotation, scale)
├── extract_frames.py          # Raw video → frames
├── extract_landmarks.py       # Frames → MediaPipe landmarks
├── train_cnn.py               # Train CNN baseline
├── train_cnn_gru.py           # Train CNN+GRU bidirectional
├── export_models.py           # Both GRU variants
├── evaluate.py                # Offline metrics
├── live_test.py               # Real-time webcam demo
├── metrics_tracker.py         # Live metrics (jitter, latency, etc.)
├── game_logger.py             # JSONL session logger
├── datasets/raw/              # Raw session videos (not tracked)
├── models/                    # Trained ONNX + weights (generated)
├── _frames/                   # Extracted frames (generated)
├── _landmarks/                # Landmark CSVs (generated)
├── logs/                      # Session logs + eval CSVs
├── requirements.txt           # Full Python deps
└── hand_landmarker.task       # MediaPipe model (auto-downloaded)
```
