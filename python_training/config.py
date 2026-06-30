import os

# ── Gesture Configuration ──────────────────────────────────────────────────
GESTURE_LABELS = [
    "Open_Palm",      # 0
    "Fist",           # 1
    "Pinch",          # 2
    "Point",          # 3
    "Two_Finger_V",   # 4
    "Thumbs_Up",      # 5
    "Swipe",          # 6
    "Push_Down",      # 7
    "Twist_Left",     # 8
    "Twist_Right",    # 9
    "Unknown",        # 10
]
NUM_CLASSES = len(GESTURE_LABELS)  # 11
UNKNOWN_ID = 10

# Map folder names (as they appear in raw/session/GestureName/) to gesture IDs
GESTURE_FOLDER_MAP = {
    "open_palm": 0,
    "fist": 1,
    "pinch": 2,
    "point": 3,
    "two_finger_v": 4,
    "thumbs_up": 5,
    "swipe": 6,
    "push_down": 7,
    "twist_left": 8,
    "twist_right": 9,
}

# ── Data Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
RAW_DIR = os.path.join(DATASET_DIR, "raw")           # session/gesture/*.mp4
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "_frames")
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "_landmarks")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Temporal Parameters ──────────────────────────────────────────────────────
FPS = 30
SEQ_LEN = 30           # sliding window width (~1s at 30 FPS)
SEQ_STRIDE = 1          # slide 1 frame at a time
FEATURE_DIM = 63        # 21 landmarks * 3 (x, y, z)

# ── Training ────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1        # remaining 0.1 is test
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
CONFIDENCE_THRESHOLD = 0.6

# ── Display ──────────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
CAM_WIDTH = 640
CAM_HEIGHT = 480
METRICS_PANEL_X = 680        # right panel starts here

# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS_WINDOW_SEC = 60       # rolling window for gest/min

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Augmentation ────────────────────────────────────────────────────────────
AUGMENT = True
AUG_NOISE_STD = 0.01   # Gaussian noise on landmarks
AUG_ROTATION_DEG = 5   # max rotation in degrees
AUG_SCALE_RANGE = (0.95, 1.05)  # random scaling
