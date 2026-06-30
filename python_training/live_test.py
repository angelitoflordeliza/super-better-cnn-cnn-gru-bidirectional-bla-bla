"""
Live webcam gesture recognition with live metrics panel.
Press 'q' to quit, 'r' to reset stats.
"""

import os
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort

from config import (
    MODELS_DIR, GESTURE_LABELS, NUM_CLASSES, CONFIDENCE_THRESHOLD,
    SEQ_LEN, FEATURE_DIM, UNKNOWN_ID,
    WINDOW_WIDTH, WINDOW_HEIGHT, CAM_WIDTH, CAM_HEIGHT,
    METRICS_PANEL_X, LOG_DIR,
)
from metrics_tracker import MetricsTracker
from game_logger import GameLogger

MODEL_PATH_CNN = os.path.join(MODELS_DIR, "cnn_model.onnx")
MODEL_PATH_GRU_UNI = os.path.join(MODELS_DIR, "cnn_gru_unidirectional.onnx")
TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
CAM_ID = 0
SHOW_LANDMARKS = True

# ── Dark Modern Palette ──────────────────────────────────
BG = (13, 13, 13)
CARD_BG = (26, 26, 26)
CARD_BORDER = (42, 42, 42)
TEXT_PRIMARY = (232, 232, 232)
TEXT_MUTED = (128, 128, 128)
TEXT_DIM = (70, 70, 70)
ACCENT_CNN = (255, 212, 0)      # cyan in BGR = (255, 212, 0)
ACCENT_GRU = (255, 0, 212)    # magenta
ACCENT_GAME = (0, 184, 255)     # amber
ACCENT_GOOD = (118, 230, 0)    # green
ACCENT_WARN = (0, 212, 255)    # yellow
ACCENT_ERROR = (60, 60, 200)   # red
DISCLAIMER_BG = (20, 20, 36)    # very dark blue tint

FONT_SM = cv2.FONT_HERSHEY_SIMPLEX
FONT_MD = cv2.FONT_HERSHEY_DUPLEX
FONT_LG = cv2.FONT_HERSHEY_TRIPLEX
AA = cv2.LINE_AA

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17),
]


def init_mediapipe():
    base_options = mp.tasks.BaseOptions(model_asset_path=TASK_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_mp_callback,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def init_onnx(model_path):
    providers = [
        ("DmlExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"  {os.path.basename(model_path)} -> {session.get_providers()[0]}")
    return session


def predict_cnn(session, landmarks):
    inp = landmarks.astype(np.float32).reshape(1, -1)
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]
    probs = out[0]
    pred = np.argmax(probs)
    return int(pred), float(probs[pred])


def predict_gru(session, buffer):
    inp = buffer.astype(np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]
    probs = out[0]
    pred = np.argmax(probs)
    return int(pred), float(probs[pred])


_mp_result = None
_mp_lock = threading.Lock()


def _mp_callback(result, output_image, timestamp_ms):
    global _mp_result
    with _mp_lock:
        _mp_result = result


def extract_landmarks(result):
    if not result or not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]
    arr = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr.flatten()


def draw_hand_skeleton(canvas, result, ox=0, oy=0, mirror_x=False):
    if not result or not result.hand_landmarks:
        return
    h, w = canvas.shape[:2]
    lm_list = result.hand_landmarks[0]
    for conn in HAND_CONNECTIONS:
        x0 = int((1.0 - lm_list[conn[0]].x if mirror_x else lm_list[conn[0]].x) * w) + ox
        y0 = int(lm_list[conn[0]].y * h) + oy
        x1 = int((1.0 - lm_list[conn[1]].x if mirror_x else lm_list[conn[1]].x) * w) + ox
        y1 = int(lm_list[conn[1]].y * h) + oy
        cv2.line(canvas, (x0, y0), (x1, y1), (0, 210, 120), 2, AA)
    for lm in lm_list:
        cx = int((1.0 - lm.x if mirror_x else lm.x) * w) + ox
        cy = int(lm.y * h) + oy
        cv2.circle(canvas, (cx, cy), 4, (120, 230, 255), -1, AA)


def draw_card(canvas, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_BORDER, 1)


def draw_conf_bar(canvas, x, y, w, conf, color):
    bw = int(conf * w)
    cv2.rectangle(canvas, (x, y), (x + w, y + 10), (50, 50, 50), -1)
    if bw > 2:
        cv2.rectangle(canvas, (x, y), (x + bw, y + 10), color, -1)
        # highlight line on top
        bright = tuple(min(255, c + 40) for c in color)
        cv2.rectangle(canvas, (x, y), (x + bw - 2, y + 3), bright, -1)
    cv2.putText(canvas, f"{conf:.2f}", (x + w + 8, y + 10),
                FONT_SM, 1.0, TEXT_PRIMARY, 1, AA)


def draw_metrics_card(canvas, x, y, w, title, label, conf, tracker, accent, H=175):
    draw_card(canvas, x, y, w, H)
    cx = x + 16
    cy = y + 18

    # Title
    cv2.putText(canvas, title, (cx, cy), FONT_MD, 0.5, accent, 1, AA)
    cy += 14

    # Separator
    cv2.line(canvas, (cx, cy + 2), (x + w - 16, cy + 2), CARD_BORDER, 1)

    # Prediction + confidence bar
    cy += 16
    cv2.putText(canvas, label, (cx, cy + 8), FONT_SM, 1.1, TEXT_PRIMARY, 1, AA)
    bar_x = cx + 155
    draw_conf_bar(canvas, bar_x, cy, 145, conf, accent)

    # Metrics 2×3 grid — stacked labels above values
    cpu_val = f"{tracker.cpu_usage:.1f}%" if tracker.cpu_usage is not None else "N/A"
    items = [
        ("Inference Time", f"{tracker.last_inf_ms:.2f}ms"),
        ("Jitter Rate",    f"{tracker.jitter_rate:.1f}%"),
        ("Latency",        f"{tracker.last_lat_ms:.0f}ms"),
        ("False Act. Rt.", f"{tracker.false_activation_rate:.1f}%"),
        ("Gestures/min",   f"{tracker.gest_per_minute:.1f}"),
        ("CPU Usage",      cpu_val),
    ]
    cy += 28
    col_w = (w - 32) // 2
    row_h = 34
    for i, (lbl, val) in enumerate(items):
        col = i // 3
        row = i % 3
        rx = cx + col * col_w
        ry = cy + row * row_h
        cv2.putText(canvas, lbl, (rx, ry), FONT_SM, 0.55, TEXT_MUTED, 1, AA)
        cv2.putText(canvas, val, (rx, ry + 18), FONT_SM, 0.8, TEXT_PRIMARY, 1, AA)


def draw_game_card(canvas, x, y, w, elapsed_s, frames, gest_min, resets):
    H = 100
    draw_card(canvas, x, y, w, H)
    cx = x + 16
    cy = y + 18

    cv2.putText(canvas, "GAME METRICS", (cx, cy), FONT_MD, 0.5, ACCENT_GAME, 1, AA)
    cy += 12
    cv2.line(canvas, (cx, cy + 2), (x + w - 16, cy + 2), CARD_BORDER, 1)
    cy += 16

    mins = int(elapsed_s // 60)
    secs = int(elapsed_s % 60)
    duration_s = f"{mins:02d}:{secs:02d}"

    items = [
        ("Duration", duration_s),
        ("Gest/min", f"{gest_min:.1f}"),
        ("Frames", f"{frames}"),
        ("Resets", f"{resets}"),
    ]
    col_w = (w - 32) // 2
    for i, (lbl, val) in enumerate(items):
        col = i // 2
        row = i % 2
        rx = cx + col * col_w
        ry = cy + row * 22
        cv2.putText(canvas, lbl, (rx, ry), FONT_SM, 0.8, TEXT_MUTED, 1, AA)
        cv2.putText(canvas, val, (rx + 65, ry), FONT_SM, 0.9, TEXT_PRIMARY, 1, AA)


def draw_system_bar(canvas, x, y, w, fps, log_path):
    H = 52
    draw_card(canvas, x, y, w, H)
    cx = x + 16
    cy = y + 18

    fps_color = ACCENT_GOOD if fps > 25 else ACCENT_WARN
    cv2.putText(canvas, f"FPS  {fps:.1f}", (cx, cy), FONT_SM, 1.0, fps_color, 1, AA)

    log_short = os.path.basename(log_path) if log_path else "no log"
    cv2.putText(canvas, log_short, (cx + 140, cy), FONT_SM, 0.75, TEXT_MUTED, 1, AA)


def draw_hint(canvas, x, y):
    cv2.putText(canvas, "[ Q ] quit    [ R ] reset", (x, y + 14),
                FONT_SM, 0.8, TEXT_DIM, 1, AA)


def main():
    print("MUTED — Live Gesture Recognition with Metrics")
    print("=" * 60)

    print("Init MediaPipe HandLandmarker...")
    hand_landmarker = init_mediapipe()

    print("Loading ONNX models...")
    cnn_session = init_onnx(MODEL_PATH_CNN)
    gru_uni_session = init_onnx(MODEL_PATH_GRU_UNI)

    print("Opening webcam...")
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cnn_tracker = MetricsTracker("CNN")
    gru_uni_tracker = MetricsTracker("CNN+GRU Uni")
    logger = GameLogger(LOG_DIR)
    session_start = time.time()

    cv2.namedWindow("MUTED - Live Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MUTED - Live Gesture Recognition",
                     WINDOW_WIDTH, WINDOW_HEIGHT)

    frame_buffer = []
    gru_counter = 0
    ts = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0.0
    cnn_label = "---"
    cnn_conf = 0.0
    gru_uni_label = "Buffering..."
    gru_uni_conf = 0.0

    CX = 20
    CY = 75
    PX = METRICS_PANEL_X
    PW = WINDOW_WIDTH - PX - 24
    CW = CAM_WIDTH

    print("\nLive gesture recognition started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        display_frame = cv2.flip(frame, 1)
        cnn_tracker.start_frame()
        gru_uni_tracker.start_frame()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts += 1
        hand_landmarker.detect_async(mp_image, ts)

        with _mp_lock:
            result = _mp_result

        # Build canvas
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[:] = BG

        # Title
        cv2.putText(canvas, "MUTED", (CX, 42), FONT_LG, 0.9, TEXT_PRIMARY, 1, AA)
        cv2.putText(canvas, "Live Gesture Recognition", (CX + 105, 42),
                    FONT_SM, 1.1, TEXT_MUTED, 1, AA)

        # Camera area background
        cv2.rectangle(canvas, (CX - 1, CY - 1),
                      (CX + CW + 1, CY + CAM_HEIGHT + 1), CARD_BORDER, 1)

        # Place camera frame (mirrored for selfie view, but inference runs on original)
        canvas[CY:CY + CAM_HEIGHT, CX:CX + CW] = display_frame

        landmarks_63 = extract_landmarks(result) if result else None
        hand_detected = landmarks_63 is not None

        if hand_detected:
            frame_buffer.append(landmarks_63)
            if len(frame_buffer) > SEQ_LEN:
                frame_buffer.pop(0)

            # Draw skeleton BEFORE inference — minimal delay = no vibration
            if SHOW_LANDMARKS:
                cam_area = canvas[CY:CY + CAM_HEIGHT, CX:CX + CW]
                draw_hand_skeleton(cam_area, result, mirror_x=True)

            cnn_tracker.start_inference()
            cnn_pred_raw, cnn_conf = predict_cnn(cnn_session, landmarks_63)
            if cnn_conf < CONFIDENCE_THRESHOLD:
                cnn_pred_raw = UNKNOWN_ID
            cnn_label = GESTURE_LABELS[cnn_pred_raw]
            cnn_meta = cnn_tracker.record_prediction(
                cnn_pred_raw, cnn_conf, hand_detected=True)

            if len(frame_buffer) == SEQ_LEN:
                buffer_np = np.array(frame_buffer, dtype=np.float32)

                if gru_counter % 2 == 0:
                    gru_uni_tracker.start_inference()
                    gru_uni_pred_raw, gru_uni_conf = predict_gru(gru_uni_session, buffer_np)
                    if gru_uni_conf < CONFIDENCE_THRESHOLD:
                        gru_uni_pred_raw = UNKNOWN_ID
                    gru_uni_label = GESTURE_LABELS[gru_uni_pred_raw]

                gru_uni_meta = gru_uni_tracker.record_prediction(
                    gru_uni_pred_raw, gru_uni_conf, hand_detected=True)
                gru_counter += 1
            else:
                gru_uni_tracker.record_prediction(
                    None, 0.0, is_accepted=False, hand_detected=False)

            logger.log_prediction(
                model="cnn", gesture=cnn_label, confidence=cnn_conf,
                inference_ms=cnn_meta["inference_time_ms"],
                latency_ms=cnn_meta["latency_ms"],
                jitter=cnn_meta["pred_changed"],
                false_activation=cnn_meta["false_activation"],
                hand_detected=True)
            if len(frame_buffer) == SEQ_LEN:
                logger.log_prediction(
                    model="cnn_gru_uni", gesture=gru_uni_label, confidence=gru_uni_conf,
                    inference_ms=gru_uni_meta["inference_time_ms"],
                    latency_ms=gru_uni_meta["latency_ms"],
                    jitter=gru_uni_meta["pred_changed"],
                    false_activation=gru_uni_meta["false_activation"],
                    hand_detected=True)

        else:
            frame_buffer.clear()
            gru_counter = 0
            cnn_label = "---"
            cnn_conf = 0.0
            gru_uni_label = "No hand"
            gru_uni_conf = 0.0
            cnn_tracker.record_prediction(
                None, 0.0, is_accepted=False, hand_detected=False)
            gru_uni_tracker.record_prediction(
                None, 0.0, is_accepted=False, hand_detected=False)
            cv2.putText(canvas, "No hand detected",
                        (CX + 10, CY + 30), FONT_SM, 1.0, ACCENT_ERROR, 1, AA)

        # ── Right panel ──────────────────────────────────────────────────
        CARD_GAP = 185

        draw_metrics_card(canvas, PX, CY, PW,
                          "CNN  BASELINE", cnn_label, cnn_conf,
                          cnn_tracker, ACCENT_CNN)

        draw_metrics_card(canvas, PX, CY + CARD_GAP, PW,
                          "CNN+GRU  UNIDIRECTIONAL", gru_uni_label, gru_uni_conf,
                          gru_uni_tracker, ACCENT_GRU)

        # Game metrics
        game_y = CY + CARD_GAP * 2
        elapsed = time.time() - session_start
        gest_min = cnn_tracker.gest_per_minute
        resets = cnn_tracker.false_activations
        draw_game_card(canvas, PX, game_y, PW,
                       elapsed, cnn_tracker.total_frames,
                       gest_min, resets)

        # System
        draw_system_bar(canvas, PX, game_y + 116, PW,
                        fps_display, logger.path)

        draw_hint(canvas, PX, game_y + 176)

        cv2.imshow("MUTED - Live Gesture Recognition", canvas)

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cnn_tracker.record_prediction(None, 0.0, hand_detected=False)
            logger.log_metrics_summary(
                cnn_tracker.summary_dict(),
                gru_uni_tracker.summary_dict(),
            )
            break
        elif key == ord('r'):
            cnn_tracker.reset()
            gru_uni_tracker.reset()
            gru_counter = 0
            print("  Stats reset.")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    logger.close()
    print(f"\nSession log saved: {logger.path}")
    print("Done.")


if __name__ == "__main__":
    main()
