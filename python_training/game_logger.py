import os
import json
import time


class GameLogger:
    """JSONL session logger — one JSON object per prediction line.

    Unity tails this file or reads it post-session for metrics computation.
    """

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"session_{session_id}.jsonl")
        self.file = open(self.path, "w", buffering=1)
        self._write_header(session_id)

    def _write_header(self, session_id):
        header = {
            "_type": "session_start",
            "session_id": session_id,
            "timestamp": time.time(),
            "gesture_labels_mapping": {
                0: "Open_Palm", 1: "Fist", 2: "Pinch", 3: "Point",
                4: "Two_Finger_V", 5: "Thumbs_Up", 6: "Swipe",
                7: "Push_Down", 8: "Twist_Left", 9: "Twist_Right",
                10: "Unknown",
            },
        }
        self.file.write(json.dumps(header) + "\n")

    def log_prediction(self, model, gesture, confidence, inference_ms, latency_ms,
                       jitter=False, false_activation=False, accepted=True,
                       mission_id="", mission_result="", hand_detected=True):
        entry = {
            "ts": round(time.time(), 3),
            "model": model,
            "gesture": gesture,
            "confidence": round(confidence, 4),
            "inference_ms": round(inference_ms, 2),
            "latency_ms": round(latency_ms, 2),
            "accepted": accepted,
            "jitter": jitter,
            "false_activation": false_activation,
            "hand_detected": hand_detected,
            "mission_id": mission_id,
            "mission_result": mission_result,
        }
        self.file.write(json.dumps(entry) + "\n")

    def log_metrics_summary(self, cnn_summary, gru_bi_summary):
        entry = {
            "_type": "metrics_summary",
            "ts": round(time.time(), 3),
            "cnn": cnn_summary,
            "cnn_gru_bi": gru_bi_summary,
        }
        self.file.write(json.dumps(entry) + "\n")

    def close(self):
        self.file.close()
