import time
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MetricsTracker:
    """Per-model live metrics tracking for real-time display and logging."""

    def __init__(self, name, window_sec=60):
        self.name = name
        self.window_sec = window_sec

        self.last_pred = None
        self.last_conf = 0.0
        self.pred_changes = 0
        self.total_frames = 0
        self.false_activations = 0
        self.no_hand_frames = 0

        self.frame_capture_time = None
        self.inference_start = None

        self.successful_preds = []

        self.cpu_sample_interval = 1.0
        self._last_cpu_time = 0.0
        self._cpu_usage = 0.0
        self._process = None
        self.last_inf_ms = 0.0
        self.last_lat_ms = 0.0
        if HAS_PSUTIL:
            try:
                self._process = psutil.Process()
                self._cpu_usage = self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

    def start_frame(self):
        self.frame_capture_time = time.perf_counter()

    def start_inference(self):
        self.inference_start = time.perf_counter()

    def record_prediction(self, pred_id, confidence, is_accepted=True, hand_detected=True):
        now = time.perf_counter()
        self.total_frames += 1

        if not hand_detected:
            self.no_hand_frames += 1
            self.frame_capture_time = None
            self.inference_start = None
            return {
                "inference_time_ms": 0.0,
                "latency_ms": 0.0,
                "pred_changed": False,
                "false_activation": False,
            }

        inference_time = (now - self.inference_start) * 1000 if self.inference_start else 0.0
        latency = (now - self.frame_capture_time) * 1000 if self.frame_capture_time else 0.0
        self.last_inf_ms = inference_time
        self.last_lat_ms = latency

        pred_changed = False
        false_activation = False
        if self.last_pred is not None and pred_id != self.last_pred:
            self.pred_changes += 1
            pred_changed = True
            if self.last_conf > 0.6 and confidence > 0.6:
                self.false_activations += 1
                false_activation = True

        if not pred_changed and is_accepted:
            self.successful_preds.append(now)
            cutoff = now - self.window_sec
            self.successful_preds = [t for t in self.successful_preds if t > cutoff]

        self.last_pred = pred_id
        self.last_conf = confidence

        if HAS_PSUTIL and self._process and (now - self._last_cpu_time) >= self.cpu_sample_interval:
            try:
                raw = self._process.cpu_percent(interval=0)
                self._cpu_usage = raw / psutil.cpu_count()
            except Exception:
                pass
            self._last_cpu_time = now

        self.frame_capture_time = None
        self.inference_start = None

        return {
            "inference_time_ms": inference_time,
            "latency_ms": latency,
            "pred_changed": pred_changed,
            "false_activation": false_activation,
        }

    def reset(self):
        self.last_pred = None
        self.last_conf = 0.0
        self.pred_changes = 0
        self.total_frames = 0
        self.false_activations = 0
        self.no_hand_frames = 0
        self.successful_preds = []

    @property
    def jitter_rate(self):
        if self.total_frames < 2:
            return 0.0
        return (self.pred_changes / self.total_frames) * 100

    @property
    def false_activation_rate(self):
        if self.total_frames < 2:
            return 0.0
        return (self.false_activations / self.total_frames) * 100

    @property
    def gest_per_minute(self):
        now = time.perf_counter()
        cutoff = now - self.window_sec
        recent = [t for t in self.successful_preds if t > cutoff]
        return len(recent) / (self.window_sec / 60.0)

    @property
    def cpu_usage(self):
        if HAS_PSUTIL and self._process:
            return self._cpu_usage
        return None

    def summary_dict(self):
        return {
            "model": self.name,
            "total_frames": self.total_frames,
            "pred_changes": self.pred_changes,
            "jitter_rate_pct": round(self.jitter_rate, 2),
            "false_activations": self.false_activations,
            "false_activation_rate_pct": round(self.false_activation_rate, 2),
            "gest_per_minute": round(self.gest_per_minute, 2),
            "no_hand_frames": self.no_hand_frames,
            "cpu_usage_pct": round(self.cpu_usage, 1) if self.cpu_usage is not None else None,
        }
