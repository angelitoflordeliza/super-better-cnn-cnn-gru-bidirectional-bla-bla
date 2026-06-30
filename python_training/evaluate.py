"""
Evaluate and compare CNN vs CNN+GRU models with thesis metrics.

Offline metrics:
  - Accuracy, precision, recall, F1 per class
  - False activation rate (high-confidence wrong predictions)
  - Per-class confidence calibration
  - Confusion matrices
  - Model agreement
  - Inference latency benchmark

Note: Temporal jitter requires contiguous per-video data.
      Live test captures jitter via MetricsTracker.
      Run `live_test.py` for real-time jitter, lat, gest/min.
"""

import os
import csv
import time
import numpy as np
import onnxruntime as ort

from config import (
    MODELS_DIR, GESTURE_LABELS, NUM_CLASSES, CONFIDENCE_THRESHOLD,
    SEQ_LEN, FEATURE_DIM, LOG_DIR,
)
from dataset_loader import load_dataset


def load_onnx_session(model_path):
    providers = [
        ("DmlExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def evaluate_model(session, input_name, output_name, X):
    predictions = []
    confidences = []
    probabilities = []

    for i in range(len(X)):
        inp = X[i:i + 1].astype(np.float32)
        output = session.run([output_name], {input_name: inp})[0]
        probs = output[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        predictions.append(pred)
        confidences.append(conf)
        probabilities.append(probs)

    return (np.array(predictions), np.array(confidences),
            np.array(probabilities))


def benchmark_latency(session, input_name, output_name, X, n_runs=100):
    sample = X[:1].astype(np.float32)
    for _ in range(10):
        session.run([output_name], {input_name: sample})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: sample})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def print_classification_report(y_true, y_pred, label_names):
    from sklearn.metrics import classification_report
    print(classification_report(
        y_true, y_pred, labels=range(len(label_names)),
        target_names=label_names, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=GESTURE_LABELS,
                    yticklabels=GESTURE_LABELS)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  Saved: {save_path}")
        plt.close()
    except ImportError:
        pass


def compute_false_activation_rate(y_true, y_pred, confs, threshold):
    """Count high-confidence predictions that are wrong."""
    mask = confs >= threshold
    total_high_conf = np.sum(mask)
    false_act = np.sum((y_pred[mask] != y_true[mask]))
    if total_high_conf == 0:
        return 0.0, 0, 0
    return false_act / total_high_conf, false_act, total_high_conf


def per_class_false_activations(y_true, y_pred, confs, threshold,
                                label_names):
    """False activations broken down by true class."""
    fa_by_class = {}
    for cid in range(len(label_names)):
        mask = (y_true == cid) & (confs >= threshold)
        n = np.sum(mask)
        wrong = np.sum(y_pred[mask] != y_true[mask])
        conf_ok = np.sum(y_pred[mask] == y_true[mask])
        fa_by_class[label_names[cid]] = {
            "samples": int(n),
            "false_activations": int(wrong),
            "correct_high_conf": int(conf_ok),
            "fa_rate": round(wrong / n, 4) if n > 0 else 0.0,
        }
    return fa_by_class


def confidence_calibration(y_true, y_pred, confs, label_names):
    """Average confidence for correct vs incorrect predictions per class."""
    cal = {}
    for cid in range(len(label_names)):
        mask = y_true == cid
        n = np.sum(mask)
        if n == 0:
            continue
        correct = y_pred[mask] == y_true[mask]
        wrong = y_pred[mask] != y_true[mask]
        avg_conf_correct = np.mean(confs[mask][correct]) if np.any(correct) else 0.0
        avg_conf_wrong = np.mean(confs[mask][wrong]) if np.any(wrong) else 0.0
        cal[label_names[cid]] = {
            "n": int(n),
            "avg_conf_correct": round(float(avg_conf_correct), 4),
            "avg_conf_wrong": round(float(avg_conf_wrong), 4),
        }
    return cal


def save_predictions_csv(path, y_true, y_pred, confs, probs, label_names,
                         model_name):
    """Save per-prediction details for external analysis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["true_class", "pred_class", "confidence", "correct"] + [
            f"prob_{l}" for l in label_names
        ]
        writer.writerow(header)

        for i in range(len(y_true)):
            row = [
                label_names[y_true[i]],
                label_names[y_pred[i]],
                round(float(confs[i]), 4),
                int(y_true[i] == y_pred[i]),
            ] + [round(float(p), 4) for p in probs[i]]
            writer.writerow(row)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("MUTED — Model Evaluation & Comparison")
    print("=" * 60)

    data = load_dataset()
    if data is None:
        return

    (_, _), (_, _), (_, y_test), \
    (_, _), (_, _), (X_gru_test, _) = data

    from dataset_loader import load_dataset as load
    (_, _), (_, _), (X_cnn_test, y_test_cnn), \
    (_, _), (_, _), (_, _) = load()

    print(f"\nTest samples: CNN={len(X_cnn_test)}, GRU={len(X_gru_test)}")

    # ── Load models ──────────────────────────────────────────────────────────
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.onnx")
    gru_path = os.path.join(MODELS_DIR, "cnn_gru_model.onnx")

    if not os.path.exists(cnn_path):
        print(f"[ERROR] CNN model not found: {cnn_path}")
        return
    if not os.path.exists(gru_path):
        print(f"[ERROR] CNN+GRU model not found: {gru_path}")
        return

    print("\nLoading CNN model...")
    cnn_session, cnn_in, cnn_out = load_onnx_session(cnn_path)
    print(f"  Provider: {cnn_session.get_providers()[0]}")

    print("Loading CNN+GRU model...")
    gru_session, gru_in, gru_out = load_onnx_session(gru_path)
    print(f"  Provider: {gru_session.get_providers()[0]}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n--- CNN (Baseline) Evaluation ---")
    cnn_preds, cnn_confs, cnn_probs = evaluate_model(
        cnn_session, cnn_in, cnn_out, X_cnn_test)
    cnn_accuracy = np.mean(cnn_preds == y_test_cnn)
    print(f"  Accuracy: {cnn_accuracy:.4f}")
    print_classification_report(y_test_cnn, cnn_preds, GESTURE_LABELS)
    plot_confusion_matrix(y_test_cnn, cnn_preds,
                          "CNN Confusion Matrix",
                          os.path.join(MODELS_DIR, "cnn_confusion.png"))
    save_predictions_csv(
        os.path.join(LOG_DIR, "predictions_cnn.csv"),
        y_test_cnn, cnn_preds, cnn_confs, cnn_probs,
        GESTURE_LABELS, "CNN")

    print("\n--- CNN+GRU (Experimental) Evaluation ---")
    gru_preds, gru_confs, gru_probs = evaluate_model(
        gru_session, gru_in, gru_out, X_gru_test)
    gru_accuracy = np.mean(gru_preds == y_test)
    print(f"  Accuracy: {gru_accuracy:.4f}")
    print_classification_report(y_test, gru_preds, GESTURE_LABELS)
    plot_confusion_matrix(y_test, gru_preds,
                          "CNN+GRU Confusion Matrix",
                          os.path.join(MODELS_DIR, "cnn_gru_confusion.png"))
    save_predictions_csv(
        os.path.join(LOG_DIR, "predictions_cnn_gru.csv"),
        y_test, gru_preds, gru_confs, gru_probs,
        GESTURE_LABELS, "CNN+GRU")

    # ── Agreement ────────────────────────────────────────────────────────────
    agreement = np.mean(cnn_preds == gru_preds[:len(cnn_preds)])
    print(f"\n--- Model Agreement ---")
    print(f"  Agreement rate: {agreement:.4f}")

    # ── False Activation Rate ────────────────────────────────────────────────
    print(f"\n--- False Activation Rate (threshold={CONFIDENCE_THRESHOLD}) ---")
    for name, preds, confs, y_true in [
        ("CNN", cnn_preds, cnn_confs, y_test_cnn),
        ("CNN+GRU", gru_preds, gru_confs, y_test),
    ]:
        fa_rate, fa_count, total_hc = compute_false_activation_rate(
            y_true, preds, confs, CONFIDENCE_THRESHOLD)
        print(f"  {name}:")
        print(f"    High-conf predictions (>{CONFIDENCE_THRESHOLD}): "
              f"{total_hc}/{len(confs)} ({100*total_hc/len(confs):.1f}%)")
        print(f"    False activations:  {fa_count}/{total_hc} "
              f"({fa_rate*100:.2f}%)")

    # ── Per-Class False Activation ───────────────────────────────────────────
    print(f"\n--- Per-Class False Activation Analysis ---")
    for name, preds, confs, y_true in [
        ("CNN", cnn_preds, cnn_confs, y_test_cnn),
        ("CNN+GRU", gru_preds, gru_confs, y_test),
    ]:
        print(f"  {name}:")
        per_class = per_class_false_activations(
            y_true, preds, confs, CONFIDENCE_THRESHOLD, GESTURE_LABELS)
        for cls, info in per_class.items():
            if info["false_activations"] > 0:
                print(f"    {cls:15s}  FA: {info['false_activations']:3d}/"
                      f"{info['samples']:3d}  ({info['fa_rate']*100:.1f}%)")

    # ── Confidence Calibration ───────────────────────────────────────────────
    print(f"\n--- Confidence Calibration ---")
    for name, preds, confs, y_true in [
        ("CNN", cnn_preds, cnn_confs, y_test_cnn),
        ("CNN+GRU", gru_preds, gru_confs, y_test),
    ]:
        print(f"  {name}:")
        cal = confidence_calibration(y_true, preds, confs, GESTURE_LABELS)
        for cls, info in cal.items():
            print(f"    {cls:15s}  correct avg: {info['avg_conf_correct']:.3f}  "
                  f"wrong avg: {info['avg_conf_wrong']:.3f}  "
                  f"(n={info['n']})")

    # ── Threshold Sweep ──────────────────────────────────────────────────────
    print(f"\n--- Threshold Effect (threshold={CONFIDENCE_THRESHOLD}) ---")
    for name, preds, confs, y_true in [
        ("CNN", cnn_preds, cnn_confs, y_test_cnn),
        ("CNN+GRU", gru_preds, gru_confs, y_test),
    ]:
        below_thresh = np.sum(confs < CONFIDENCE_THRESHOLD)
        corrected = (confs >= CONFIDENCE_THRESHOLD) & (
            preds == y_true[:len(preds)])
        thresh_accuracy = np.sum(corrected) / len(y_true)
        print(f"  {name}:")
        print(f"    Below threshold: {below_thresh}/{len(confs)} "
              f"({100*below_thresh/len(confs):.1f}%)")
        print(f"    Accuracy with threshold: {thresh_accuracy:.4f}")

    # ── Latency Benchmark ───────────────────────────────────────────────────
    print(f"\n--- Latency Benchmark (100 runs each) ---")
    for name, sess, inp_name, out_name, X in [
        ("CNN", cnn_session, cnn_in, cnn_out, X_cnn_test),
        ("CNN+GRU", gru_session, gru_in, gru_out, X_gru_test),
    ]:
        mean_ms, std_ms = benchmark_latency(
            sess, inp_name, out_name, X, n_runs=100)
        print(f"  {name}:  {mean_ms:.2f} ± {std_ms:.2f} ms per inference")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  CNN Accuracy:      {cnn_accuracy:.4f}")
    print(f"  CNN+GRU Accuracy:  {gru_accuracy:.4f}")
    print(f"  Model Agreement:   {agreement:.4f}")
    print(f"  Improvement:       {(gru_accuracy - cnn_accuracy)*100:+.2f} pp")
    print(f"{'=' * 60}")
    print()
    print("Note: Temporal jitter requires contiguous per-video data.")
    print("      Run live_test.py for real-time jitter, latency, gest/min.")


if __name__ == "__main__":
    main()
