"""
Export both GRU variants for 3-way comparison.

Builds:
  - Bidirectional GRU(128)  → cnn_gru_bidirectional.onnx
  - Unidirectional GRU(256) → cnn_gru_unidirectional.onnx

Both share CNN frontend. Only GRU direction differs.
Exports at opset=15 for Unity compatibility.
Saves .h5 weights for re-export without retrain.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import onnx
import tf2onnx

from config import (
    MODELS_DIR, SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE, LEARNING_RATE,
    AUGMENT,
)
from dataset_loader import load_dataset, get_class_weights
from preprocessing import augment_sequence


def build_cnn_gru(bidirectional=True):
    """Build CNN+GRU model with shared CNN frontend."""
    inputs = keras.Input(shape=(SEQ_LEN, FEATURE_DIM), name="input")

    x = layers.Conv1D(64, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    if bidirectional:
        x = layers.Bidirectional(
            layers.GRU(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
        )(x)
        suffix = "_bidirectional"
    else:
        x = layers.GRU(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
        suffix = "_unidirectional"

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name=f"CNN_GRU{suffix}"
    )
    return model, suffix


class SequenceAugmentationGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx]
        if self.augment:
            for i in range(len(X_batch)):
                X_batch[i] = augment_sequence(X_batch[i])
        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def train_and_export(data, bidirectional, force_retrain=False):
    """Train one variant and export ONNX + .h5. Skips if .h5 exists."""
    model, suffix = build_cnn_gru(bidirectional=bidirectional)
    onnx_path = os.path.join(MODELS_DIR, f"cnn_gru{suffix}.onnx")
    h5_path = os.path.join(MODELS_DIR, f"cnn_gru{suffix}.weights.h5")
    checkpoint_path = os.path.join(MODELS_DIR, f"cnn_gru{suffix}_best.weights.h5")

    if os.path.exists(h5_path) and not force_retrain:
        print(f"\n  Found existing weights: {h5_path}")
        print(f"  Loading and re-exporting ONNX...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.load_weights(h5_path)
    else:
        label = "BIDIRECTIONAL" if bidirectional else "UNIDIRECTIONAL"
        print(f"\n{'='*60}")
        print(f"  {label} GRU — Training")
        print(f"{'='*60}")

        (_, _), (_, _), (_, _), \
        (X_gru_train, y_train), (X_gru_val, y_val), (X_gru_test, _) = data

        class_weights = get_class_weights(y_train)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True, mode="max",
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy",
                save_best_only=True, save_weights_only=True,
            ),
        ]

        train_gen = SequenceAugmentationGenerator(
            X_gru_train, y_train, BATCH_SIZE, augment=AUGMENT
        )
        val_gen = SequenceAugmentationGenerator(
            X_gru_val, y_val, BATCH_SIZE, augment=False
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    # Load best checkpoint weights if available
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"  Loaded best checkpoint: {checkpoint_path}")

    # Export ONNX with opset=15 for Unity compatibility (before save_weights — main deliverable)
    print(f"  Exporting ONNX: {onnx_path}")
    spec = (tf.TensorSpec((None, SEQ_LEN, FEATURE_DIM), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=15, output_path=onnx_path,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX validation: OK")
    print(f"  Input:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Output: {[o.name for o in onnx_model.graph.output]}")

    # Save .h5 weights (secondary — for future re-exports without retrain)
    try:
        model.save_weights(h5_path)
        print(f"  Weights saved: {h5_path}")
    except Exception as e:
        print(f"  Weights save skipped ({e})")
    print()


def main():
    print("=" * 60)
    print("Export CNN+GRU — Both Variants")
    print("=" * 60)

    data = load_dataset()
    if data is None:
        return

    force_retrain = os.environ.get("FORCE_RETRAIN", "").lower() in ("1", "true", "yes")

    train_and_export(data, bidirectional=True, force_retrain=force_retrain)
    train_and_export(data, bidirectional=False, force_retrain=force_retrain)

    print("=" * 60)
    print("Done. Files in models/:")
    for f in sorted(os.listdir(MODELS_DIR)):
        if "cnn_gru" in f:
            print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
