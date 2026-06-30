"""
Train CNN baseline model on hand landmark data.

Architecture:
  Input(63) → Reshape(21, 3) → Conv1D(64, k=3) → BatchNorm → ReLU
                             → Conv1D(128, k=3) → BatchNorm → ReLU → GlobalAvgPool
                             → Dense(64, ReLU) → Dropout(0.2)
                             → Dense(11, Softmax)

Exports: models/cnn_model.onnx
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import onnx
import tf2onnx

from config import (
    MODELS_DIR, FEATURE_DIM, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE, LEARNING_RATE,
    GESTURE_LABELS, AUGMENT,
)
from dataset_loader import load_dataset, get_class_weights
from preprocessing import augment_landmarks


def build_cnn_model():
    inputs = keras.Input(shape=(FEATURE_DIM,), name="input")

    # Reshape flat 63 → (21 landmarks, 3 coords) so Conv1D slides across landmarks
    x = layers.Reshape((21, 3), name="landmark_reshape")(inputs)

    x = layers.Conv1D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Gesture")
    return model


class AugmentationGenerator(keras.utils.Sequence):
    """Data generator with on-the-fly augmentation for CNN."""

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
                X_batch[i] = augment_landmarks(X_batch[i])

        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def main():
    print("=" * 60)
    print("MUTED — Train CNN Gesture Classifier (Baseline)")
    print("=" * 60)

    data = load_dataset()
    if data is None:
        return

    (X_cnn_train, y_train), (X_cnn_val, y_val), (X_cnn_test, y_test), \
    (_, _), (_, _), (_, _) = data

    class_weights = get_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")

    model = build_cnn_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    train_gen = AugmentationGenerator(X_cnn_train, y_train, BATCH_SIZE, augment=AUGMENT)
    val_gen = AugmentationGenerator(X_cnn_val, y_val, BATCH_SIZE, augment=False)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_cnn_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "cnn_model.onnx")
    spec = (tf.TensorSpec((None, FEATURE_DIM), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=18,
        output_path=onnx_path,
    )
    print(f"\nONNX model exported to: {onnx_path}")

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX validation: OK")
    print(f"  Input:  {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Output: {[out.name for out in onnx_model.graph.output]}")

    # Save training history plot
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history["accuracy"], label="train")
        ax1.plot(history.history["val_accuracy"], label="val")
        ax1.set_title("Accuracy")
        ax1.legend()

        ax2.plot(history.history["loss"], label="train")
        ax2.plot(history.history["val_loss"], label="val")
        ax2.set_title("Loss")
        ax2.legend()

        plot_path = os.path.join(MODELS_DIR, "cnn_training_history.png")
        plt.savefig(plot_path)
        print(f"Training plot: {plot_path}")
    except ImportError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
