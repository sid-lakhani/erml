"""Keras CNN architecture for ERML training.

This module is only required during training. It is NOT part of the
installable erml package — inference uses ONNX Runtime instead.

Requires: pip install erml[train]  (or: uv pip install -r requirements-train.txt)
"""

try:
    from tensorflow import keras
    from tensorflow.keras import layers  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Training requires TensorFlow. Install it with:\n"
        "    uv pip install -r requirements-train.txt\n"
        "or:\n"
        "    pip install erml[train]"
    ) from exc

import sys
import os

# Make erml importable when running as a standalone script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from erml.constants import EMOTION_LABELS  # noqa: E402


def build_model() -> keras.Model:
    """Build and return the ERML CNN model.

    Architecture:
        - Input: 48x48x1 grayscale
        - Three Conv2D blocks with BatchNormalization and MaxPooling
        - Dense layers with Dropout regularization
        - Softmax output over 7 emotion classes

    Returns:
        Compiled Keras model ready for training or weight loading.
    """
    model = keras.Sequential(
        [
            # Explicit input layer — preferred over input_shape= in Conv2D (Keras 3+).
            keras.Input(shape=(48, 48, 1)),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Classifier head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(EMOTION_LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
