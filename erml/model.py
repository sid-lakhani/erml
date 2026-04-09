"""CNN architecture definition for ERML emotion recognition.

This module only defines the model architecture. No training logic here.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


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
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
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
