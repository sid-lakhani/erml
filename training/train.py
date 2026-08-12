"""Training script for the ERML CNN model.

Not part of the installable erml package. Run directly from the repo root:

    python training/train.py

Requires the FER-2013 dataset under dataset/train/ and dataset/test/
organised by emotion subfolder, plus the training dependencies:

    uv pip install -r requirements-train.txt
"""

import logging
import os
import sys

# Suppress TensorFlow C++ / CUDA / absl log spam before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Guard TF imports with a clear error if not installed.
try:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as exc:
    raise ImportError(
        "Training requires TensorFlow. Install it with:\n"
        "    uv pip install -r requirements-train.txt\n"
        "or:\n"
        "    pip install erml[train]"
    ) from exc

# Make erml importable when running as a standalone script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from erml.constants import EMOTION_LABELS
from training.model import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATASET_TRAIN = os.path.join(_REPO_ROOT, "dataset", "train")
_DATASET_TEST = os.path.join(_REPO_ROOT, "dataset", "test")
_ASSETS_DIR = os.path.join(_REPO_ROOT, "erml", "assets")
_MODEL_OUT = os.path.join(_ASSETS_DIR, "erml_v1.h5")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.2


def _make_generators():
    """Create training and validation ImageDataGenerators.

    Returns:
        Tuple of (train_generator, val_generator) with augmentation applied
        to train and only rescaling applied to validation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=VAL_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        _DATASET_TRAIN,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        classes=EMOTION_LABELS,
        class_mode="sparse",
        batch_size=BATCH_SIZE,
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        _DATASET_TRAIN,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        classes=EMOTION_LABELS,
        class_mode="sparse",
        batch_size=BATCH_SIZE,
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def train() -> None:
    """Run the full training pipeline and save the best model.

    Loads data from dataset/train/, trains for up to EPOCHS epochs with
    early stopping, and saves the best checkpoint to erml/assets/erml_v1.h5.
    Prints final train and validation accuracy after completion.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    if not os.path.isdir(_DATASET_TRAIN):
        raise FileNotFoundError(
            f"Training dataset not found at: {_DATASET_TRAIN}\n"
            "Download FER-2013 and place it at dataset/train/ organised by "
            "emotion subfolder."
        )

    os.makedirs(_ASSETS_DIR, exist_ok=True)

    logger.info("Building model...")
    model = build_model()
    model.summary(print_fn=logger.info)

    logger.info("Loading dataset from %s", _DATASET_TRAIN)
    train_gen, val_gen = _make_generators()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=_MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    logger.info("Starting training (epochs=%d, batch=%d)...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info("Training complete.")
    logger.info("Final train accuracy : %.4f", final_train_acc)
    logger.info("Final val accuracy   : %.4f", final_val_acc)
    logger.info("Model saved to: %s", _MODEL_OUT)


if __name__ == "__main__":
    train()
