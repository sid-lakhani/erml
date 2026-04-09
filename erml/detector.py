"""Public API for ERML emotion detection.

The EmotionDetector class is the sole public interface of the erml package.
"""

import logging
import os

# Suppress TensorFlow C++ / CUDA / absl log spam before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict, List, Union

import cv2
import numpy as np

from erml.model import EMOTION_LABELS, build_model
from erml.preprocess import preprocess_face

logger = logging.getLogger(__name__)

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_MODEL_PATH = os.path.join(_ASSETS_DIR, "erml_v1.h5")
_CASCADE_PATH = os.path.join(_ASSETS_DIR, "haarcascade_frontalface_default.xml")


def _load_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier for face detection.

    Falls back to OpenCV's bundled cascade if the assets copy is absent.

    Returns:
        Loaded CascadeClassifier instance.

    Raises:
        RuntimeError: If no cascade file can be found.
    """
    if os.path.isfile(_CASCADE_PATH):
        path = _CASCADE_PATH
    else:
        bundled = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.isfile(bundled):
            logger.debug("Using bundled OpenCV cascade: %s", bundled)
            path = bundled
        else:
            raise RuntimeError(
                "Haar cascade not found. Expected at "
                f"{_CASCADE_PATH} or bundled with OpenCV."
            )
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier from: {path}")
    return cascade


class EmotionDetector:
    """Facial emotion recognition detector.

    Accepts a frame as a numpy array, file path, or PIL Image and returns
    structured emotion data for every detected face.

    Example::

        detector = EmotionDetector()
        results = detector.analyze("photo.jpg")
        for face in results:
            print(face["emotion"], face["confidence"])
    """

    def __init__(self) -> None:
        """Initialise the detector by loading the model and face cascade.

        The trained weights must exist at erml/assets/erml_v1.h5. If the
        file is absent the detector raises FileNotFoundError immediately so
        the error is surfaced at construction time rather than inference time.

        Raises:
            FileNotFoundError: If the model weights file does not exist.
            RuntimeError: If the face cascade cannot be loaded.
        """
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at: {_MODEL_PATH}\n"
                "Run erml/train.py to train and save the model, or download "
                "a pre-trained erml_v1.h5 into erml/assets/."
            )

        logger.info("Loading model from %s", _MODEL_PATH)
        self._model = build_model()
        self._model.load_weights(_MODEL_PATH)
        self._cascade = _load_cascade()
        logger.info("EmotionDetector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, frame: Union[np.ndarray, str, "PIL.Image.Image"]
    ) -> List[Dict]:
        """Detect faces in a frame and return per-face emotion scores.

        Args:
            frame: Input image as one of:
                - numpy array (BGR, as returned by OpenCV)
                - str file path to an image readable by OpenCV
                - PIL.Image object

        Returns:
            List of dicts, one per detected face, each containing:
                - ``emotion`` (str): Top predicted emotion label.
                - ``confidence`` (float): Confidence score in [0, 1].
                - ``all`` (dict): Scores for all 7 emotion labels.
                - ``bbox`` (dict): Bounding box with keys x, y, w, h.
            Returns an empty list if no face is detected.

        Raises:
            TypeError: If frame is not a supported input type.
            ValueError: If a file path is given but the image cannot be read.
        """
        bgr = self._to_bgr(frame)
        if bgr is None or bgr.size == 0:
            return []

        faces = self._detect_faces(bgr)
        if len(faces) == 0:
            return []

        results = []
        for x, y, w, h in faces:
            roi = bgr[y : y + h, x : x + w]
            preprocessed = preprocess_face(roi)
            predictions = self._model.predict(preprocessed, verbose=0)[0]

            top_idx = int(np.argmax(predictions))
            all_scores = {
                label: float(predictions[i])
                for i, label in enumerate(EMOTION_LABELS)
            }

            results.append(
                {
                    "emotion": EMOTION_LABELS[top_idx],
                    "confidence": float(predictions[top_idx]),
                    "all": all_scores,
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                }
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_bgr(self, frame: Union[np.ndarray, str, "PIL.Image.Image"]) -> np.ndarray:
        """Convert any supported input type to a BGR numpy array.

        Args:
            frame: Input image (numpy array, file path string, or PIL Image).

        Returns:
            BGR numpy array, or None if the input is empty / unreadable.

        Raises:
            TypeError: If frame is not a supported type.
            ValueError: If a file path cannot be read as an image.
        """
        if isinstance(frame, np.ndarray):
            return frame

        if isinstance(frame, str):
            if not os.path.isfile(frame):
                raise ValueError(f"Image file not found: {frame}")
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Could not read image file: {frame}")
            return img

        # PIL.Image — import lazily to avoid hard dependency at module level
        try:
            from PIL import Image as PILImage

            if isinstance(frame, PILImage.Image):
                img_rgb = np.array(frame.convert("RGB"))
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported frame type: {type(frame).__name__}. "
            "Expected numpy array, file path string, or PIL.Image."
        )

    def _detect_faces(self, bgr: np.ndarray) -> np.ndarray:
        """Run Haar cascade face detection on a BGR frame.

        Args:
            bgr: BGR numpy array.

        Returns:
            Array of shape (N, 4) with columns [x, y, w, h] for each
            detected face. Empty array if none are found.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if not isinstance(faces, np.ndarray):
            return np.empty((0, 4), dtype=np.int32)
        return faces
