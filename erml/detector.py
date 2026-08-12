"""Public API for ERML emotion detection.

The EmotionDetector class is the sole public interface of the erml package.
Uses ONNX Runtime for inference (no TensorFlow required at runtime) and
OpenCV's YuNet face detector for significantly more accurate face detection
than the legacy Haar cascade approach.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

# Suppress OpenCV / ONNX verbosity.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import onnxruntime as ort

from erml.constants import EMOTION_LABELS
from erml.download import ensure_model
from erml.preprocess import preprocess_face
from erml.schemas import BoundingBox, FacePrediction

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"
_MODEL_FILENAME = "erml_v1.onnx"
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _resolve_model(filename: str) -> Path:
    """Return path to a model file, downloading it on first run if absent.

    Checks the package assets dir first, then ~/.cache/erml/.

    Args:
        filename: Model filename registered in :data:`~erml.download.KNOWN_MODELS`.

    Returns:
        Absolute path to the verified local file.
    """
    local = _ASSETS_DIR / filename
    if local.is_file():
        return local
    return ensure_model(filename, assets_dir=_ASSETS_DIR)


class EmotionDetector:
    """Facial emotion recognition detector.

    Accepts a frame as a numpy array, file path, or PIL Image and returns
    structured emotion data for every detected face.

    Uses **ONNX Runtime** for inference (no TensorFlow dependency) and
    **YuNet** (OpenCV's built-in deep face detector) for significantly more
    accurate face localisation than the legacy Haar cascade approach.

    Example::

        detector = EmotionDetector()
        results = detector.analyze("photo.jpg")

        # Access fields directly (IDE autocomplete supported)
        for face in results:
            print(face.emotion, face.confidence)

        # Export to plain dicts (e.g. for JSON serialisation)
        raw = [r.model_dump() for r in results]
    """

    def __init__(self) -> None:
        """Initialise the detector by loading both ONNX models.

        On first run, if either model file is not found locally, it is
        downloaded automatically from GitHub Releases, verified via SHA-256,
        and cached in ``~/.cache/erml/`` for all future runs.

        Raises:
            RuntimeError: If a download fails or a checksum does not match.
        """
        # Resolve / download emotion model and YuNet face detector.
        model_path = _resolve_model(_MODEL_FILENAME)
        yunet_path = _resolve_model(_YUNET_FILENAME)

        logger.info("Loading ONNX inference session from %s", model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress ONNX Runtime noise
        sess_opts.intra_op_num_threads = 1  # prevent thread thrashing in concurrent web servers
        self._session = ort.InferenceSession(
            str(model_path), 
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name

        logger.info("Loading YuNet face detector from %s", yunet_path)
        # Placeholder size — resized per-frame in _detect_faces.
        self._yunet = cv2.FaceDetectorYN.create(
            model=str(yunet_path),
            config="",
            input_size=(640, 480),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        logger.info("EmotionDetector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame: Union[np.ndarray, str, PILImage],
    ) -> List[FacePrediction]:
        """Detect faces in a frame and return per-face emotion scores.

        Args:
            frame: Input image as one of:
                - numpy array (BGR, as returned by OpenCV)
                - str file path to an image readable by OpenCV
                - PIL.Image object

        Returns:
            List of :class:`~erml.schemas.FacePrediction` objects, one per
            detected face. Each object exposes ``.emotion``, ``.confidence``,
            ``.all``, and ``.bbox`` as attributes with IDE autocomplete.

            To convert to a plain dictionary::

                raw_dicts = [r.model_dump() for r in results]

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

        results: List[FacePrediction] = []
        for x, y, w, h in faces:
            roi = bgr[y : y + h, x : x + w]
            preprocessed = preprocess_face(roi)
            outputs = self._session.run(None, {self._input_name: preprocessed})
            predictions = outputs[0][0]

            top_idx = int(np.argmax(predictions))
            all_scores = {
                label: float(predictions[i]) for i, label in enumerate(EMOTION_LABELS)
            }

            results.append(
                FacePrediction(
                    emotion=EMOTION_LABELS[top_idx],
                    confidence=float(predictions[top_idx]),
                    all=all_scores,
                    bbox=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_bgr(
        self,
        frame: Union[np.ndarray, str, PILImage],
    ) -> np.ndarray:
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

        # PIL.Image — import lazily to avoid hard dependency at module level.
        try:
            from PIL import Image as _PILImage

            if isinstance(frame, _PILImage.Image):
                img_rgb = np.array(frame.convert("RGB"))
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported frame type: {type(frame).__name__}. "
            "Expected numpy array, file path string, or PIL.Image."
        )

    def _detect_faces(self, bgr: np.ndarray) -> np.ndarray:
        """Run YuNet face detection on a BGR frame.

        YuNet is a deep-learning face detector built into OpenCV (>= 4.8)
        that handles rotation, occlusion, and varied lighting far better
        than the legacy Haar cascade approach.

        Args:
            bgr: BGR numpy array.

        Returns:
            Array of shape (N, 4) with columns [x, y, w, h] for each
            detected face, clipped to the image boundaries. Empty array
            if none are found.
        """
        h, w = bgr.shape[:2]
        # YuNet requires input_size to match the frame dimensions.
        self._yunet.setInputSize((w, h))

        _, faces = self._yunet.detect(bgr)
        if faces is None or len(faces) == 0:
            return np.empty((0, 4), dtype=np.int32)

        # YuNet returns [x, y, w, h, *landmarks, score] — take first 4 cols.
        boxes = faces[:, :4].astype(np.int32)

        # Clip to image bounds to avoid out-of-bounds ROI slices.
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 1, w - boxes[:, 0])
        boxes[:, 3] = np.clip(boxes[:, 3], 1, h - boxes[:, 1])

        return boxes
