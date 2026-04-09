"""Shared pytest fixtures for ERML tests."""

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


SAMPLE_FACES_DIR = os.path.join(os.path.dirname(__file__), "sample_faces")


@pytest.fixture()
def happy_face_path() -> str:
    """Path to a sample happy face image."""
    return os.path.join(SAMPLE_FACES_DIR, "happy_face.jpg")


@pytest.fixture()
def no_face_path() -> str:
    """Path to a sample image containing no face."""
    return os.path.join(SAMPLE_FACES_DIR, "no_face.jpg")


@pytest.fixture()
def bgr_face_array() -> np.ndarray:
    """Synthetic BGR numpy array shaped like a real face region."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture()
def mock_detector():
    """EmotionDetector with model loading and face detection mocked out.

    The model's predict() returns a fixed probability distribution where
    'happy' (index 3) has the highest score.
    """
    import erml.detector as det_module
    from erml.detector import EmotionDetector
    from erml.model import EMOTION_LABELS

    # Build a softmax-like prediction: happy = 0.9, rest share 0.1
    predictions = np.array([0.02, 0.01, 0.02, 0.90, 0.02, 0.02, 0.01], dtype=np.float32)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([predictions])

    mock_cascade = MagicMock()
    # detectMultiScale returns one face bounding box
    mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 60, 60]])
    mock_cascade.empty.return_value = False

    with (
        patch.object(det_module, "_MODEL_PATH", "/fake/model.h5"),
        patch("os.path.isfile", return_value=True),
        patch("erml.detector.build_model", return_value=mock_model),
        patch("erml.detector._load_cascade", return_value=mock_cascade),
    ):
        detector = EmotionDetector()
        detector._model = mock_model
        detector._cascade = mock_cascade
        yield detector
