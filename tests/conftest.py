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
    """EmotionDetector with ONNX session and YuNet detection fully mocked.

    The ONNX session returns a fixed probability distribution where
    'happy' (index 3) has the highest score.
    """
    from erml.detector import EmotionDetector

    # softmax-like prediction: happy = 0.9, rest share 0.1
    predictions = np.array(
        [[0.02, 0.01, 0.02, 0.90, 0.02, 0.02, 0.01]], dtype=np.float32
    )
    mock_session = MagicMock()
    mock_session.run.return_value = [predictions]
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    # YuNet: detect() returns (retval, faces) where faces is (N, 15)
    faces_15col = np.array([[10, 10, 60, 60, *([0.0] * 10), 0.95]], dtype=np.float32)
    mock_yunet = MagicMock()
    mock_yunet.detect.return_value = (1, faces_15col)

    with (
        patch("erml.detector._resolve_model", return_value="/fake/model.onnx"),
        patch("erml.detector.ort.SessionOptions", return_value=MagicMock()),
        patch("erml.detector.ort.InferenceSession", return_value=mock_session),
        patch("erml.detector.cv2.FaceDetectorYN.create", return_value=mock_yunet),
    ):
        detector = EmotionDetector()

    detector._input_name = "input"
    yield detector
