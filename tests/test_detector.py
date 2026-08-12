"""Tests for erml.detector.EmotionDetector."""

from __future__ import annotations

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import erml.detector  # noqa: F401 — kept for patch target resolution
from erml.detector import EmotionDetector
from erml.constants import EMOTION_LABELS
from erml.schemas import FacePrediction

SAMPLE_FACES_DIR = os.path.join(os.path.dirname(__file__), "sample_faces")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_detector(face_boxes=None):
    """Return an EmotionDetector with ONNX session and YuNet fully mocked.

    Args:
        face_boxes: numpy array of shape (N, 4) returned by YuNet detect().
            Defaults to one face at (10, 10, 60, 60).

    Returns:
        Tuple of (detector, mock_session, mock_yunet).
    """
    if face_boxes is None:
        face_boxes = np.array([[10, 10, 60, 60]])

    # ONNX session mock — outputs softmax probabilities (happy wins at idx 3)
    predictions = np.array(
        [[0.02, 0.01, 0.02, 0.90, 0.02, 0.02, 0.01]], dtype=np.float32
    )
    mock_session = MagicMock()
    mock_session.run.return_value = [predictions]
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    # YuNet mock — detect() returns (retval, faces_array)
    # YuNet returns [x, y, w, h, *landmarks, score], so pad to 15 columns.
    def _make_yunet_faces(boxes):
        if boxes is None or len(boxes) == 0:
            return None
        n = len(boxes)
        faces = np.zeros((n, 15), dtype=np.float32)
        faces[:, :4] = boxes
        return faces

    yunet_faces = _make_yunet_faces(face_boxes)
    mock_yunet = MagicMock()
    mock_yunet.detect.return_value = (1, yunet_faces)

    with (
        patch("erml.detector._resolve_model", return_value="/fake/model.onnx"),
        patch("erml.detector.ort.SessionOptions", return_value=MagicMock()),
        patch("erml.detector.ort.InferenceSession", return_value=mock_session),
        patch("erml.detector.cv2.FaceDetectorYN.create", return_value=mock_yunet),
    ):
        detector = EmotionDetector()

    # Ensure the session input name is properly set
    mock_session.get_inputs.return_value[0].name = "input"
    detector._input_name = "input"

    return detector, mock_session, mock_yunet


def _make_no_face_detector():
    """EmotionDetector mocked to return no faces from YuNet."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    mock_yunet = MagicMock()
    mock_yunet.detect.return_value = (0, None)

    with (
        patch("erml.detector._resolve_model", return_value="/fake/model.onnx"),
        patch("erml.detector.ort.SessionOptions", return_value=MagicMock()),
        patch("erml.detector.ort.InferenceSession", return_value=mock_session),
        patch("erml.detector.cv2.FaceDetectorYN.create", return_value=mock_yunet),
    ):
        detector = EmotionDetector()

    detector._input_name = "input"
    return detector, mock_session, mock_yunet


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_raises_if_model_download_fails():
    """EmotionDetector raises RuntimeError when model download fails."""
    with patch(
        "erml.detector._resolve_model",
        side_effect=RuntimeError("Download failed: no network"),
    ):
        with pytest.raises(RuntimeError, match="Download failed"):
            EmotionDetector()


# ---------------------------------------------------------------------------
# analyze() — numpy array input
# ---------------------------------------------------------------------------


def test_analyze_numpy_returns_list():
    """analyze() on a BGR array with one face returns a non-empty list."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    results = detector.analyze(frame)
    assert isinstance(results, list)
    assert len(results) == 1


def test_analyze_returns_face_prediction_objects():
    """Each result must be a FacePrediction instance."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    assert isinstance(result, FacePrediction)


def test_analyze_numpy_result_keys():
    """model_dump() on each result must contain emotion, confidence, all, bbox."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0].model_dump()
    assert set(result.keys()) == {"emotion", "confidence", "all", "bbox"}


def test_analyze_emotion_is_string():
    """emotion field must be a string from EMOTION_LABELS."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    assert result.emotion in EMOTION_LABELS


def test_analyze_top_emotion_matches_highest_score():
    """emotion must be the label with the highest score in all."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    top_label = max(result.all, key=result.all.get)
    assert result.emotion == top_label


def test_analyze_confidence_float_in_range():
    """confidence must be a float in [0, 1]."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    conf = detector.analyze(frame)[0].confidence
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_analyze_all_scores_keys():
    """all dict must contain exactly the 7 emotion labels."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    all_scores = detector.analyze(frame)[0].all
    assert set(all_scores.keys()) == set(EMOTION_LABELS)


def test_analyze_all_scores_sum_to_one():
    """Softmax scores in all must sum to approximately 1.0."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    all_scores = detector.analyze(frame)[0].all
    assert abs(sum(all_scores.values()) - 1.0) < 1e-4


def test_analyze_bbox_keys():
    """bbox must contain x, y, w, h attributes."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = detector.analyze(frame)[0].bbox
    assert hasattr(bbox, "x")
    assert hasattr(bbox, "y")
    assert hasattr(bbox, "w")
    assert hasattr(bbox, "h")


def test_analyze_bbox_values_are_ints():
    """bbox values must be Python ints."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = detector.analyze(frame)[0].bbox
    for key in ("x", "y", "w", "h"):
        val = getattr(bbox, key)
        assert isinstance(val, int), f"bbox.{key} is {type(val).__name__}, expected int"


def test_analyze_model_dump_roundtrip():
    """model_dump() must produce a plain dict matching expected structure."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    d = detector.analyze(frame)[0].model_dump()
    assert isinstance(d, dict)
    assert isinstance(d["bbox"], dict)
    assert set(d["bbox"].keys()) == {"x", "y", "w", "h"}
    assert isinstance(d["emotion"], str)
    assert isinstance(d["confidence"], float)


def test_analyze_multiple_faces():
    """analyze() returns one result per detected face."""
    two_faces = np.array([[10, 10, 50, 50], [100, 80, 50, 50]])
    detector, _, _ = _make_mock_detector(face_boxes=two_faces)
    frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    results = detector.analyze(frame)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# analyze() — no face detected
# ---------------------------------------------------------------------------


def test_analyze_returns_empty_list_when_no_face():
    """analyze() returns [] when YuNet detects no faces."""
    detector, _, _ = _make_no_face_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)
    assert result == []


def test_analyze_does_not_raise_on_empty_array():
    """analyze() never raises when passed an empty frame."""
    detector, _, _ = _make_mock_detector()
    result = detector.analyze(np.array([]))
    assert result == []


# ---------------------------------------------------------------------------
# analyze() — file path input
# ---------------------------------------------------------------------------


def test_analyze_file_path_input():
    """analyze() accepts a file path string and reads the image."""
    happy_path = os.path.join(SAMPLE_FACES_DIR, "happy_face.jpg")
    if not os.path.isfile(happy_path):
        pytest.skip("Sample face image not available")

    detector, _, mock_yunet = _make_mock_detector()
    faces_15col = np.array([[5, 5, 40, 40, *([0.0] * 10), 0.9]], dtype=np.float32)
    mock_yunet.detect.return_value = (1, faces_15col)
    result = detector.analyze(happy_path)
    assert isinstance(result, list)


def test_analyze_raises_on_missing_file():
    """analyze() raises ValueError when the file path does not exist."""
    detector, _, _ = _make_mock_detector()
    with pytest.raises(ValueError, match="not found"):
        detector.analyze("/nonexistent/path/image.jpg")


def test_analyze_raises_on_invalid_image_file(tmp_path):
    """analyze() raises ValueError when the file is not a valid image."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"not an image")
    detector, _, _ = _make_mock_detector()
    with pytest.raises(ValueError, match="Could not read"):
        detector.analyze(str(bad_file))


# ---------------------------------------------------------------------------
# analyze() — PIL input
# ---------------------------------------------------------------------------


def test_analyze_pil_input():
    """analyze() accepts a PIL.Image and returns a list."""
    pytest.importorskip("PIL")
    from PIL import Image

    pil_img = Image.fromarray(
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8), mode="RGB"
    )
    detector, _, _ = _make_mock_detector()
    result = detector.analyze(pil_img)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# analyze() — type errors
# ---------------------------------------------------------------------------


def test_analyze_raises_on_unsupported_type():
    """analyze() raises TypeError for unsupported input types."""
    detector, _, _ = _make_mock_detector()
    with pytest.raises(TypeError, match="Unsupported frame type"):
        detector.analyze(12345)  # type: ignore[arg-type]
