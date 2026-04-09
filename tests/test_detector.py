"""Tests for erml.detector.EmotionDetector."""

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import erml.detector as det_module
from erml.detector import EmotionDetector
from erml.model import EMOTION_LABELS

SAMPLE_FACES_DIR = os.path.join(os.path.dirname(__file__), "sample_faces")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_detector(face_boxes=None):
    """Return an EmotionDetector with model and cascade fully mocked.

    Args:
        face_boxes: numpy array of shape (N, 4) returned by detectMultiScale.
            Defaults to one face at (10, 10, 60, 60).

    Returns:
        Tuple of (detector, mock_model, mock_cascade).
    """
    if face_boxes is None:
        face_boxes = np.array([[10, 10, 60, 60]])

    predictions = np.array(
        [0.02, 0.01, 0.02, 0.90, 0.02, 0.02, 0.01], dtype=np.float32
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([predictions])

    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = face_boxes
    mock_cascade.empty.return_value = False

    with (
        patch("os.path.isfile", return_value=True),
        patch("erml.detector.build_model", return_value=mock_model),
        patch("erml.detector._load_cascade", return_value=mock_cascade),
    ):
        detector = EmotionDetector()

    return detector, mock_model, mock_cascade


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_raises_if_model_missing(tmp_path):
    """EmotionDetector.__init__ raises FileNotFoundError when weights absent."""
    fake_path = str(tmp_path / "missing.h5")
    with patch.object(det_module, "_MODEL_PATH", fake_path):
        with pytest.raises(FileNotFoundError, match="erml_v1.h5"):
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


def test_analyze_numpy_result_keys():
    """Each result dict must contain emotion, confidence, all, and bbox."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    assert set(result.keys()) == {"emotion", "confidence", "all", "bbox"}


def test_analyze_emotion_is_string():
    """emotion field must be a string from EMOTION_LABELS."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    assert result["emotion"] in EMOTION_LABELS


def test_analyze_top_emotion_matches_highest_score():
    """emotion must be the label with the highest score in all."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    result = detector.analyze(frame)[0]
    top_label = max(result["all"], key=result["all"].get)
    assert result["emotion"] == top_label


def test_analyze_confidence_float_in_range():
    """confidence must be a float in [0, 1]."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    conf = detector.analyze(frame)[0]["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_analyze_all_scores_keys():
    """all dict must contain exactly the 7 emotion labels."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    all_scores = detector.analyze(frame)[0]["all"]
    assert set(all_scores.keys()) == set(EMOTION_LABELS)


def test_analyze_all_scores_sum_to_one():
    """Softmax scores in all must sum to approximately 1.0."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    all_scores = detector.analyze(frame)[0]["all"]
    assert abs(sum(all_scores.values()) - 1.0) < 1e-4


def test_analyze_bbox_keys():
    """bbox must contain x, y, w, h keys."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = detector.analyze(frame)[0]["bbox"]
    assert set(bbox.keys()) == {"x", "y", "w", "h"}


def test_analyze_bbox_values_are_ints():
    """bbox values must be Python ints."""
    detector, _, _ = _make_mock_detector()
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = detector.analyze(frame)[0]["bbox"]
    for key, val in bbox.items():
        assert isinstance(val, int), f"bbox[{key!r}] is {type(val).__name__}, expected int"


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
    """analyze() returns [] when no face is detected."""
    no_faces = np.empty((0, 4), dtype=np.int32)
    detector, _, _ = _make_mock_detector(face_boxes=no_faces)
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

    detector, _, mock_cascade = _make_mock_detector()
    # Ensure cascade finds a face so the full pipeline runs
    mock_cascade.detectMultiScale.return_value = np.array([[5, 5, 40, 40]])
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
