"""Tests for erml.preprocess module."""

import numpy as np
import pytest

from erml.preprocess import preprocess_face

# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------


def test_output_shape_from_bgr():
    """BGR (H x W x 3) input should produce (1, 48, 48, 1) output."""
    bgr = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    out = preprocess_face(bgr)
    assert out.shape == (1, 48, 48, 1)


def test_output_shape_from_grayscale_2d():
    """2D grayscale (H x W) input should produce (1, 48, 48, 1) output."""
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    out = preprocess_face(gray)
    assert out.shape == (1, 48, 48, 1)


def test_output_shape_from_grayscale_3d():
    """Grayscale with trailing channel (H x W x 1) should produce (1, 48, 48, 1)."""
    gray = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)
    out = preprocess_face(gray)
    assert out.shape == (1, 48, 48, 1)


def test_output_shape_from_bgra():
    """BGRA (H x W x 4) input should produce (1, 48, 48, 1) output."""
    bgra = np.random.randint(0, 256, (60, 60, 4), dtype=np.uint8)
    out = preprocess_face(bgra)
    assert out.shape == (1, 48, 48, 1)


def test_output_dtype_is_float32():
    """Output dtype must be float32."""
    bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    out = preprocess_face(bgr)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Value range tests
# ---------------------------------------------------------------------------


def test_output_range_min_zero():
    """Output minimum value should be >= 0.0."""
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    out = preprocess_face(img)
    assert out.min() >= 0.0


def test_output_range_max_one():
    """Output maximum value should be <= 1.0."""
    img = np.full((48, 48, 3), 255, dtype=np.uint8)
    out = preprocess_face(img)
    assert out.max() <= 1.0


def test_all_white_gives_one():
    """All-white image should normalize to exactly 1.0."""
    img = np.full((48, 48), 255, dtype=np.uint8)
    out = preprocess_face(img)
    assert np.allclose(out, 1.0)


def test_all_black_gives_zero():
    """All-black image should normalize to exactly 0.0."""
    img = np.zeros((48, 48), dtype=np.uint8)
    out = preprocess_face(img)
    assert np.allclose(out, 0.0)


def test_pixel_value_scaling():
    """A pixel of value 128 should normalise to approximately 128/255."""
    img = np.full((48, 48), 128, dtype=np.uint8)
    out = preprocess_face(img)
    expected = 128.0 / 255.0
    assert abs(out.mean() - expected) < 1e-4


# ---------------------------------------------------------------------------
# Resize correctness
# ---------------------------------------------------------------------------


def test_non_square_input_resizes_to_48x48():
    """Non-square input should still produce 48x48 spatial dimensions."""
    img = np.random.randint(0, 256, (100, 30, 3), dtype=np.uint8)
    out = preprocess_face(img)
    assert out.shape == (1, 48, 48, 1)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_raises_on_non_array():
    """Passing a non-array should raise ValueError."""
    with pytest.raises(ValueError, match="numpy array"):
        preprocess_face("not an array")  # type: ignore[arg-type]


def test_raises_on_unsupported_channels():
    """Array with 2 channels should raise ValueError."""
    img = np.zeros((48, 48, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="channels"):
        preprocess_face(img)


def test_raises_on_4d_array():
    """4D array (batch) should raise ValueError."""
    img = np.zeros((1, 48, 48, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        preprocess_face(img)
