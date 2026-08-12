"""Tests for erml.download — atomic downloader and checksum verification."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from erml.download import (
    _sha256,
    _verify_checksum,
    ensure_model,
)

# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------


def test_sha256_of_known_content(tmp_path):
    """_sha256 must match hashlib's direct computation."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"erml-test-content-12345")
    expected = hashlib.sha256(b"erml-test-content-12345").hexdigest()
    assert _sha256(f) == expected


# ---------------------------------------------------------------------------
# _verify_checksum
# ---------------------------------------------------------------------------


def test_verify_checksum_passes(tmp_path):
    """_verify_checksum does not raise when digest matches."""
    f = tmp_path / "model.bin"
    data = b"fake-model-data"
    f.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    _verify_checksum(f, digest, "model.bin")  # should not raise


def test_verify_checksum_fails_and_deletes(tmp_path):
    """_verify_checksum raises ValueError and deletes file on mismatch."""
    f = tmp_path / "model.bin"
    f.write_bytes(b"corrupt-data")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_checksum(f, "deadbeef" * 8, "model.bin")
    assert not f.exists(), "Corrupt file should be deleted after mismatch"


def test_verify_checksum_skips_placeholder(tmp_path):
    """_verify_checksum skips verification for the placeholder dev hash."""
    f = tmp_path / "model.bin"
    f.write_bytes(b"anything")
    # Should not raise even though content doesn't match placeholder
    _verify_checksum(f, "REPLACE_WITH_REAL_SHA256_AFTER_RELEASE", "model.bin")


# ---------------------------------------------------------------------------
# ensure_model
# ---------------------------------------------------------------------------


def test_ensure_model_raises_for_unknown_filename():
    """ensure_model raises KeyError for unregistered filenames."""
    with pytest.raises(KeyError, match="unknown_model.onnx"):
        ensure_model("unknown_model.onnx")


def test_ensure_model_returns_existing_asset(tmp_path):
    """ensure_model returns the local asset path if it already exists."""
    data = b"fake-weights"
    digest = hashlib.sha256(data).hexdigest()

    # Write a fake asset
    (tmp_path / "erml_v1.h5").write_bytes(data)

    with patch.dict(
        "erml.download.KNOWN_MODELS",
        {"erml_v1.h5": {"url": "http://example.com/erml_v1.h5", "sha256": digest}},
    ):
        result = ensure_model("erml_v1.h5", assets_dir=tmp_path)

    assert result == tmp_path / "erml_v1.h5"


def test_ensure_model_returns_cached_file(tmp_path):
    """ensure_model returns the cache path when the asset dir has no file."""
    data = b"cached-weights"
    digest = hashlib.sha256(data).hexdigest()

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "erml_v1.h5").write_bytes(data)

    with (
        patch.dict(
            "erml.download.KNOWN_MODELS",
            {
                "erml_v1.h5": {
                    "url": "http://example.com/erml_v1.h5",
                    "sha256": digest,
                }
            },
        ),
        patch("erml.download._CACHE_DIR", cache_dir),
    ):
        result = ensure_model("erml_v1.h5", assets_dir=asset_dir)

    assert result == cache_dir / "erml_v1.h5"


def test_ensure_model_download_called_when_missing(tmp_path):
    """ensure_model calls _download when no local or cached file exists."""
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dest_path = cache_dir / "erml_v1.h5"

    with (
        patch.dict(
            "erml.download.KNOWN_MODELS",
            {
                "erml_v1.h5": {
                    "url": "http://example.com/erml_v1.h5",
                    "sha256": "abc123",
                }
            },
        ),
        patch("erml.download._CACHE_DIR", cache_dir),
        patch("erml.download._download", return_value=dest_path) as mock_dl,
    ):
        result = ensure_model("erml_v1.h5", assets_dir=asset_dir)

    mock_dl.assert_called_once()
    assert result == dest_path
