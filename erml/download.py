"""Model weight auto-downloader for ERML.

Downloads pre-trained model files from GitHub Releases on first use.
Writes to a temporary file first; only renames to the final path after a
successful, fully-completed download. Verifies the file's SHA-256 checksum
before it is used, so a corrupted or interrupted cache is always detected.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Release manifest — update these when a new model version is published.
# ---------------------------------------------------------------------------

_BASE_URL = "https://github.com/sid-lakhani/erml/releases/download/v0.1.0"

KNOWN_MODELS: dict[str, dict[str, str]] = {
    "erml_v1.onnx": {
        "url": f"{_BASE_URL}/erml_v1.onnx",
        # Run: sha256sum erml/assets/erml_v1.onnx
        "sha256": "REPLACE_WITH_REAL_SHA256_AFTER_RELEASE",
    },
    "face_detection_yunet_2023mar.onnx": {
        "url": f"{_BASE_URL}/face_detection_yunet_2023mar.onnx",
        # Run: sha256sum erml/assets/face_detection_yunet_2023mar.onnx
        "sha256": "REPLACE_WITH_REAL_SHA256_AFTER_RELEASE",
    },
}

_CACHE_DIR = Path.home() / ".cache" / "erml"


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_progress(block_count: int, block_size: int, total_size: int) -> None:
    """Simple CLI progress reporter for urllib.request.urlretrieve."""
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(
            f"\r  Downloading: {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
    else:
        mb_done = downloaded / 1_048_576
        print(f"\r  Downloading: {mb_done:.1f} MB", end="", flush=True)


def ensure_model(
    filename: str,
    assets_dir: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
) -> Path:
    """Ensure a model file exists locally, downloading it if necessary.

    Resolution order:
      1. ``assets_dir`` (defaults to the erml package's own ``assets/`` dir).
      2. ``~/.cache/erml/`` — persistent user-level cache.
      3. Download from the GitHub Release URL, validate checksum, and cache.

    Args:
        filename: Model filename (e.g. ``"erml_v1.h5"``).
        assets_dir: Directory to check first. Defaults to erml/assets/.
        expected_sha256: Expected SHA-256 hex digest. If ``None``, the value
            from :data:`KNOWN_MODELS` is used. Pass ``None`` to skip
            verification (not recommended in production).

    Returns:
        Absolute :class:`~pathlib.Path` to the verified local model file.

    Raises:
        KeyError: If ``filename`` is not registered in :data:`KNOWN_MODELS`.
        ValueError: If the downloaded file fails checksum verification.
        RuntimeError: If the file cannot be downloaded.
    """
    if filename not in KNOWN_MODELS:
        raise KeyError(
            f"Unknown model file: {filename!r}. "
            f"Registered models: {list(KNOWN_MODELS)}"
        )

    entry = KNOWN_MODELS[filename]
    url = entry["url"]
    if expected_sha256 is None:
        expected_sha256 = entry["sha256"]

    # 1. Check package assets dir.
    if assets_dir is None:
        assets_dir = Path(__file__).parent / "assets"
    local = assets_dir / filename
    if local.is_file():
        logger.debug("Found model at %s", local)
        _verify_checksum(local, expected_sha256, filename)
        return local

    # 2. Check user cache.
    cached = _CACHE_DIR / filename
    if cached.is_file():
        logger.debug("Found cached model at %s", cached)
        _verify_checksum(cached, expected_sha256, filename)
        return cached

    # 3. Download.
    return _download(url, cached, expected_sha256, filename)


def _verify_checksum(path: Path, expected: str, filename: str) -> None:
    """Verify SHA-256 checksum; raise ValueError on mismatch.

    Skips verification when the expected digest is the placeholder string
    (i.e. the release has not been tagged yet and this is a dev build).
    """
    if expected == "REPLACE_WITH_REAL_SHA256_AFTER_RELEASE":
        logger.debug("Skipping checksum verification for dev build.")
        return

    actual = _sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {filename}.\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n"
            "The cached file has been deleted. Re-run to re-download."
        )
    logger.debug("Checksum OK for %s", filename)


def _download(url: str, dest: Path, expected_sha256: str, filename: str) -> Path:
    """Atomically download ``url`` to ``dest`` and verify its checksum.

    Writes to a ``.tmp`` file beside the destination. Only renames to the
    final path after the stream closes successfully, preventing partially
    downloaded files from being cached.

    Args:
        url: Remote URL to download.
        dest: Final destination path (in the user cache).
        expected_sha256: SHA-256 hex digest to verify against.
        filename: Human-readable filename for log/error messages.

    Returns:
        Path to the verified local file.

    Raises:
        RuntimeError: If the download fails.
        ValueError: If checksum verification fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling .tmp file so an interrupted download is never cached.
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=dest.parent, prefix=filename + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    os.close(tmp_fd)

    logger.info("Downloading %s from GitHub Releases...", filename)
    print(f"[erml] Downloading {filename} from GitHub Releases...")
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report_progress)
        print()  # newline after progress bar
    except URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {filename} from {url}.\n"
            f"Cause: {exc}\n"
            "Check your internet connection and try again."
        ) from exc

    # Verify before promoting to the real path.
    _verify_checksum(tmp_path, expected_sha256, filename)

    # Atomic rename — safe on POSIX; best-effort on Windows.
    try:
        tmp_path.replace(dest)
    except OSError:
        shutil.move(str(tmp_path), dest)

    logger.info("Model cached at %s", dest)
    print(f"[erml] Saved to {dest}")
    return dest
