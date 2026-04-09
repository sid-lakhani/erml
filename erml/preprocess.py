"""Image preprocessing utilities for ERML.

Accepts a face ROI and returns a normalised numpy array ready for model inference.
"""

import cv2
import numpy as np


def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """Preprocess a face region-of-interest for model inference.

    Converts to grayscale if needed, resizes to 48x48, normalises pixel
    values to [0, 1], and reshapes to (1, 48, 48, 1) for batch inference.

    Args:
        face_roi: Numpy array representing a cropped face region. May be
            BGR (H x W x 3) or already grayscale (H x W) or (H x W x 1).

    Returns:
        Float32 numpy array of shape (1, 48, 48, 1) with values in [0, 1].

    Raises:
        ValueError: If face_roi is not a valid numpy array or has an
            unsupported number of channels.
    """
    if not isinstance(face_roi, np.ndarray):
        raise ValueError(
            f"face_roi must be a numpy array, got {type(face_roi).__name__}"
        )

    if face_roi.ndim == 2:
        gray = face_roi
    elif face_roi.ndim == 3:
        channels = face_roi.shape[2]
        if channels == 1:
            gray = face_roi[:, :, 0]
        elif channels == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                f"Unsupported number of channels: {channels}. Expected 1, 3, or 4."
            )
    else:
        raise ValueError(
            f"face_roi must be 2D or 3D array, got ndim={face_roi.ndim}"
        )

    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 48, 48, 1)
