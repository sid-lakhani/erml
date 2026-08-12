"""ERML — Emotion Recognition ML SDK.

Public API:
    EmotionDetector: Main class for facial emotion recognition.
    format_results: Pretty-print helper for analyze() output.
    FacePrediction: Structured prediction result for a single face.
    BoundingBox: Bounding box coordinates for a detected face.
    EMOTION_LABELS: Ordered list of the 7 emotion class names.
"""

from erml.constants import EMOTION_LABELS
from erml.detector import EmotionDetector
from erml.schemas import BoundingBox, FacePrediction
from erml.utils import format_results

__all__ = [
    "EmotionDetector",
    "format_results",
    "FacePrediction",
    "BoundingBox",
    "EMOTION_LABELS",
]
__version__ = "0.1.0"
