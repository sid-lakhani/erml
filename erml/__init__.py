"""ERML — Emotion Recognition ML SDK.

Public API:
    EmotionDetector: Main class for facial emotion recognition.
    format_results: Pretty-print helper for analyze() output.
"""

from erml.detector import EmotionDetector
from erml.utils import format_results

__all__ = ["EmotionDetector", "format_results"]
__version__ = "0.1.0"
