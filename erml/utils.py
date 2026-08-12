"""Display utilities for ERML results."""

from __future__ import annotations

from typing import List, Union

from erml.schemas import FacePrediction


def format_results(results: List[Union[FacePrediction, dict]]) -> str:
    """Format analyze() output as a human-readable string.

    Accepts both :class:`~erml.schemas.FacePrediction` objects (the default
    return type from ``EmotionDetector.analyze()``) and plain dictionaries
    for backwards compatibility.

    Args:
        results: List of FacePrediction objects or face dicts.

    Returns:
        Formatted multi-line string, or a message if no faces were found.

    Example::

        from erml import EmotionDetector, format_results

        detector = EmotionDetector()
        results = detector.analyze("photo.jpg")

        # Pretty-print
        print(format_results(results))

        # Or convert to plain dicts first
        print(format_results([r.model_dump() for r in results]))
    """
    if not results:
        return "No faces detected."

    lines = []
    for i, face in enumerate(results, 1):
        # Support both FacePrediction objects and plain dicts.
        if isinstance(face, FacePrediction):
            bbox = face.bbox
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            emotion = face.emotion
            confidence = face.confidence
            all_scores = face.all
        else:
            bbox = face["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            emotion = face["emotion"]
            confidence = face["confidence"]
            all_scores = face["all"]

        lines.append(f"Face {i}  [x={x} y={y} w={w} h={h}]")
        lines.append(f"  Emotion    : {emotion.capitalize()}")
        lines.append(f"  Confidence : {confidence:.1%}")
        lines.append("  All scores :")
        for label, score in sorted(all_scores.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(score * 20)
            lines.append(f"    {label:<10} {score:5.1%}  {bar}")
        lines.append("")
    return "\n".join(lines)
