"""Display utilities for ERML results."""

from typing import List, Dict


def format_results(results: List[Dict]) -> str:
    """Format analyze() output as a human-readable string.

    Args:
        results: List of face dicts returned by EmotionDetector.analyze().

    Returns:
        Formatted multi-line string, or a message if no faces were found.
    """
    if not results:
        return "No faces detected."

    lines = []
    for i, face in enumerate(results, 1):
        bbox = face["bbox"]
        lines.append(f"Face {i}  [x={bbox['x']} y={bbox['y']} w={bbox['w']} h={bbox['h']}]")
        lines.append(f"  Emotion    : {face['emotion'].capitalize()}")
        lines.append(f"  Confidence : {face['confidence']:.1%}")
        lines.append("  All scores :")
        for label, score in sorted(face["all"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            lines.append(f"    {label:<10} {score:5.1%}  {bar}")
        lines.append("")
    return "\n".join(lines)
