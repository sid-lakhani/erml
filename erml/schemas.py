"""Pydantic output schemas for ERML.

These models define the structured output returned by EmotionDetector.analyze().
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel coordinates of a detected face bounding box."""

    x: int = Field(..., description="Left edge of the bounding box in pixels.")
    y: int = Field(..., description="Top edge of the bounding box in pixels.")
    w: int = Field(..., description="Width of the bounding box in pixels.")
    h: int = Field(..., description="Height of the bounding box in pixels.")


class FacePrediction(BaseModel):
    """Structured emotion prediction result for a single detected face.

    Example::

        results = detector.analyze("photo.jpg")

        # Object access (IDE autocomplete supported)
        print(results[0].emotion)
        print(results[0].confidence)
        print(results[0].bbox.x)

        # Export to plain dictionary (e.g. for JSON logging)
        raw_dict = results[0].model_dump()
        raw_dicts = [r.model_dump() for r in results]
    """

    emotion: str = Field(..., description="Top predicted emotion label.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the top emotion."
    )
    all: dict[str, float] = Field(
        ..., description="Scores for all 7 emotion labels, summing to 1.0."
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the detected face.")
