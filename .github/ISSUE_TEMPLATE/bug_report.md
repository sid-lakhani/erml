---
name: Bug Report
about: Report a reproducible bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ""
---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

```python
# Minimal code to reproduce the problem
from erml import EmotionDetector

detector = EmotionDetector()
results = detector.analyze("photo.jpg")
```

## Expected Behavior

What did you expect to happen?

## Actual Behavior

What actually happened? Include the full error traceback if applicable:

```
Traceback (most recent call last):
  ...
```

## Environment

- **OS**: [e.g. Ubuntu 22.04, macOS 14, Windows 11]
- **Python version**: [e.g. 3.12.1]
- **ERML version**: [e.g. 0.1.0 — run `python -c "import erml; print(erml.__version__)"`]
- **OpenCV version**: [e.g. 4.10.0 — run `python -c "import cv2; print(cv2.__version__)"`]
- **Installed via**: [pip / from source]

## Additional Context

Any other context, screenshots, or sample images (if relevant and shareable).
