# ERML — Emotion Recognition ML

A pip-installable Python SDK for facial emotion recognition. Drop it into your own project, pass it a frame, get back structured emotion data — no camera or display logic included.

```python
from erml import EmotionDetector, format_results

detector = EmotionDetector()
results = detector.analyze("photo.jpg")

# Object access — full IDE autocomplete
print(results[0].emotion, results[0].confidence)

# Pretty-print
print(format_results(results))
```

```
Face 1  [x=85 y=67 w=259 h=259]
  Emotion    : Neutral
  Confidence : 42.9%
  All scores :
    neutral    42.9%  ████████
    sad        18.2%  ███
    fear       15.4%  ███
    angry      10.2%  ██
    happy       7.5%  █
    surprise    5.2%  █
    disgust     0.1%
```

## Features

- **Drop-in SDK** — no camera or display logic; pass any image, get structured data back
- **Typed output** — results are Pydantic models with full IDE autocomplete
- **Auto-downloading** — weights download and cache automatically on first run
- **Flexible input** — accepts file paths, OpenCV numpy arrays, or PIL Images
- **7 emotions** — `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

## Quick Install

```bash
pip install erml
```

See [Installation](getting-started/installation.md) for full setup instructions.
