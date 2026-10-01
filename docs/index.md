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

## Real-world Applications

ERML is designed as a foundational **computer-vision perception SDK** for developers to plug into broader AI systems and tools.

- 🤖 **AI Agents**: Multimodal agents understanding facial signals
- 🎮 **Games**: NPC/player emotion → adaptive gameplay
- 🧑‍💻 **Video Calls**: Engagement, attention, and reaction signals
- 🧪 **HCI Research**: Studying human-computer interaction
- 🏥 **Research**: Controlled affective-computing experiments
- 🎥 **Content Creation**: Automatic reaction/emotion metadata
- 🛍️ **UX Research**: Aggregate reactions during usability testing
- 🧠 **Robotics**: Robot perception of human affect
- 🖥️ **Accessibility**: Alternative interaction signals

*Note: Facial expression ≠ actual emotional state. ERML detects facial expressions to provide a structured signal for your downstream application.*

## Quick Install

```bash
pip install erml
```

See [Installation](getting-started/installation.md) for full setup instructions.

## Roadmap

ERML is evolving from an emotion classifier into a comprehensive **human visual perception SDK**. 

- **Phase 1 (Current):** Stable facial expression recognition SDK with local ONNX inference.
- **Phase 2 (Planned):** Expand perception to include face landmarks, head pose, and gaze tracking.
- **Phase 3 (Planned):** Expose structured multi-modal signals natively designed for AI agent consumption.
