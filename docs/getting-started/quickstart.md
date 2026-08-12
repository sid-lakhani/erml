# Quick Start

## Basic Usage

```python
from erml import EmotionDetector

detector = EmotionDetector()
results = detector.analyze("photo.jpg")
```

`analyze()` returns a list of [`FacePrediction`](../api/schemas.md) objects — one per detected face.

## Input Types

ERML accepts three input formats:

=== "File Path"

    ```python
    results = detector.analyze("photo.jpg")
    ```

=== "OpenCV Array"

    ```python
    import cv2
    frame = cv2.imread("photo.jpg")
    results = detector.analyze(frame)  # BGR array
    ```

=== "PIL Image"

    ```python
    from PIL import Image
    img = Image.open("photo.jpg")
    results = detector.analyze(img)
    ```

## Working with Results

Each result is a `FacePrediction` Pydantic object with full IDE autocomplete:

```python
for face in results:
    print(face.emotion)       # "happy"
    print(face.confidence)    # 0.87
    print(face.bbox.x)        # 120  (bounding box left edge)
    print(face.all)           # {"angry": 0.01, "happy": 0.87, ...}
```

### Export to Dictionary

For JSON logging or third-party integrations:

```python
raw = results[0].model_dump()
# {"emotion": "happy", "confidence": 0.87, "all": {...}, "bbox": {"x": 120, ...}}

raw_list = [r.model_dump() for r in results]
```

### Pretty-Print

```python
from erml import format_results

print(format_results(results))
```

```
Face 1  [x=85 y=67 w=259 h=259]
  Emotion    : Happy
  Confidence : 87.0%
  All scores :
    happy      87.0%  █████████████████
    surprise    5.2%  █
    ...
```

## No Face Detected

`analyze()` always returns an empty list if no face is found — it never raises on empty input:

```python
results = detector.analyze("landscape.jpg")
assert results == []
```
