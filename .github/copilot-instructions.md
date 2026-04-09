# Copilot Instructions for ERML

## Project Overview
ERML (Emotion Recognition ML) is a pip-installable Python SDK for facial emotion
recognition. It is NOT an app — it is a library that developers drop into their
own projects. The camera, UI, and display logic is always the user's responsibility.
The SDK accepts a frame (numpy array, file path, or PIL image) and returns structured
emotion data.

Target: pip install erml

## Stack
- Python 3.8+
- TensorFlow / Keras — CNN model
- OpenCV — face detection via Haar cascade
- NumPy — array operations and preprocessing
- pytest — all testing
- pyproject.toml — modern packaging (no setup.py)
- Black — code formatting
- Flake8 — linting

## Project Structure
erml/
├── erml/
│   ├── __init__.py          # exposes only EmotionDetector
│   ├── detector.py          # main public API class — EmotionDetector
│   ├── model.py             # CNN architecture definition only
│   ├── train.py             # training script, not part of public API
│   ├── preprocess.py        # image normalization and face preprocessing
│   └── assets/
│       ├── haarcascade_frontalface_default.xml
│       └── erml_v1.h5       # trained model (gitignored, downloaded on first use)
├── tests/
│   ├── test_detector.py
│   ├── test_preprocess.py
│   └── sample_faces/        # static test images from FER-2013 test set
├── examples/
│   └── webcam_demo.py       # example only, not part of package
├── models/                  # local training output, gitignored
├── dataset/                 # FER-2013 dataset, gitignored
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
├── .github/
│   ├── copilot-instructions.md
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── .gitignore
├── CHANGELOG.md
├── LICENSE                  # MIT
├── README.md
└── pyproject.toml

## Public API Contract
The ONLY public interface is the EmotionDetector class in detector.py.
__init__.py must only expose: from erml import EmotionDetector

### EmotionDetector usage:
detector = EmotionDetector()
result = detector.analyze(frame)

### analyze() input:
- numpy array (BGR, as returned by OpenCV)
- file path string to an image
- PIL.Image object

### analyze() output — list of dicts, one per detected face:
[
  {
    "emotion": "happy",              # top predicted emotion string
    "confidence": 0.87,              # float 0-1
    "all": {                         # scores for all 7 emotions
      "angry": 0.01,
      "disgust": 0.00,
      "fear": 0.02,
      "happy": 0.87,
      "sad": 0.03,
      "surprise": 0.05,
      "neutral": 0.02
    },
    "bbox": {                        # face bounding box in pixels
      "x": 120,
      "y": 80,
      "w": 64,
      "h": 64
    }
  }
]
Returns empty list [] if no face detected. Never raises on empty input.

## Emotion Labels (fixed order, do not change)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

## CNN Architecture (model.py)
- Input: 48x48x1 grayscale
- Conv2D(32, 3x3, relu) → BatchNormalization → MaxPooling2D(2,2)
- Conv2D(64, 3x3, relu) → BatchNormalization → MaxPooling2D(2,2)
- Conv2D(128, 3x3, relu) → BatchNormalization → MaxPooling2D(2,2)
- Flatten
- Dense(256, relu) → Dropout(0.5)
- Dense(128, relu) → Dropout(0.3)
- Dense(7, softmax)
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Metrics: accuracy

## Training (train.py)
- Load images from dataset/train/ organized by emotion subfolder
- Grayscale, resize to 48x48, normalize to 0-1
- Train/val split: 80/20
- Data augmentation: horizontal flip, zoom 0.1, width/height shift 0.1
- Epochs: 20, Batch size: 32
- Callbacks: ModelCheckpoint (save best val_accuracy only), EarlyStopping (patience=5)
- Save final model to erml/assets/erml_v1.h5
- Print final train accuracy, val accuracy after training

## Preprocessing Rules (preprocess.py)
- Input is always a face ROI (numpy array, already cropped)
- Convert to grayscale if not already
- Resize to 48x48
- Normalize: divide by 255.0
- Reshape to (1, 48, 48, 1)
- Return numpy array ready for model.predict()

## Coding Rules — follow strictly
1. No hardcoded paths anywhere. Always use os.path.join(__file__, ...) for asset paths.
2. Type hints on every function signature.
3. Google-style docstrings on every function and class.
4. Black formatting — max line length 88.
5. Every function in erml/ must have a corresponding test in tests/.
6. No print statements in library code — use Python logging module.
7. No camera or display code inside erml/ — that belongs in examples/ only.
8. predict() calls must use verbose=0 to suppress TensorFlow output.
9. All exceptions must be meaningful — no bare except clauses.
10. model.py only defines architecture — no training logic there.

## What NOT to do
- Do not add Flask, FastAPI, or any server code — this is a library not an app
- Do not add webcam/VideoCapture code inside the package
- Do not save images inside the package
- Do not use setup.py — pyproject.toml only
- Do not import train.py from detector.py — training is separate from inference

## Testing Rules
- Use pytest only
- Tests must run fully headless — no camera, no display windows
- Use static images from tests/sample_faces/ for all tests
- Test analyze() with: numpy array input, file path input, PIL input, no-face image
- test_preprocess.py must test input/output shapes and value ranges (0-1)
- All tests must pass with: pytest tests/

## Git / versioning
- main branch is always stable and installable
- Feature branches: feature/description
- Version format: MAJOR.MINOR.PATCH
- v0.1.0 — current: face only, FER-2013, basic CNN
- v0.2.0 — planned: model improvements, confidence calibration
- v1.0.0 — planned: stable public API, full docs
- v1.1.0 — planned: multimodal (voice signal added)
