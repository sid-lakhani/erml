# Training

To retrain the ERML model from scratch, you need the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and the training dependencies.

## Setup

```bash
uv pip install -r requirements-train.txt
```

## Dataset Structure

Download FER-2013 and place it under `dataset/` with one subfolder per emotion:

```
dataset/
└── train/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Run Training

```bash
python training/train.py
```

This will:

1. Build the CNN model defined in `erml/model.py`
2. Load and augment data using Keras `ImageDataGenerator`
3. Train for up to 20 epochs with early stopping (patience=5)
4. Save the best checkpoint to `erml/assets/erml_v1.h5`

## Expected Results

| Version | Val Accuracy | Notes |
|---------|-------------|-------|
| v0.1.0 | ~52% | Basic 3-block CNN, FER-2013 |

FER-2013 is a challenging dataset with significant label noise. 52% is consistent with the published literature for comparable architectures.
