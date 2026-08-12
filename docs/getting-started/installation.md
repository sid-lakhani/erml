# Installation

## Stable (PyPI)

```bash
pip install erml
```

Or with `uv`:

```bash
uv pip install erml
```

> **Requirements**: Python 3.8–3.12. TensorFlow does not yet support Python 3.13+.

## From Source

```bash
git clone https://github.com/sid-lakhani/erml
cd erml
uv venv --python 3.12
source .venv/bin/activate   # bash/zsh
# source .venv/bin/activate.fish  # fish
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
```

## For Training

If you want to retrain the model, install the optional training dependencies:

```bash
uv pip install -r requirements-train.txt
```

This adds TensorFlow and scipy. These are **not** required for inference.

## Model Weights

On the first call to `EmotionDetector()`, the pre-trained weights are downloaded automatically from GitHub Releases and cached at `~/.cache/erml/erml_v1.h5`. No manual setup is needed.
