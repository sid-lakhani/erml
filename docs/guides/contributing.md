# Contributing to ERML

Thanks for taking the time to contribute! This guide covers environment setup, running tests, formatting code, and the pull request process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Training the Model](#training-the-model)
- [Pull Request Process](#pull-request-process)

---

## Development Setup

ERML uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environments. Make sure you have it installed:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Then clone the repo and set up your environment:

```bash
git clone https://github.com/sid-lakhani/erml
cd erml

# Create a virtualenv pinned to Python 3.12 (or any 3.8–3.12)
uv venv --python 3.12
source .venv/bin/activate        # bash / zsh
# source .venv/bin/activate.fish  # fish shell

# Install runtime + dev dependencies and the package in editable mode
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
```

> **Note**: TensorFlow is **not** a runtime dependency. It is only required for training.
> If you need to retrain the model, also run:
> ```bash
> uv pip install -r requirements-train.txt
> ```

---

## Running Tests

All 42 tests are fully headless — no camera, no display, no trained model required:

```bash
pytest tests/ -v
```

The CI pipeline runs the same suite on Ubuntu and Windows across Python 3.10, 3.11, and 3.12. Please ensure your changes pass locally before opening a PR.

---

## Code Style

ERML enforces Black formatting and Flake8 linting. Both are included in `requirements-dev.txt`.

**Format your code before committing:**

```bash
black erml/ tests/ training/
```

**Check for lint issues:**

```bash
flake8 erml/ tests/ training/
```

The line length is set to **88** (Black default). The `.flake8` config suppresses `E203`, `W503` (Black-incompatible rules), and `E402` (intentional for TensorFlow log suppression).

---

## Training the Model

Training requires the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and TensorFlow:

```bash
uv pip install -r requirements-train.txt
```

Place the dataset at `dataset/train/` organised by emotion subfolder, then:

```bash
python training/train.py
```

This trains for up to 20 epochs with early stopping and saves best weights to `erml/assets/erml_v1.h5`.

---

## Pull Request Process

1. Fork the repository and create a new branch from `main`.
2. Make your changes and ensure all tests pass.
3. Ensure `black` and `flake8` report no issues.
4. Open a PR with a clear description of what changed and why.
5. Maintainers will review and merge.

For large changes, please open an issue first to discuss the approach before writing code.

---

## Reporting Bugs

Use the [Bug Report](https://github.com/sid-lakhani/erml/issues/new?template=bug_report.md) template on GitHub. The more detail you provide (OS, Python version, exact error traceback), the faster we can help.
