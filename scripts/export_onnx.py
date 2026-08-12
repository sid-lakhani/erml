"""Export a trained Keras .h5 model to ONNX format for deployment.

Run from the repo root after training:

    python scripts/export_onnx.py

Or with explicit paths:

    python scripts/export_onnx.py \
        --input  erml/assets/erml_v1.h5 \
        --output erml/assets/erml_v1.onnx \
        --opset  13

After exporting, update the sha256 entries in erml/download.py:

    sha256sum erml/assets/erml_v1.onnx
    sha256sum erml/assets/face_detection_yunet_2023mar.onnx

Requires: pip install erml[train] tf2onnx
"""

import argparse
import logging
import os
import sys

# Suppress TensorFlow C++ / CUDA / absl log spam before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Required for tf2onnx compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid Grappler crashes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_H5 = os.path.join(_REPO_ROOT, "erml", "assets", "erml_v1.h5")
_DEFAULT_ONNX = os.path.join(_REPO_ROOT, "erml", "assets", "erml_v1.onnx")


def export(h5_path: str, onnx_path: str, opset: int = 13) -> None:
    """Convert a Keras .h5 model to ONNX.

    Args:
        h5_path: Path to the trained Keras weights file (.h5).
        onnx_path: Destination path for the ONNX model.
        opset: ONNX opset version to target (default: 13).

    Raises:
        FileNotFoundError: If h5_path does not exist.
        ImportError: If tensorflow or tf2onnx are not installed.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"Keras model not found at: {h5_path}\n"
            "Train the model first with: python training/train.py"
        )

    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise ImportError(
            "Export requires tensorflow and tf2onnx. Install them with:\n"
            "    uv pip install -r requirements-train.txt tf2onnx\n"
            "or:\n"
            "    pip install erml[train] tf2onnx"
        ) from exc

    from training.model import build_model

    logger.info("Loading Keras model from %s", h5_path)
    model = build_model()
    model.load_weights(h5_path)

    logger.info("Converting to ONNX (opset=%d)...", opset)
    input_signature = [
        tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32, name="input")
    ]
    _, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=onnx_path,
    )

    size_mb = os.path.getsize(onnx_path) / 1_048_576
    logger.info("Exported ONNX model to %s (%.2f MB)", onnx_path, size_mb)
    logger.info(
        "Next step — update SHA256 in erml/download.py:\n" "    sha256sum %s",
        onnx_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an ERML Keras model to ONNX.")
    parser.add_argument(
        "--input",
        default=_DEFAULT_H5,
        help=f"Path to the .h5 weights file (default: {_DEFAULT_H5})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_ONNX,
        help=f"Destination ONNX path (default: {_DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(h5_path=args.input, onnx_path=args.output, opset=args.opset)
