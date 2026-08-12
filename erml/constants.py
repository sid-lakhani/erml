"""Shared constants for the ERML package.

Kept in a standalone module so that neither the runtime package nor the
training scripts need to import TensorFlow just to access the label list.
"""

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
"""Ordered list of emotion class names used by the ERML model.

The index position of each label corresponds directly to the output neuron
index of both the Keras training model and the ONNX inference session.
"""
