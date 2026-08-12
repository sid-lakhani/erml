"""Webcam demo for ERML — real-time facial emotion recognition.

Usage:
    python examples/webcam_demo.py
    python examples/webcam_demo.py --camera 1   # use a different camera index

Press Q to quit.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from erml import EmotionDetector


# Colour per emotion (BGR)
EMOTION_COLORS = {
    "happy": (0, 220, 0),
    "sad": (200, 80, 0),
    "angry": (0, 0, 220),
    "fear": (180, 0, 180),
    "surprise": (0, 200, 200),
    "disgust": (0, 140, 80),
    "neutral": (180, 180, 180),
}


def draw_results(frame: np.ndarray, results: list) -> None:
    """Draw bounding boxes and emotion labels onto a frame in-place.

    Args:
        frame: BGR numpy array (modified in-place).
        results: List of dicts returned by EmotionDetector.analyze().
    """
    for face in results:
        bbox = face["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        emotion = face["emotion"]
        confidence = face["confidence"]
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion}  {confidence:.0%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame, (x, y - lh - baseline - 6), (x + lw + 4, y), color, -1
        )
        cv2.putText(
            frame,
            label,
            (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def _is_wayland() -> bool:
    """Return True if the current session is running under Wayland.

    Returns:
        True if WAYLAND_DISPLAY is set, False otherwise.
    """
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def _try_imshow(frame: np.ndarray) -> bool:
    """Attempt cv2.imshow and return False if it raises or produces no window.

    Args:
        frame: BGR numpy array to display.

    Returns:
        True if imshow succeeded, False otherwise.
    """
    try:
        cv2.imshow("ERML — Emotion Detection", frame)
        return cv2.waitKey(1) != -2  # -2 signals no window system
    except cv2.error:
        return False


def _run_matplotlib(cap: cv2.VideoCapture, detector: EmotionDetector) -> None:
    """Fallback display loop using matplotlib (works on Wayland/headless).

    Args:
        cap: Open VideoCapture instance.
        detector: Initialised EmotionDetector.
    """
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("ERML — Emotion Detection")
    img_display = None

    print("Using matplotlib display — close the window to quit.")

    while plt.fignum_exists(fig.number):
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from camera.")
            break

        results = detector.analyze(frame)
        draw_results(frame, results)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if img_display is None:
            img_display = ax.imshow(rgb)
            ax.axis("off")
        else:
            img_display.set_data(rgb)

        fig.canvas.draw()
        fig.canvas.flush_events()


def main() -> None:
    parser = argparse.ArgumentParser(description="ERML webcam demo")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    args = parser.parse_args()

    detector = EmotionDetector()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Probe whether cv2.imshow works in this environment.
    ret, probe = cap.read()
    if not ret:
        print("Error: could not read from camera.")
        cap.release()
        sys.exit(1)

    use_imshow = not _is_wayland() and _try_imshow(probe)

    if not use_imshow:
        cv2.destroyAllWindows()
        _run_matplotlib(cap, detector)
        cap.release()
        return

    print("ERML webcam demo — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from camera.")
            break

        results = detector.analyze(frame)
        draw_results(frame, results)
        cv2.imshow("ERML — Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
