"""Quick inference test script — runs analyze() on a real image and prints
human-readable results. Usage:

    python examples/test_inference.py path/to/photo.jpg
"""

import sys
import os


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/test_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    from erml import EmotionDetector, format_results

    detector = EmotionDetector()
    results = detector.analyze(image_path)
    print(format_results(results))


if __name__ == "__main__":
    main()
