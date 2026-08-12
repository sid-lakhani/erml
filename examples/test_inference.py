"""Test script to evaluate ERML accuracy on static images.

Usage:
    python examples/test_inference.py <path_to_image>

Example:
    python examples/test_inference.py tests/sample_faces/happy_face.jpg
"""

import sys
import os
import cv2

from erml import EmotionDetector, format_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/test_inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Load detector (will auto-download ONNX weights if missing)
    detector = EmotionDetector()
    
    # Load image via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    print(f"\nAnalyzing: {image_path}")
    results = detector.analyze(img)

    if not results:
        print("No faces detected.")
        sys.exit(0)

    # Print pretty formatted results
    print("\n--- Console Output ---")
    print(format_results(results))

    # Draw bounding boxes and labels on the image for visual verification
    out_img = img.copy()
    for face in results:
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h
        
        # Draw bounding box
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{face.emotion.capitalize()} ({face.confidence*100:.1f}%)"
        
        # Draw text background for readability
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            out_img,
            (x, y - text_height - 10),
            (x + text_width, y),
            (0, 255, 0),
            -1,
        )
        
        # Draw text
        cv2.putText(
            out_img,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    out_path = os.path.join(out_dir, f"result_{base_name}")
    cv2.imwrite(out_path, out_img)
    
    print(f"\n--- Visual Output ---")
    print(f"Saved visualization to: {out_path}")
    print("Open this file in your IDE to see the exact bounding boxes and labels!")

if __name__ == "__main__":
    main()
