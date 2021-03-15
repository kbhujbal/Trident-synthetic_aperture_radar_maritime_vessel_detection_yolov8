"""
Generate sample output images for README showcase.

This script creates:
1. Detection results with OBB visualization
2. Before/after speckle filter comparison
3. Grid of multiple detections
"""

import sys
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from preprocessing import SARPreprocessor, apply_lee_filter

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed")
    sys.exit(1)


def generate_detection_samples(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_samples: int = 6
):
    """Generate detection result images."""

    model = YOLO(model_path)
    preprocessor = SARPreprocessor(lee_window_size=7, enhance_contrast=True)

    # Get validation images
    val_dir = Path(data_dir) / "images" / "val"
    images = list(val_dir.glob("*.jpg"))

    if not images:
        print(f"[ERROR] No images found in {val_dir}")
        return

    # Select random samples with good variety
    random.seed(42)
    samples = random.sample(images, min(num_samples, len(images)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_images = []

    for i, img_path in enumerate(samples):
        print(f"[INFO] Processing {i+1}/{len(samples)}: {img_path.name}")

        # Load original image
        original = cv2.imread(str(img_path))
        if original is None:
            continue

        # Apply preprocessing
        filtered = preprocessor.process(original)

        # Run detection
        results = model.predict(
            source=filtered,
            conf=0.25,
            iou=0.45,
            verbose=False
        )[0]

        # Draw detections on filtered image
        annotated = filtered.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Colors for visualization
        colors = [
            (0, 255, 0),    # Green
            (255, 100, 0),  # Blue
            (0, 100, 255),  # Orange
            (255, 0, 255),  # Magenta
        ]

        ship_count = 0
        if results.obb is not None:
            for j, obb in enumerate(results.obb.xyxyxyxy):
                corners = obb.cpu().numpy().astype(np.int32)
                color = colors[j % len(colors)]

                # Draw rotated rectangle
                cv2.polylines(annotated, [corners], isClosed=True, color=color, thickness=2)

                # Draw confidence
                conf = float(results.obb.conf[j].cpu().numpy())
                label = f"{conf:.2f}"
                cv2.putText(annotated, label,
                           (int(corners[0][0]), int(corners[0][1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                ship_count += 1

        # Add title
        cv2.putText(annotated, f"Ships Detected: {ship_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save individual result
        out_path = output_dir / f"detection_{i+1}.jpg"
        cv2.imwrite(str(out_path), annotated)
        results_images.append(annotated)
        print(f"[INFO] Saved: {out_path}")

    # Create grid of results (2x3)
    if len(results_images) >= 6:
        # Resize all to same size
        target_size = (320, 320)
        resized = [cv2.resize(img, target_size) for img in results_images[:6]]

        # Create 2x3 grid
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:6])
        grid = np.vstack([row1, row2])

        grid_path = output_dir / "detection_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"[INFO] Saved grid: {grid_path}")

    return results_images


def generate_filter_comparison(
    data_dir: str,
    output_dir: str
):
    """Generate before/after speckle filter comparison."""

    preprocessor = SARPreprocessor(lee_window_size=7, enhance_contrast=True)

    # Get a sample image
    val_dir = Path(data_dir) / "images" / "val"
    images = list(val_dir.glob("*.jpg"))

    if not images:
        return

    # Pick an image with visible speckle
    random.seed(123)
    img_path = random.choice(images)

    original = cv2.imread(str(img_path))
    if original is None:
        return

    # Apply Lee filter only (no CLAHE for clear comparison)
    filtered = apply_lee_filter(original, window_size=7)

    # Convert to BGR if grayscale
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(filtered.shape) == 2:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    # Add labels
    h, w = original.shape[:2]

    original_labeled = original.copy()
    filtered_labeled = filtered.copy()

    # Add semi-transparent header
    cv2.rectangle(original_labeled, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.rectangle(filtered_labeled, (0, 0), (w, 35), (0, 0, 0), -1)

    cv2.putText(original_labeled, "Original SAR (Speckled)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(filtered_labeled, "Lee Filtered (Denoised)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Create side-by-side comparison
    comparison = np.hstack([original_labeled, filtered_labeled])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_path = output_dir / "speckle_filter_comparison.jpg"
    cv2.imwrite(str(comp_path), comparison)
    print(f"[INFO] Saved comparison: {comp_path}")

    return comparison


def generate_pipeline_diagram(output_dir: str):
    """Generate a simple pipeline visualization."""

    # Create a simple pipeline diagram
    width, height = 800, 200
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw boxes for each step
    box_width = 150
    box_height = 60
    y_center = height // 2

    steps = [
        ("SAR Image", (50, 50, 50)),
        ("Lee Filter", (0, 100, 200)),
        ("YOLOv8-OBB", (0, 150, 0)),
        ("Detection", (200, 100, 0))
    ]

    x_positions = [50, 230, 410, 590]

    for i, ((label, color), x) in enumerate(zip(steps, x_positions)):
        # Draw box
        cv2.rectangle(img,
                     (x, y_center - box_height//2),
                     (x + box_width, y_center + box_height//2),
                     color, 2)

        # Add label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (box_width - text_size[0]) // 2
        text_y = y_center + text_size[1] // 2
        cv2.putText(img, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw arrow to next box
        if i < len(steps) - 1:
            arrow_start = (x + box_width + 5, y_center)
            arrow_end = (x_positions[i+1] - 5, y_center)
            cv2.arrowedLine(img, arrow_start, arrow_end, (100, 100, 100), 2, tipLength=0.3)

    # Add title
    cv2.putText(img, "Trident Pipeline", (width//2 - 80, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_path = output_dir / "pipeline.jpg"
    cv2.imwrite(str(pipe_path), img)
    print(f"[INFO] Saved pipeline: {pipe_path}")


def main():
    project_root = Path(__file__).parent.parent

    model_path = project_root / "trident_sar" / "ship_detection" / "weights" / "best.pt"
    data_dir = project_root / "data" / "ssdd_yolo_obb"
    output_dir = project_root / "assets"

    print("=" * 50)
    print("  Generating Sample Images for README")
    print("=" * 50)

    # Check if model exists
    if not model_path.exists():
        print(f"[WARN] Model not found at {model_path}")
        print("[INFO] Using pretrained yolov8n-obb.pt")
        model_path = project_root / "yolov8n-obb.pt"

        if not model_path.exists():
            # Download pretrained
            from ultralytics import YOLO
            model = YOLO("yolov8n-obb.pt")
            model_path = project_root / "yolov8n-obb.pt"

    print(f"\n[1/3] Generating detection samples...")
    generate_detection_samples(str(model_path), str(data_dir), str(output_dir))

    print(f"\n[2/3] Generating filter comparison...")
    generate_filter_comparison(str(data_dir), str(output_dir))

    print(f"\n[3/3] Generating pipeline diagram...")
    generate_pipeline_diagram(str(output_dir))

    print("\n" + "=" * 50)
    print(f"  Done! Images saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
