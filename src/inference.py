"""
SAR Ship Detection Inference Pipeline

This module performs inference on SAR imagery to detect ships using the trained
YOLOv8-OBB model. It handles:

1. Loading and preprocessing SAR images (Lee filter)
2. Running OBB detection with the trained model
3. Drawing rotated bounding boxes on the output
4. Saving results with confidence scores and ship statistics

OUTPUT VISUALIZATION:
---------------------
The output image shows:
- Rotated rectangles (OBB) tightly fitting around detected ships
- Confidence scores for each detection
- Ship heading direction (inferred from box orientation)
- Detection statistics overlay

WHY OBB VISUALIZATION MATTERS:
------------------------------
Standard rectangular boxes would show:
┌─────────────────────┐
│   ████████          │  <- 60% of box is water (false area)
│        █████████    │
└─────────────────────┘

OBB visualization shows:
   ╱████████████████╲     <- Tight fit shows actual ship footprint
  ╱████████████████╲      <- Orientation visible at a glance
   ╲████████████████╱     <- No wasted pixels
    ╲████████████████╱

This is critical for:
- Maritime traffic monitoring
- Port congestion analysis
- Search and rescue operations
- Environmental monitoring (oil spills, illegal fishing)
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Import local modules
from preprocessing import SARPreprocessor, apply_lee_filter


class SARShipDetector:
    """
    Production-grade SAR ship detection inference engine.

    This class encapsulates the complete inference pipeline:
    1. Image preprocessing (speckle filtering)
    2. Model inference (YOLOv8-OBB)
    3. Result visualization (rotated bounding boxes)
    4. Statistics computation
    """

    # Color palette for visualization (BGR format)
    COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        preprocess: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the SAR ship detector.

        Args:
            model_path: Path to trained YOLOv8-OBB weights (.pt file)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            preprocess: Apply Lee filter before inference
            device: Inference device ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.preprocess = preprocess
        self.device = device

        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        print(f"[INFO] Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # Initialize preprocessor
        self.preprocessor = SARPreprocessor(
            lee_window_size=7,
            use_enhanced_lee=False,
            enhance_contrast=True
        )

        print(f"[INFO] Detector initialized")
        print(f"[INFO] Confidence threshold: {confidence_threshold}")
        print(f"[INFO] IoU threshold: {iou_threshold}")
        print(f"[INFO] Preprocessing: {preprocess}")

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        visualize: bool = True
    ) -> dict:
        """
        Detect ships in a SAR image.

        Args:
            image: Input image (path or numpy array)
            visualize: Whether to create visualization

        Returns:
            Dictionary containing:
            - 'detections': List of detection dictionaries
            - 'count': Number of ships detected
            - 'visualization': Annotated image (if visualize=True)
            - 'processed_image': Preprocessed image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image.copy()

        original = img.copy()

        # Preprocess with Lee filter
        if self.preprocess:
            img = self.preprocessor.process(img)

        # Run inference
        results = self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        # Parse detections
        detections = self._parse_obb_results(results)

        # Create visualization
        visualization = None
        if visualize:
            visualization = self._visualize_detections(original, img, detections)

        return {
            'detections': detections,
            'count': len(detections),
            'visualization': visualization,
            'processed_image': img,
            'original_image': original
        }

    def _parse_obb_results(self, results) -> List[dict]:
        """
        Parse YOLOv8-OBB results into structured detection dictionaries.

        OBB results contain:
        - xyxyxyxy: Four corner points (x1,y1,x2,y2,x3,y3,x4,y4)
        - conf: Confidence score
        - cls: Class ID

        Args:
            results: YOLO prediction results

        Returns:
            List of detection dictionaries
        """
        detections = []

        if results.obb is None:
            return detections

        obb = results.obb

        for i in range(len(obb)):
            # Get the 4 corners of the OBB (shape: [4, 2])
            corners = obb.xyxyxyxy[i].cpu().numpy()

            # Calculate center, dimensions, and angle
            center = corners.mean(axis=0)

            # Width is distance between adjacent corners
            width = np.linalg.norm(corners[1] - corners[0])
            height = np.linalg.norm(corners[2] - corners[1])

            # Angle from the first edge
            angle = np.arctan2(
                corners[1][1] - corners[0][1],
                corners[1][0] - corners[0][0]
            ) * 180 / np.pi

            # Get confidence and class
            conf = float(obb.conf[i].cpu().numpy())
            cls = int(obb.cls[i].cpu().numpy())

            detection = {
                'corners': corners.tolist(),
                'center': center.tolist(),
                'width': float(width),
                'height': float(height),
                'angle': float(angle),
                'confidence': conf,
                'class_id': cls,
                'class_name': 'ship'
            }

            detections.append(detection)

        return detections

    def _visualize_detections(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        detections: List[dict]
    ) -> np.ndarray:
        """
        Create visualization with OBB annotations.

        The visualization includes:
        - Side-by-side original and processed images
        - Rotated bounding boxes around ships
        - Confidence scores
        - Detection count overlay

        Args:
            original: Original SAR image
            processed: Lee-filtered image
            detections: List of detection dictionaries

        Returns:
            Annotated visualization image
        """
        # Draw on the processed image
        canvas = processed.copy()

        # Ensure 3-channel for visualization
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        # Draw each detection
        for i, det in enumerate(detections):
            color = self.COLORS[i % len(self.COLORS)]
            corners = np.array(det['corners'], dtype=np.int32)

            # Draw the rotated rectangle
            cv2.polylines(canvas, [corners], isClosed=True, color=color, thickness=2)

            # Draw corner points
            for corner in corners:
                cv2.circle(canvas, tuple(corner), 3, color, -1)

            # Draw center point
            center = tuple(np.array(det['center'], dtype=np.int32))
            cv2.circle(canvas, center, 5, (0, 0, 255), -1)

            # Draw confidence label
            conf_text = f"Ship {det['confidence']:.2f}"
            label_pos = (int(corners[0][0]), int(corners[0][1]) - 10)

            # Add background for text
            (text_w, text_h), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                canvas,
                (label_pos[0], label_pos[1] - text_h - 5),
                (label_pos[0] + text_w, label_pos[1] + 5),
                color, -1
            )
            cv2.putText(
                canvas, conf_text, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Add statistics overlay
        h, w = canvas.shape[:2]
        stats_text = f"Ships Detected: {len(detections)}"
        cv2.rectangle(canvas, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(
            canvas, stats_text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Create side-by-side comparison
        # Add labels to original
        original_labeled = original.copy()
        cv2.putText(
            original_labeled, "Original SAR", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            canvas, "Lee Filtered + Detections", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # Stack horizontally
        visualization = np.hstack([original_labeled, canvas])

        return visualization

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> List[dict]:
        """
        Run detection on multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
            show_progress: Print progress updates

        Returns:
            List of detection result dictionaries
        """
        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        total = len(image_paths)

        for i, img_path in enumerate(image_paths):
            if show_progress:
                print(f"[INFO] Processing {i + 1}/{total}: {Path(img_path).name}")

            result = self.detect(img_path, visualize=True)
            result['source_path'] = str(img_path)

            # Save visualization
            if output_dir and result['visualization'] is not None:
                out_path = output_dir / f"detected_{Path(img_path).name}"
                cv2.imwrite(str(out_path), result['visualization'])
                result['output_path'] = str(out_path)

            results.append(result)

        # Print summary
        total_ships = sum(r['count'] for r in results)
        avg_ships = total_ships / len(results) if results else 0

        print(f"\n[SUMMARY] Processed {len(results)} images")
        print(f"[SUMMARY] Total ships detected: {total_ships}")
        print(f"[SUMMARY] Average ships per image: {avg_ships:.1f}")

        return results


def run_demo_inference(
    model_path: str,
    data_yaml: str,
    output_dir: str = "./output",
    num_samples: int = 5
):
    """
    Run demo inference on random validation images.

    This function:
    1. Loads a trained model
    2. Selects random images from the validation set
    3. Runs detection and saves visualizations

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml configuration
        output_dir: Directory for output visualizations
        num_samples: Number of sample images to process
    """
    import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    data_root = Path(config['path'])
    val_dir = data_root / config['val']

    # Get validation images
    val_images = list(val_dir.glob('*'))
    val_images = [p for p in val_images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}]

    if not val_images:
        print(f"[ERROR] No validation images found in: {val_dir}")
        return

    # Select random samples
    random.seed(42)
    samples = random.sample(val_images, min(num_samples, len(val_images)))

    print(f"[INFO] Running inference on {len(samples)} sample images")
    print(f"[INFO] Output directory: {output_dir}")

    # Initialize detector
    detector = SARShipDetector(
        model_path=model_path,
        confidence_threshold=0.25,
        preprocess=True
    )

    # Run batch detection
    results = detector.detect_batch(
        samples,
        output_dir=output_dir,
        show_progress=True
    )

    print(f"\n[SUCCESS] Detection complete!")
    print(f"[INFO] Results saved to: {output_dir}")

    return results


def main():
    """Main entry point for inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAR Ship Detection Inference"
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained YOLOv8-OBB model (.pt file)'
    )
    parser.add_argument(
        '--source', type=str, default=None,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to data.yaml (for demo mode with val images)'
    )
    parser.add_argument(
        '--output', type=str, default='./output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou', type=float, default=0.45,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Inference device (cuda, mps, cpu)'
    )
    parser.add_argument(
        '--no-preprocess', action='store_true',
        help='Skip Lee filter preprocessing'
    )
    parser.add_argument(
        '--samples', type=int, default=5,
        help='Number of random samples for demo mode'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demo mode: use random validation images
    if args.data and not args.source:
        run_demo_inference(
            model_path=args.model,
            data_yaml=args.data,
            output_dir=args.output,
            num_samples=args.samples
        )
        return

    # Standard inference mode
    if not args.source:
        print("[ERROR] Either --source or --data must be provided")
        parser.print_help()
        return

    # Initialize detector
    detector = SARShipDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        preprocess=not args.no_preprocess,
        device=args.device
    )

    source = Path(args.source)

    if source.is_file():
        # Single image
        print(f"[INFO] Processing: {source}")
        result = detector.detect(source, visualize=True)

        # Save result
        out_path = output_dir / f"detected_{source.name}"
        if result['visualization'] is not None:
            cv2.imwrite(str(out_path), result['visualization'])
            print(f"[INFO] Saved: {out_path}")

        print(f"[INFO] Ships detected: {result['count']}")

        for i, det in enumerate(result['detections']):
            print(f"  Ship {i + 1}: conf={det['confidence']:.2f}, "
                  f"angle={det['angle']:.1f}°, "
                  f"size={det['width']:.0f}x{det['height']:.0f}")

    elif source.is_dir():
        # Directory of images
        images = list(source.glob('*'))
        images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}]

        if not images:
            print(f"[ERROR] No images found in: {source}")
            return

        print(f"[INFO] Processing {len(images)} images from: {source}")
        detector.detect_batch(images, output_dir=output_dir)

    else:
        print(f"[ERROR] Source not found: {source}")


if __name__ == "__main__":
    main()
