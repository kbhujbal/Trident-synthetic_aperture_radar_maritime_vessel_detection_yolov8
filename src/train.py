"""
YOLOv8-OBB Training Pipeline for SAR Ship Detection

WHY YOLOv8-OBB (Oriented Bounding Box)?
=======================================

Standard object detection models use Axis-Aligned Bounding Boxes (AABB), which
work well when objects are roughly horizontal/vertical. But ships in SAR imagery:

1. Sail in arbitrary directions (0-360°)
2. Appear at various angles due to satellite viewing geometry
3. Are often elongated (length >> width)

PROBLEMS WITH HORIZONTAL BOXES:
-------------------------------
┌─────────────────────────┐
│  ████████               │    <- Lots of wasted background pixels
│       █████████         │    <- May overlap with adjacent ships
│           ████████      │    <- Poor IoU even with correct detection
└─────────────────────────┘

ADVANTAGES OF ORIENTED BOXES:
-----------------------------
    ╱████████████████╲
   ╱████████████████╲        <- Tight fit around ship hull
    ╲████████████████╱       <- Better IoU metrics
     ╲████████████████╱      <- No overlap with parallel vessels

YOLOv8-OBB predicts: (x_center, y_center, width, height, rotation_angle)
This 5-DOF representation perfectly captures ship orientation.

TRAINING CONSIDERATIONS:
------------------------
1. We preprocess images with Lee filter BEFORE training
2. Data augmentation must preserve OBB annotation validity
3. Learning rate scheduling is crucial for convergence
4. SAR images benefit from larger input sizes (640-1024)
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import cv2
import numpy as np

# Import ultralytics for YOLOv8-OBB
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Import our preprocessing module
from preprocessing import SARPreprocessor, apply_lee_filter
from data_manager import SSDDDataManager


class SARShipTrainer:
    """
    YOLOv8-OBB training pipeline for SAR ship detection.

    This trainer:
    1. Preprocesses SAR images with speckle filtering
    2. Configures YOLOv8-OBB model architecture
    3. Handles training with appropriate hyperparameters for SAR data
    4. Manages checkpoints and experiment logging
    """

    # Default hyperparameters optimized for SAR ship detection
    DEFAULT_HYPERPARAMS = {
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 20,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'hsv_h': 0.015,  # Minimal color augmentation (SAR is grayscale-like)
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'degrees': 180.0,  # Full rotation augmentation (ships can face any direction)
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,  # No shear (preserves ship aspect ratio)
        'perspective': 0.0,  # No perspective (SAR is top-down)
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
    }

    def __init__(
        self,
        data_yaml: str,
        model_variant: str = "yolov8n-obb",
        project_name: str = "trident_sar",
        experiment_name: str = "ship_detection",
        preprocess_images: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the SAR ship detection trainer.

        Args:
            data_yaml: Path to data.yaml configuration file
            model_variant: YOLOv8-OBB model variant:
                          - 'yolov8n-obb': Nano (fastest, least accurate)
                          - 'yolov8s-obb': Small (good balance)
                          - 'yolov8m-obb': Medium
                          - 'yolov8l-obb': Large
                          - 'yolov8x-obb': XLarge (most accurate, slowest)
            project_name: Name for experiment grouping
            experiment_name: Name for this specific run
            preprocess_images: Whether to apply Lee filter before training
            device: Training device ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.data_yaml = Path(data_yaml)
        self.model_variant = model_variant
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.preprocess_images = preprocess_images
        self.device = device

        # Validate data configuration
        self._validate_data_config()

        # Initialize model (will load pretrained weights)
        self.model = YOLO(f"{model_variant}.pt")

        # Preprocessor for SAR images
        self.preprocessor = SARPreprocessor(
            lee_window_size=7,
            use_enhanced_lee=False,
            enhance_contrast=True
        )

        print(f"[INFO] Initialized trainer with model: {model_variant}")
        print(f"[INFO] Data config: {self.data_yaml}")
        print(f"[INFO] Preprocessing enabled: {preprocess_images}")

    def _validate_data_config(self):
        """Validate the data.yaml configuration."""
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_yaml}")

        with open(self.data_yaml) as f:
            config = yaml.safe_load(f)

        required_keys = ['path', 'train', 'val', 'names']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in data.yaml: {key}")

        print(f"[INFO] Data config validated: {len(config.get('names', {}))} classes")

    def preprocess_dataset(self, force: bool = False):
        """
        Apply Lee speckle filter to all training and validation images.

        This creates a preprocessed copy of the dataset. The preprocessing
        is done ONCE before training, not on-the-fly, for efficiency.

        Args:
            force: If True, re-preprocess even if already done
        """
        if not self.preprocess_images:
            print("[INFO] Preprocessing disabled, skipping")
            return

        with open(self.data_yaml) as f:
            config = yaml.safe_load(f)

        data_root = Path(config['path'])

        for split in ['train', 'val']:
            split_dir = data_root / config[split]
            processed_marker = split_dir / '.preprocessed'

            if processed_marker.exists() and not force:
                print(f"[INFO] {split} already preprocessed, skipping")
                continue

            print(f"[INFO] Preprocessing {split} images with Lee filter...")

            image_files = list(split_dir.glob('*'))
            image_files = [p for p in image_files if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}]

            for i, img_path in enumerate(image_files):
                if (i + 1) % 50 == 0:
                    print(f"[INFO] Processed {i + 1}/{len(image_files)}")

                # Load, process, and overwrite
                img = cv2.imread(str(img_path))
                if img is not None:
                    filtered = self.preprocessor.process(img)
                    cv2.imwrite(str(img_path), filtered)

            # Mark as preprocessed
            processed_marker.touch()
            print(f"[INFO] Completed preprocessing {len(image_files)} {split} images")

    def train(
        self,
        hyperparams: Optional[Dict[str, Any]] = None,
        resume: bool = False,
        resume_path: Optional[str] = None
    ) -> str:
        """
        Train the YOLOv8-OBB model on the SAR ship dataset.

        Args:
            hyperparams: Custom hyperparameters (merged with defaults)
            resume: Whether to resume from last checkpoint
            resume_path: Specific checkpoint to resume from

        Returns:
            Path to the best trained model weights
        """
        # Preprocess dataset first
        self.preprocess_dataset()

        # Merge hyperparameters
        params = self.DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            params.update(hyperparams)

        print("\n" + "=" * 60)
        print("  TRIDENT SAR SHIP DETECTION - TRAINING")
        print("=" * 60)
        print(f"  Model:      {self.model_variant}")
        print(f"  Epochs:     {params['epochs']}")
        print(f"  Batch size: {params['batch']}")
        print(f"  Image size: {params['imgsz']}")
        print(f"  Device:     {self.device or 'auto'}")
        print("=" * 60 + "\n")

        # Configure training arguments
        train_args = {
            'data': str(self.data_yaml),
            'epochs': params['epochs'],
            'batch': params['batch'],
            'imgsz': params['imgsz'],
            'patience': params['patience'],
            'project': self.project_name,
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': params['optimizer'],
            'lr0': params['lr0'],
            'lrf': params['lrf'],
            'momentum': params['momentum'],
            'weight_decay': params['weight_decay'],
            'warmup_epochs': params['warmup_epochs'],
            'warmup_momentum': params['warmup_momentum'],
            'warmup_bias_lr': params['warmup_bias_lr'],
            'hsv_h': params['hsv_h'],
            'hsv_s': params['hsv_s'],
            'hsv_v': params['hsv_v'],
            'degrees': params['degrees'],
            'translate': params['translate'],
            'scale': params['scale'],
            'shear': params['shear'],
            'perspective': params['perspective'],
            'flipud': params['flipud'],
            'fliplr': params['fliplr'],
            'mosaic': params['mosaic'],
            'mixup': params['mixup'],
            'copy_paste': params['copy_paste'],
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'plots': True,
            'verbose': True,
        }

        # Add device if specified
        if self.device:
            train_args['device'] = self.device

        # Handle resume
        if resume:
            if resume_path:
                train_args['resume'] = resume_path
            else:
                # Find last checkpoint
                last_ckpt = Path(self.project_name) / self.experiment_name / 'weights' / 'last.pt'
                if last_ckpt.exists():
                    train_args['resume'] = str(last_ckpt)
                    print(f"[INFO] Resuming from: {last_ckpt}")

        # Start training
        results = self.model.train(**train_args)

        # Get best model path
        best_model = Path(self.project_name) / self.experiment_name / 'weights' / 'best.pt'

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best model: {best_model}")
        print(f"  Results:    {Path(self.project_name) / self.experiment_name}")
        print("=" * 60 + "\n")

        return str(best_model)

    def validate(self, model_path: Optional[str] = None) -> Dict:
        """
        Validate the trained model on the validation set.

        Args:
            model_path: Path to model weights (uses best.pt if None)

        Returns:
            Validation metrics dictionary
        """
        if model_path:
            model = YOLO(model_path)
        else:
            best_path = Path(self.project_name) / self.experiment_name / 'weights' / 'best.pt'
            model = YOLO(str(best_path))

        results = model.val(
            data=str(self.data_yaml),
            imgsz=self.DEFAULT_HYPERPARAMS['imgsz'],
            batch=self.DEFAULT_HYPERPARAMS['batch'],
            project=self.project_name,
            name=f"{self.experiment_name}_val",
            exist_ok=True
        )

        return results

    def export_model(
        self,
        model_path: Optional[str] = None,
        format: str = 'onnx'
    ) -> str:
        """
        Export the trained model to various formats.

        Args:
            model_path: Path to model weights
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)

        Returns:
            Path to exported model
        """
        if model_path:
            model = YOLO(model_path)
        else:
            best_path = Path(self.project_name) / self.experiment_name / 'weights' / 'best.pt'
            model = YOLO(str(best_path))

        exported = model.export(format=format)
        print(f"[INFO] Model exported to: {exported}")

        return exported


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLOv8-OBB for SAR Ship Detection"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to data.yaml configuration'
    )
    parser.add_argument(
        '--model', type=str, default='yolov8n-obb',
        choices=['yolov8n-obb', 'yolov8s-obb', 'yolov8m-obb', 'yolov8l-obb', 'yolov8x-obb'],
        help='YOLOv8-OBB model variant'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', type=int, default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Training device (cuda, mps, cpu)'
    )
    parser.add_argument(
        '--project', type=str, default='trident_sar',
        help='Project name for experiment tracking'
    )
    parser.add_argument(
        '--name', type=str, default='ship_detection',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--no-preprocess', action='store_true',
        help='Skip Lee filter preprocessing'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = SARShipTrainer(
        data_yaml=args.data,
        model_variant=args.model,
        project_name=args.project,
        experiment_name=args.name,
        preprocess_images=not args.no_preprocess,
        device=args.device
    )

    # Custom hyperparameters from args
    hyperparams = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz
    }

    # Train
    best_model = trainer.train(hyperparams=hyperparams, resume=args.resume)

    # Validate
    print("\n[INFO] Running validation on best model...")
    trainer.validate(best_model)


if __name__ == "__main__":
    main()
