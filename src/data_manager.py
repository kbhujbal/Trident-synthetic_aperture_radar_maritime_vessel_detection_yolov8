"""
SSDD Dataset Manager for SAR Ship Detection

This module handles:
1. Downloading the SSDD dataset from Kaggle via the kaggle API
2. Extracting and crawling the directory structure (no assumptions about folder names)
3. Converting annotations from COCO/VOC format to YOLO OBB format
4. Auto-generating the data.yaml configuration for YOLOv8-OBB training
5. Splitting data into train/val sets (or using existing splits)

WHY SSDD?
---------
The SAR Ship Detection Dataset (SSDD) is the "ImageNet" of maritime radar detection.
It contains real SAR imagery from multiple satellites (RadarSat-2, TerraSAR-X, Sentinel-1)
with bounding box annotations for ships. This diversity makes models robust to different
SAR sensor characteristics.

WHY OBB (Oriented Bounding Boxes)?
----------------------------------
Ships in SAR imagery are rarely axis-aligned. They appear at arbitrary angles based on
their heading and the satellite's viewing geometry. Standard horizontal bounding boxes
(HBB) waste pixels on water and may overlap adjacent vessels. OBB provides tight fits
around rotated ships, improving both IoU metrics and downstream tracking accuracy.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random
import yaml
import numpy as np
from dataclasses import dataclass


@dataclass
class AnnotationBox:
    """Represents a single bounding box annotation."""
    x_center: float
    y_center: float
    width: float
    height: float
    angle: float  # Rotation angle in radians for OBB
    class_id: int = 0  # Single class: ship


class SSDDDataManager:
    """
    Manages the SSDD dataset lifecycle: download, extraction, conversion, and splitting.

    Supports multiple annotation formats:
    - COCO JSON (from Roboflow exports)
    - Pascal VOC XML
    - YOLO TXT

    The SSDD dataset structure varies between versions. This class crawls the extracted
    directory to find images and annotations regardless of folder naming conventions.
    """

    # Known Kaggle dataset identifiers for SSDD
    # Updated to use available datasets on Kaggle
    KAGGLE_DATASETS = [
        "kailaspsudheer/sarscope-unveiling-the-maritime-landscape",  # Best quality, 378MB
        "petrarodriguez/ls-ssdd-v1-0",  # Large-Scale SSDD, 2.8GB
        "mrearthworm/ssdd-sar-images",  # Compact SSDD, 49MB
    ]

    def __init__(
        self,
        data_root: str = "./data",
        kaggle_dataset: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize the SSDD Data Manager.

        Args:
            data_root: Root directory for dataset storage
            kaggle_dataset: Kaggle dataset identifier (auto-detected if None)
            train_ratio: Fraction of data for training (rest goes to validation)
            seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root).resolve()
        self.kaggle_dataset = kaggle_dataset or self.KAGGLE_DATASETS[0]
        self.train_ratio = train_ratio
        self.seed = seed

        # Paths will be populated after crawling
        self.image_dir: Optional[Path] = None
        self.annotation_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.annotation_files: List[Path] = []

        # Output directories for YOLO format
        self.yolo_dir = self.data_root / "ssdd_yolo_obb"
        self.train_images_dir = self.yolo_dir / "images" / "train"
        self.val_images_dir = self.yolo_dir / "images" / "val"
        self.train_labels_dir = self.yolo_dir / "labels" / "train"
        self.val_labels_dir = self.yolo_dir / "labels" / "val"

    def download_dataset(self, force: bool = False) -> Path:
        """
        Download the SSDD dataset from Kaggle.

        Uses the kaggle CLI which requires:
        1. kaggle package installed: pip install kaggle
        2. API credentials in ~/.kaggle/kaggle.json

        Args:
            force: If True, re-download even if dataset exists

        Returns:
            Path to the downloaded/extracted dataset directory
        """
        download_dir = self.data_root / "ssdd_raw"
        zip_path = self.data_root / "ssdd.zip"

        if download_dir.exists() and not force:
            print(f"[INFO] Dataset already exists at {download_dir}")
            return download_dir

        # Create data directory
        self.data_root.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading SSDD dataset from Kaggle: {self.kaggle_dataset}")
        print("[INFO] Ensure you have kaggle API credentials in ~/.kaggle/kaggle.json")

        try:
            # Use kaggle CLI for download
            cmd = [
                "kaggle", "datasets", "download",
                "-d", self.kaggle_dataset,
                "-p", str(self.data_root),
                "--unzip"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[INFO] Download complete: {result.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Kaggle download failed: {e.stderr}")
            print("[HINT] Run: pip install kaggle && kaggle datasets download -d " + self.kaggle_dataset)
            raise RuntimeError("Failed to download dataset from Kaggle") from e
        except FileNotFoundError:
            print("[ERROR] kaggle CLI not found. Install with: pip install kaggle")
            raise

        # Handle various extraction scenarios
        # Kaggle --unzip extracts directly, but we need to find the actual data
        extracted_items = list(self.data_root.iterdir())

        # Look for the extracted folder (could have various names)
        for item in extracted_items:
            if item.is_dir() and item.name not in ["ssdd_yolo_obb", "ssdd_raw"]:
                # Found an extracted directory, use it
                if download_dir.exists():
                    shutil.rmtree(download_dir)
                item.rename(download_dir)
                break
        else:
            # Files extracted directly into data_root, move them
            download_dir.mkdir(exist_ok=True)
            for item in extracted_items:
                if item.name not in ["ssdd_yolo_obb", "ssdd_raw"]:
                    shutil.move(str(item), str(download_dir / item.name))

        print(f"[INFO] Dataset extracted to: {download_dir}")
        return download_dir

    def crawl_dataset(self, root_dir: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:
        """
        Crawl the dataset directory to find images and annotations.

        WHY CRAWL?
        ----------
        Different versions of SSDD use different folder structures:
        - Some use JPEGImages/, some use images/
        - Some use Annotations/, some use labels/
        - Some have nested structures
        - Some use COCO JSON, some use VOC XML

        By crawling, we make the pipeline robust to these variations.

        Args:
            root_dir: Directory to crawl (defaults to data_root/ssdd_raw)

        Returns:
            Tuple of (image_paths, annotation_paths)
        """
        root_dir = root_dir or (self.data_root / "ssdd_raw")

        if not root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        print(f"[INFO] Crawling dataset directory: {root_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        annotation_extensions = {'.xml', '.txt', '.json'}

        images = []
        annotations = []

        # Use os.walk for recursive directory traversal
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirpath = Path(dirpath)

            for filename in filenames:
                filepath = dirpath / filename
                ext = filepath.suffix.lower()

                if ext in image_extensions:
                    images.append(filepath)
                elif ext in annotation_extensions:
                    annotations.append(filepath)

        # Sort for reproducibility
        images.sort()
        annotations.sort()

        print(f"[INFO] Found {len(images)} images and {len(annotations)} annotations")

        # Store for later use
        self.image_files = images
        self.annotation_files = annotations

        # Infer directories
        if images:
            self.image_dir = images[0].parent
        if annotations:
            self.annotation_dir = annotations[0].parent

        return images, annotations

    def detect_dataset_format(self, root_dir: Optional[Path] = None) -> str:
        """
        Detect the annotation format of the dataset.

        Returns:
            'coco': COCO JSON format (_annotations.coco.json)
            'voc': Pascal VOC XML format
            'yolo': YOLO TXT format
            'presplit_coco': Pre-split with train/valid/test and COCO JSON
        """
        root_dir = root_dir or (self.data_root / "ssdd_raw")

        # Check for pre-split COCO format (Roboflow style)
        train_coco = root_dir / "train" / "_annotations.coco.json"
        valid_coco = root_dir / "valid" / "_annotations.coco.json"

        if train_coco.exists() and valid_coco.exists():
            print("[INFO] Detected pre-split COCO format (Roboflow style)")
            return 'presplit_coco'

        # Check for single COCO file
        coco_files = list(root_dir.glob("**/*.json"))
        if coco_files:
            for f in coco_files:
                if 'annotations' in f.name.lower() or 'coco' in f.name.lower():
                    return 'coco'

        # Check for VOC XML
        xml_files = list(root_dir.glob("**/*.xml"))
        if xml_files:
            return 'voc'

        # Check for YOLO TXT
        txt_files = list(root_dir.glob("**/*.txt"))
        # Filter out README files
        txt_files = [f for f in txt_files if 'readme' not in f.name.lower()]
        if txt_files:
            return 'yolo'

        return 'unknown'

    def parse_coco_json(self, json_path: Path) -> Dict[str, List[Dict]]:
        """
        Parse a COCO format JSON annotation file.

        Args:
            json_path: Path to the COCO JSON file

        Returns:
            Dictionary mapping image filenames to list of bounding boxes
        """
        with open(json_path, 'r') as f:
            coco = json.load(f)

        # Build image ID to filename mapping
        id_to_image = {}
        for img in coco.get('images', []):
            id_to_image[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }

        # Group annotations by image
        image_annotations = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in id_to_image:
                continue

            img_info = id_to_image[img_id]
            filename = img_info['file_name']

            if filename not in image_annotations:
                image_annotations[filename] = []

            # COCO bbox format: [x, y, width, height] (top-left corner)
            bbox = ann['bbox']
            x, y, w, h = bbox

            # Convert to center format
            cx = x + w / 2
            cy = y + h / 2

            box_data = {
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'angle': 0.0,  # COCO doesn't have rotation, default to 0
                'img_width': img_info['width'],
                'img_height': img_info['height']
            }

            image_annotations[filename].append(box_data)

        return image_annotations

    def convert_to_yolo_obb(self, box: Dict) -> str:
        """
        Convert a bounding box to YOLO OBB format.

        YOLO OBB FORMAT:
        ----------------
        class_id x1 y1 x2 y2 x3 y3 x4 y4

        Where (x1,y1), (x2,y2), (x3,y3), (x4,y4) are the four corners of the
        rotated bounding box, normalized to [0, 1] by image dimensions.

        This format is required by ultralytics YOLOv8-OBB models.

        Args:
            box: Dictionary with cx, cy, w, h, angle, img_width, img_height

        Returns:
            YOLO OBB format string
        """
        cx = box['cx']
        cy = box['cy']
        w = box['w']
        h = box['h']
        angle = box['angle']
        img_w = box['img_width']
        img_h = box['img_height']

        # Calculate the four corners of the rotated rectangle
        # Start with corners relative to center
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Half dimensions
        hw, hh = w / 2, h / 2

        # Corner offsets (before rotation)
        corners = np.array([
            [-hw, -hh],  # top-left
            [hw, -hh],   # top-right
            [hw, hh],    # bottom-right
            [-hw, hh],   # bottom-left
        ])

        # Rotation matrix
        rotation = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Rotate corners and translate to center
        rotated_corners = corners @ rotation.T + np.array([cx, cy])

        # Normalize by image dimensions
        rotated_corners[:, 0] /= img_w
        rotated_corners[:, 1] /= img_h

        # Clip to [0, 1]
        rotated_corners = np.clip(rotated_corners, 0, 1)

        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        coords = rotated_corners.flatten()
        return f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f} {coords[4]:.6f} {coords[5]:.6f} {coords[6]:.6f} {coords[7]:.6f}"

    def prepare_yolo_dataset(self) -> Path:
        """
        Prepare the dataset in YOLO OBB format with train/val split.

        This method:
        1. Detects the annotation format
        2. Creates the YOLO directory structure
        3. Converts all annotations to OBB format
        4. Copies images to train/val directories
        5. Generates data.yaml configuration

        Returns:
            Path to the generated data.yaml file
        """
        print("[INFO] Preparing YOLO OBB dataset...")

        raw_dir = self.data_root / "ssdd_raw"

        # Detect format
        format_type = self.detect_dataset_format(raw_dir)
        print(f"[INFO] Detected format: {format_type}")

        if format_type == 'presplit_coco':
            return self._prepare_from_presplit_coco(raw_dir)
        elif format_type == 'coco':
            return self._prepare_from_coco(raw_dir)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _prepare_from_presplit_coco(self, raw_dir: Path) -> Path:
        """
        Prepare dataset from pre-split COCO format (Roboflow style).

        Expected structure:
        raw_dir/
            train/
                _annotations.coco.json
                image1.jpg
                ...
            valid/
                _annotations.coco.json
                ...
            test/ (optional)
                _annotations.coco.json
                ...
        """
        # Create output directories
        for dir_path in [
            self.train_images_dir, self.val_images_dir,
            self.train_labels_dir, self.val_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Process train split
        train_dir = raw_dir / "train"
        train_coco = train_dir / "_annotations.coco.json"
        self._process_coco_split(train_coco, train_dir, self.train_images_dir, self.train_labels_dir, "train")

        # Process validation split (Roboflow uses "valid")
        val_dir = raw_dir / "valid"
        val_coco = val_dir / "_annotations.coco.json"
        self._process_coco_split(val_coco, val_dir, self.val_images_dir, self.val_labels_dir, "val")

        # Generate data.yaml
        data_yaml_path = self._generate_data_yaml()

        print(f"[INFO] Dataset prepared at: {self.yolo_dir}")
        print(f"[INFO] Configuration saved to: {data_yaml_path}")

        return data_yaml_path

    def _process_coco_split(
        self,
        coco_json: Path,
        images_src_dir: Path,
        images_dst_dir: Path,
        labels_dst_dir: Path,
        split_name: str
    ):
        """Process a single split from COCO format."""
        print(f"[INFO] Processing {split_name} split...")

        # Parse COCO annotations
        image_annotations = self.parse_coco_json(coco_json)

        processed = 0
        skipped = 0

        for filename, boxes in image_annotations.items():
            src_image = images_src_dir / filename

            if not src_image.exists():
                skipped += 1
                continue

            # Copy image
            dst_image = images_dst_dir / filename
            shutil.copy2(src_image, dst_image)

            # Write labels in YOLO OBB format
            label_filename = Path(filename).stem + ".txt"
            label_path = labels_dst_dir / label_filename

            with open(label_path, 'w') as f:
                for box in boxes:
                    line = self.convert_to_yolo_obb(box)
                    f.write(line + '\n')

            processed += 1

        print(f"[INFO] {split_name}: Processed {processed} images, skipped {skipped}")

    def _prepare_from_coco(self, raw_dir: Path) -> Path:
        """Prepare dataset from a single COCO annotation file with random split."""
        # Find COCO JSON file
        coco_files = list(raw_dir.glob("**/*annotations*.json"))
        if not coco_files:
            coco_files = list(raw_dir.glob("**/*.json"))

        if not coco_files:
            raise FileNotFoundError("No COCO JSON file found")

        coco_json = coco_files[0]
        print(f"[INFO] Using COCO file: {coco_json}")

        # Create output directories
        for dir_path in [
            self.train_images_dir, self.val_images_dir,
            self.train_labels_dir, self.val_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Parse annotations
        image_annotations = self.parse_coco_json(coco_json)

        # Find images directory
        images_dir = coco_json.parent
        if (images_dir / "images").exists():
            images_dir = images_dir / "images"

        # Split filenames
        filenames = list(image_annotations.keys())
        random.seed(self.seed)
        random.shuffle(filenames)

        split_idx = int(len(filenames) * self.train_ratio)
        train_files = filenames[:split_idx]
        val_files = filenames[split_idx:]

        print(f"[INFO] Train: {len(train_files)}, Val: {len(val_files)}")

        # Process splits
        self._process_file_list(train_files, image_annotations, images_dir,
                                self.train_images_dir, self.train_labels_dir)
        self._process_file_list(val_files, image_annotations, images_dir,
                                self.val_images_dir, self.val_labels_dir)

        return self._generate_data_yaml()

    def _process_file_list(
        self,
        filenames: List[str],
        annotations: Dict[str, List[Dict]],
        src_dir: Path,
        images_dst: Path,
        labels_dst: Path
    ):
        """Process a list of files."""
        for filename in filenames:
            src_image = src_dir / filename
            if not src_image.exists():
                continue

            # Copy image
            shutil.copy2(src_image, images_dst / filename)

            # Write labels
            boxes = annotations.get(filename, [])
            label_path = labels_dst / (Path(filename).stem + ".txt")

            with open(label_path, 'w') as f:
                for box in boxes:
                    f.write(self.convert_to_yolo_obb(box) + '\n')

    def _generate_data_yaml(self) -> Path:
        """
        Generate the data.yaml configuration file for YOLOv8-OBB.

        This file tells the YOLO trainer where to find images and labels,
        and defines the class names.

        Returns:
            Path to the generated data.yaml file
        """
        data_yaml_path = self.yolo_dir / "data.yaml"

        config = {
            'path': str(self.yolo_dir.resolve()),  # Absolute path to dataset root
            'train': 'images/train',                # Relative path to train images
            'val': 'images/val',                    # Relative path to val images

            # Class configuration
            'names': {
                0: 'ship'
            },

            # Number of classes
            'nc': 1,

            # Metadata
            '_comment': 'SSDD Dataset for SAR Ship Detection with Oriented Bounding Boxes'
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return data_yaml_path

    def get_sample_images(self, n: int = 5, split: str = 'val') -> List[Path]:
        """
        Get random sample images from the prepared dataset.

        Args:
            n: Number of samples to return
            split: 'train' or 'val'

        Returns:
            List of image paths
        """
        if split == 'train':
            img_dir = self.train_images_dir
        else:
            img_dir = self.val_images_dir

        if not img_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {img_dir}")

        images = list(img_dir.glob('*'))
        images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}]

        random.seed(self.seed)
        return random.sample(images, min(n, len(images)))


def main():
    """Main entry point for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SSDD Dataset Manager for SAR Ship Detection"
    )
    parser.add_argument(
        '--data-root', type=str, default='./data',
        help='Root directory for dataset storage'
    )
    parser.add_argument(
        '--kaggle-dataset', type=str, default=None,
        help='Kaggle dataset identifier'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Fraction of data for training'
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download dataset from Kaggle'
    )
    parser.add_argument(
        '--prepare', action='store_true',
        help='Prepare YOLO OBB format dataset'
    )

    args = parser.parse_args()

    manager = SSDDDataManager(
        data_root=args.data_root,
        kaggle_dataset=args.kaggle_dataset,
        train_ratio=args.train_ratio
    )

    if args.download:
        manager.download_dataset()

    if args.prepare:
        data_yaml = manager.prepare_yolo_dataset()
        print(f"\n[SUCCESS] Dataset ready for training!")
        print(f"[INFO] Use this config for training: {data_yaml}")


if __name__ == "__main__":
    main()
