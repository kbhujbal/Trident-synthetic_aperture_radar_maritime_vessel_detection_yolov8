"""
SSDD Dataset Manager for SAR Ship Detection

This module handles:
1. Downloading the SSDD dataset from Kaggle via the kaggle API
2. Extracting and crawling the directory structure (no assumptions about folder names)
3. Converting annotations from VOC XML format to YOLO OBB format
4. Auto-generating the data.yaml configuration for YOLOv8-OBB training
5. Splitting data into train/val sets

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
import shutil
import zipfile
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random
import yaml
import cv2
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
        annotation_extensions = {'.xml', '.txt'}

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

    def parse_voc_xml(self, xml_path: Path) -> List[Dict]:
        """
        Parse a Pascal VOC format XML annotation file.

        SSDD typically provides annotations in VOC XML format with:
        - <filename>: Image filename
        - <size>: Image dimensions
        - <object>: Ship bounding boxes with <bndbox> or rotated <robndbox>

        Args:
            xml_path: Path to the XML annotation file

        Returns:
            List of bounding box dictionaries
        """
        boxes = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image size for normalization
            size_elem = root.find('size')
            if size_elem is not None:
                img_width = int(size_elem.find('width').text)
                img_height = int(size_elem.find('height').text)
            else:
                # Default size if not specified
                img_width, img_height = 500, 500

            # Parse each object (ship)
            for obj in root.findall('object'):
                box_data = {
                    'class': obj.find('name').text if obj.find('name') is not None else 'ship',
                    'img_width': img_width,
                    'img_height': img_height
                }

                # Check for rotated bounding box first (OBB format)
                robndbox = obj.find('robndbox')
                if robndbox is not None:
                    # Rotated bounding box format
                    cx = float(robndbox.find('cx').text)
                    cy = float(robndbox.find('cy').text)
                    w = float(robndbox.find('w').text)
                    h = float(robndbox.find('h').text)
                    angle = float(robndbox.find('angle').text)

                    box_data.update({
                        'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle,
                        'is_rotated': True
                    })
                else:
                    # Standard bounding box - convert to OBB with 0 rotation
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)

                        # Convert to center format
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        w = xmax - xmin
                        h = ymax - ymin

                        box_data.update({
                            'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': 0.0,
                            'is_rotated': False
                        })

                if 'cx' in box_data:
                    boxes.append(box_data)

        except ET.ParseError as e:
            print(f"[WARN] Failed to parse XML {xml_path}: {e}")

        return boxes

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
        1. Creates the YOLO directory structure
        2. Converts all annotations to OBB format
        3. Copies images to train/val directories
        4. Generates data.yaml configuration

        Returns:
            Path to the generated data.yaml file
        """
        print("[INFO] Preparing YOLO OBB dataset...")

        # Ensure we have crawled the dataset
        if not self.image_files:
            self.crawl_dataset()

        # Create output directories
        for dir_path in [
            self.train_images_dir, self.val_images_dir,
            self.train_labels_dir, self.val_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Build image-annotation mapping
        # Match by filename stem (without extension)
        annotation_map = {}
        for ann_path in self.annotation_files:
            stem = ann_path.stem
            annotation_map[stem] = ann_path

        # Collect valid image-annotation pairs
        valid_pairs = []
        for img_path in self.image_files:
            stem = img_path.stem
            if stem in annotation_map:
                valid_pairs.append((img_path, annotation_map[stem]))
            else:
                print(f"[WARN] No annotation found for image: {img_path.name}")

        print(f"[INFO] Found {len(valid_pairs)} valid image-annotation pairs")

        # Shuffle and split
        random.seed(self.seed)
        random.shuffle(valid_pairs)

        split_idx = int(len(valid_pairs) * self.train_ratio)
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]

        print(f"[INFO] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        # Process each split
        self._process_split(train_pairs, self.train_images_dir, self.train_labels_dir, "train")
        self._process_split(val_pairs, self.val_images_dir, self.val_labels_dir, "val")

        # Generate data.yaml
        data_yaml_path = self._generate_data_yaml()

        print(f"[INFO] Dataset prepared at: {self.yolo_dir}")
        print(f"[INFO] Configuration saved to: {data_yaml_path}")

        return data_yaml_path

    def _process_split(
        self,
        pairs: List[Tuple[Path, Path]],
        images_dir: Path,
        labels_dir: Path,
        split_name: str
    ):
        """Process a train or val split."""
        for img_path, ann_path in pairs:
            # Copy image
            dst_img = images_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # Convert and save annotation
            if ann_path.suffix.lower() == '.xml':
                boxes = self.parse_voc_xml(ann_path)
            else:
                # Assume already in YOLO format for .txt files
                # Just copy and handle later if needed
                shutil.copy2(ann_path, labels_dir / ann_path.name)
                continue

            # Write YOLO OBB format
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for box in boxes:
                    line = self.convert_to_yolo_obb(box)
                    f.write(line + '\n')

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
        manager.crawl_dataset()
        data_yaml = manager.prepare_yolo_dataset()
        print(f"\n[SUCCESS] Dataset ready for training!")
        print(f"[INFO] Use this config for training: {data_yaml}")


if __name__ == "__main__":
    main()
