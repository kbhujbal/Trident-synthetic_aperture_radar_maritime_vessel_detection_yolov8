# Trident - SAR Maritime Vessel Detection

A production-grade maritime surveillance system for detecting ships in Synthetic Aperture Radar (SAR) imagery using Oriented Bounding Boxes (OBB) and speckle noise filtering.

## Mission

Detect ships in SAR images where vessels appear as noisy "salt-and-pepper" clusters, using:
- **Lee Speckle Filter**: Remove radar-specific multiplicative noise
- **YOLOv8-OBB**: Detect ships with rotated bounding boxes (ships are rarely horizontal)

## Why This Approach?

### The SAR Challenge
SAR images suffer from "speckle noise" - a granular interference pattern unique to coherent radar imaging. Ships appear as bright clusters mixed with noise, making standard detection difficult.

### Why Speckle Filtering?
```
Before Lee Filter:          After Lee Filter:
░░█░░█░░░░█░░█░░           ░░░░░░░░░░░░░░░░
░█░█░█░░░█░█░░░░           ░░░████████░░░░░
░░█░░░█░█░░█░░░░    →      ░░░████████░░░░░
░█░█░░░█░░█░░░░░           ░░░████████░░░░░
░░░█░█░░░░░█░░░░           ░░░░░░░░░░░░░░░░
```

### Why Oriented Bounding Boxes?
Ships sail in arbitrary directions. Axis-aligned boxes waste pixels and overlap:
```
Standard Box (AABB):        Oriented Box (OBB):
┌─────────────────┐            ╱████████████╲
│  ████████       │           ╱████████████╲
│       ████████  │    →       ╲████████████╱
│           █████ │            ╲████████████╱
└─────────────────┘
  60% background!              Tight fit!
```

## Project Structure

```
Trident/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_manager.py      # Kaggle download, VOC→YOLO conversion
│   ├── preprocessing.py     # Lee speckle filter implementation
│   ├── train.py             # YOLOv8-OBB training pipeline
│   └── inference.py         # Detection and visualization
├── config/
│   └── config.yaml          # Configuration parameters
├── data/                    # Dataset storage
├── output/                  # Detection results
├── runs/                    # Training checkpoints
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd Trident

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Kaggle Setup

```bash
# Install kaggle CLI
pip install kaggle

# Configure API credentials
# 1. Go to kaggle.com → Account → Create New API Token
# 2. Save kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download & Prepare Dataset

```bash
cd src

# Download SSDD from Kaggle
python data_manager.py --download --data-root ../data

# Prepare YOLO OBB format
python data_manager.py --prepare --data-root ../data
```

### 4. Train the Model

```bash
# Train with default settings (YOLOv8n-OBB)
python train.py --data ../data/ssdd_yolo_obb/data.yaml

# Train with custom settings
python train.py \
    --data ../data/ssdd_yolo_obb/data.yaml \
    --model yolov8s-obb \
    --epochs 150 \
    --batch 32 \
    --imgsz 640
```

### 5. Run Inference

```bash
# Demo mode (random validation images)
python inference.py \
    --model ../trident_sar/ship_detection/weights/best.pt \
    --data ../data/ssdd_yolo_obb/data.yaml \
    --output ../output

# Single image
python inference.py \
    --model ../trident_sar/ship_detection/weights/best.pt \
    --source /path/to/sar_image.jpg \
    --output ../output

# Directory of images
python inference.py \
    --model ../trident_sar/ship_detection/weights/best.pt \
    --source /path/to/images/ \
    --output ../output
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
preprocessing:
  lee_filter:
    window_size: 7        # Larger = more smoothing

model:
  variant: "yolov8n-obb"  # n/s/m/l/x variants
  image_size: 640

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
```

## Model Variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n-obb | 3.2M | Fastest | Good |
| yolov8s-obb | 11.2M | Fast | Better |
| yolov8m-obb | 25.9M | Medium | High |
| yolov8l-obb | 43.7M | Slow | Higher |
| yolov8x-obb | 68.2M | Slowest | Highest |

## API Usage

```python
from src.preprocessing import SARPreprocessor, apply_lee_filter
from src.inference import SARShipDetector

# Preprocess a SAR image
preprocessor = SARPreprocessor(lee_window_size=7)
filtered = preprocessor.process(sar_image)

# Detect ships
detector = SARShipDetector(
    model_path="trident_sar/ship_detection/weights/best.pt",
    confidence_threshold=0.25
)
result = detector.detect("path/to/sar_image.jpg")

print(f"Ships detected: {result['count']}")
for det in result['detections']:
    print(f"  Confidence: {det['confidence']:.2f}")
    print(f"  Angle: {det['angle']:.1f}°")
```

## Output Format

Detection results include:
- `corners`: 4 corner points of the rotated box
- `center`: Center coordinates
- `width`, `height`: Box dimensions
- `angle`: Rotation angle in degrees
- `confidence`: Detection confidence

## Dataset: SSDD

The SAR Ship Detection Dataset (SSDD) contains:
- 1,160 SAR images from RadarSat-2, TerraSAR-X, Sentinel-1
- 2,456 ship annotations
- Mixed resolutions and polarizations

## Technical Details

### Lee Filter Algorithm

The Lee filter assumes multiplicative noise: `I = R × N`

```
R̂ = μ + W × (I - μ)
W = σ² / (σ² + σ_noise²)
```

Where:
- `μ` = local mean
- `σ²` = local variance
- `W` = adaptive weight (preserves edges)

### OBB Output Format

YOLO OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4`

Four normalized corner coordinates defining the rotated rectangle.

## License

MIT License

## Acknowledgments

- SSDD Dataset creators
- Ultralytics YOLOv8 team
- Jong-Sen Lee (Lee filter inventor)
