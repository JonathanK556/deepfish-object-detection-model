# DeepFish Object Detection Project

A YOLO-based object detection project for detecting fish in underwater images using the DeepFish dataset.

## Dataset Credits

This project uses the **DeepFish** dataset, which is available in YOLO format on Kaggle:

**Kaggle Dataset:**  
[https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection](https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection)

**Original Source:**  
The DeepFish dataset was originally created by Alzayats et al. and made available through 'Papers with Code'. The original dataset and repository can be found at:

- **GitHub Repository:** [https://github.com/alzayats/DeepFish](https://github.com/alzayats/DeepFish)

### Dataset Description

The DeepFish dataset is a robust dataset offering high-resolution fish images in YOLO (You Only Look Once) format, designed for real-time object detection tasks. The dataset comprises several thousands of high-resolution images showcasing a diverse variety of fish species in underwater environments.

Each image file is associated with a corresponding annotation file (.txt file) following the YOLO dataset standard. These annotation files contain:
- Normalized coordinates (0 to 1) for bounding boxes that encapsulate fish in each image
- Class identifiers for the detected fish

### Collection Methodology

The dataset was created to foster advancements in fish detection and classification in underwater environments. Images were collected from various sources, including public image databases, citizen science websites, and commercial fish monitoring systems, to ensure diversity and generalizability. Fish were annotated with bounding boxes manually or through semi-automatic methods.

### License

The dataset is provided under "Other (specified in description)" license. Please refer to the original dataset providers' terms and conditions when utilizing this dataset. If you find this dataset useful for your research or projects, please consider citing the original source to acknowledge the creators' efforts.

## Project Structure

```
ObjectDetection/
├── Deepfish/
│   ├── classes.txt                 # Class labels (Fish)
│   ├── [species_id]/              # Multiple fish species directories
│   │   ├── train/                 # Training images and annotations
│   │   │   ├── *.jpg              # Image files
│   │   │   └── *.txt              # YOLO format annotations
│   │   └── valid/                 # Validation images and annotations
│   │       ├── *.jpg
│   │       └── *.txt
│   └── Nagative_samples/          # Negative samples (no fish)
└── README.md
```

## Dataset Format

The dataset is already formatted for YOLO training:
- **Images:** JPEG format (.jpg)
- **Annotations:** Text files (.txt) containing normalized bounding box coordinates in YOLO format
  - Format: `class_id x_center y_center width height`
  - All coordinates are normalized (0 to 1)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- **ultralytics** - YOLO implementation (supports YOLOv8, YOLOv9, YOLOv10, YOLOv11)
- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision utilities for PyTorch
- **opencv-python** - Image processing and computer vision
- **numpy** - Numerical computing
- **matplotlib** - Visualization
- **pyyaml** - Configuration file handling

See `requirements.txt` for complete dependency list with versions.

## Quick Start

### Prerequisites

1. Download the DeepFish dataset from [Kaggle](https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection)
2. Extract the dataset to this directory as `Deepfish/` (maintaining the original structure)
3. Install dependencies (see Installation section below)

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd ObjectDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook yolo_fish_detection.ipynb
```

The notebook will:
- Explore and validate the dataset structure
- Create a unified YOLO dataset format
- Train a YOLOv8n model on the fish detection task
- Evaluate model performance
- Demonstrate inference on validation images

### Using the Trained Model

After training, you can use the saved model for inference:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/fish_detection/weights/best.pt')

# Run inference on an image
results = model('path/to/your/image.jpg')

# Results contain bounding boxes, confidence scores, and class predictions
for result in results:
    result.show()  # Display image with detections
```

## References

- **Original Dataset Paper:** Alzayats et al. (refer to GitHub repository for full citation)
- **Original GitHub Repository:** [https://github.com/alzayats/DeepFish](https://github.com/alzayats/DeepFish)
- **Kaggle Dataset:** [https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection](https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection)
- **YOLO Documentation:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

## Acknowledgments

We acknowledge and thank:
- Alzayats et al. for creating and sharing the original DeepFish dataset
- Raijin (Kaggle user) for making the dataset available in YOLO format on Kaggle
- The Ultralytics team for providing the YOLO implementation

---

**Note:** This project is for educational and research purposes. Please ensure compliance with the original dataset's license terms when using this dataset.

