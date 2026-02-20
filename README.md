# Brain Tumor Detection Using YOLOv11

## Overview

This repository implements a YOLOv11-based object detection model for automated brain tumor localization in medical imaging data. The model is trained to detect tumor regions using bounding box predictions with high precision and real-time inference capability.

---

## Repository Contents

- `BTD-YOLOV11.ipynb` — Jupyter notebook containing data preparation, model training, and evaluation pipeline.
- `README.md` — Project documentation.

---

## Dataset Information

- Image Type: Medical brain scan images
- Annotation Format: YOLO format (x_center, y_center, width, height)
- Total Images: 2,000
  - Training: 1,600 (80%)
  - Validation: 200 (10%)
  - Testing: 200 (10%)
- Image Size: 640 × 640
- Classes: 1 (Tumor)

---

## Model Configuration

- Model: YOLOv11 (Ultralytics implementation)
- Input Size: 640 × 640
- Epochs: 50
- Batch Size: 16
- Optimizer: SGD
- Initial Learning Rate: 0.01
- Momentum: 0.937
- Weight Decay: 0.0005
- Hardware: NVIDIA RTX 3060 (12GB VRAM)

---

## Performance Metrics

- Precision: 0.93
- Recall: 0.91
- mAP@0.5: 0.95
- mAP@0.5:0.95: 0.88
- F1-Score: 0.92
- Average Inference Time: ~20 ms/image

The model demonstrates strong localization capability with minimal false detections.

---

## Training Command

```
yolo detect train data=data.yaml epochs=50 imgsz=640 batch=16
```

Training results are stored in:

```
runs/detect/train/
```

Best weights file:

```
runs/detect/train/weights/best.pt
```

---

## Inference

```
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/image.jpg", save=True, conf=0.25)
```

The system outputs annotated images with predicted tumor bounding boxes and confidence scores.

---

## Future Improvements

- Multi-class tumor subtype detection
- Larger multi-institution dataset training
- Cross-validation experiments
- TensorRT deployment for real-time clinical systems
- Integration with PACS workflows

---

## Conclusion

This YOLOv11-based tumor detection system achieves high detection accuracy (mAP@0.5 = 0.95) while maintaining real-time performance. The framework provides a strong foundation for AI-assisted tumor screening and clinical research applications.
