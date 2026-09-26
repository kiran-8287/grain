# Rice Quality AI — System Architecture

## 1. High-Level Architecture Overview

The system is designed as a modular, production-ready computer vision and machine learning platform for inspecting raw milled rice grain samples from arbitrary photographs.

```mermaid
graph TD
    A[Client Upload / Frontend UI] -->|HTTP Multipart Form| B[FastAPI Gateway]
    B --> C[RiceQualityPipeline Orchestrator]
    C --> D[Image Preprocessing & EXIF Validation]
    D --> E{Rice Presence Detector}
    E -->|No Rice Detected| F[Early Return: 'No rice grains detected']
    E -->|Rice Present| G[Grain Instance Segmentation]
    G --> H[Classical CV Watershed / Mask R-CNN]
    H --> I[Per-Grain Geometric Analysis]
    I --> J[Calibration Engine: ArUco / Manual / Pixels]
    I --> K[Multi-Label Defect Classifiers]
    K --> K1[Broken Kernel Estimator: Iterative Median]
    K --> K2[Chalky Classifier: Logistic Regression]
    K --> K3[Damaged Classifier: VGG-19 CNN]
    K --> K4[Discoloured Classifier: Adaptive LAB DeltaE]
    K --> K5[Red Grain Classifier: LAB + HSV Proxy]
    K --> K6[Dehusked Classifier: Bran Coverage Proxy]
    K --> K7[Immature/Shrunken: Geometry Proxy]
    K --> K8[Sprouted/Weevilled: ResNet-18 CNN]
    C --> L[Foreign Matter Detector: Full Image YOLO]
    C --> M[Admixture Detector: Robust Mahalanobis]
    C --> N[Image Quality Assessment: Laplacian Blur & Median Pixel Area]
   C --> O[Standards Engine: Historical/Reference Screening]
    C --> P[Annotated Image Renderer: Masks, Contours, IDs]
    P --> Q[Pydantic Result Schema & JSON / CSV Exporters]
```

## 2. Core Modules and Responsibilities

| Module | Location | Purpose | Key Algorithms / Models |
|---|---|---|---|
| **Pipeline** | `ml/pipeline.py` | Orchestrates the 9-stage analysis flow | End-to-end coordination |
| **Preprocessing** | `ml/preprocessing.py` | Validates format, corrects EXIF orientation, guards against decompression bombs | OpenCV, Pillow |
| **Segmentation** | `ml/segmentation.py` | Segments individual grains, splits touching grains, tags merged instances | Watershed, Distance Transform, Contours, Mask R-CNN |
| **Geometry** | `ml/geometry.py` | Calculates Length, Breadth, L/B ratio, Area, Solidity, Ellipse fitting | cv2.fitEllipse, cv2.minAreaRect, Moments |
| **Calibration** | `ml/calibration.py` | Detects physical reference or uses pixel scale | ArUco dictionary, Homography, Manual factor |
| **Chalkiness** | `ml/texture.py` | Evaluates opaque/chalky kernel areas | Logistic Regression, GLCM Contrast/Energy, LAB Brightness |
| **Damaged** | `ml/classifiers.py` | Detects insect/fungal/weather damage | VGG-19 Transfer Learning (Aspect-preserving pad_to_square) |
| **Sprouted/Weevilled** | `ml/classifiers.py` | Identifies germinated tips and insect holes | ResNet-18 Transfer Learning |
| **Colour Analysis** | `ml/colour.py` | Red cuticle coverage, Discolouration DeltaE, Dehusked bran coverage | CIELAB Euclidean DeltaE, HSV ranges |
| **Foreign Matter** | `ml/foreign_matter.py` | Detects non-rice objects across the full image | YOLO11n fine-tuning, Contour morphology |
| **Admixture** | `ml/admixture.py` | Sample-level outlier detection on whole grains | Robust Mahalanobis distance, covariance |
| **Standards** | `ml/standards.py` | Screens observed image fractions against historical/reference limits; suppresses small-sample and unreliable-image verdicts | Reference limits; current KMS 2026-27 not verified |
| **Backend API** | `backend/app/` | FastAPI REST endpoints, job management, export streaming | FastAPI, Pydantic v2, Starlette |
| **Frontend UI** | `frontend/` | Interactive dashboard, annotated viewer, grain inspector | React 18, Vite, TypeScript, Tailwind CSS |

## 3. Data Flow & Separation of Concerns

1. **Training vs Inference Separation**:
   - Model training logic lives in `training/` (`train_chalky.py`, `train_damaged.py`, etc.).
   - Inference strictly loads serialized weights and scalers from `models/` without retraining on startup.
   - If model artifacts are missing, labelled fallback components run automatically.

2. **Per-Grain vs Sample-Level Data**:
   - Grains are independently evaluated with multi-label defect booleans.
   - Foreign Matter, Admixture, and Total Count are strictly flagged as sample-level parameters.
