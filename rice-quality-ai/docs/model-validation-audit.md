# Model Validation Audit

**Audit date:** 2026-09-26  
**Scope:** The 14 project parameters, checked-in training code, model registry and metadata, manifests, available data, checkpoints, inference preprocessing, confidence/threshold handling, image-quality calculation, and standard-screening threshold. This audit does not modify the rice-presence gate or segmentation pipeline.

## Executive Finding

No defect-classification model in this repository is demonstrated to have been trained and independently evaluated on real, labeled rice-grain data. The damaged and sprouted/weevilled checkpoints are synthetic demonstrations. The chalky checkpoint is based on generated feature vectors and is disabled for inference. The foreign-matter YOLO model is configured but not trained or available. There are no image or annotation files under `data/raw` or `data/processed` beyond `.gitkeep` files.

A model architecture name and a saved checkpoint do not establish dataset validity. The checked-in perfect metrics for synthetic examples are not evidence of real rice-grain performance.

## External Dataset Discovery

| Dataset | Verified source facts | Supported task candidates | Blocker before use |
|---|---|---|---|
| GrainDet-Rice v2 | Figshare API reports CC BY 4.0, 1.84 GB, about 20K single-kernel images. Project documentation describes expert labels: normal, sprouted, fusarium/shriveled, broken, pest-attacked, impurities and unripe. Project page reports 24.7K and split counts totaling 24.2K; Figshare says 20K. | Broken, sprouted-only, damaged subtypes, unripe/shriveled proxy; impurities may be object-level labels. | Direct download redirected to HTTP 403 after 146,966,036 bytes. The partial archive was not extracted or used. Reconcile counts, inspect image/mask content, split manifest, duplicates and source-group policy. Pest-attacked is not confirmed as weevilled. |
| GrainSet Rice v3 | Figshare API reports CC BY 4.0, 2.62 GB, >30K single-kernel images/masks and a `rice.xml` annotations file; publisher describes expert quality annotations and sample metadata. | Geometry/count evaluation; damage/unsound categories after annotation audit. | Archive not downloaded; annotation class map, image modes, group split and individual parameter labels remain uninspected. |
| Roboflow Rice Grain Defects v1 | Dataset card reports 2,000 images, object detection, CC BY 4.0, 1,600/200/200 split, stretched 256px input. Inspected sample pages show filenames such as `Broken0959.jpg` with ground-truth boxes; class IDs 0-3 are not named on the public card. | Potential defect detection after a verified class map. | Do not use until export reveals class names, original/sample groups, annotations, and source leakage. |
| Roboflow rice-grain-quality-detection v1 | 224 images, object detection, CC BY 4.0; classes `broken_rice`, `chalky_rice`, `foreign_object`, `head_rice`, `unhulled_rice`; 180/22/22 split, stretched 640px, no augmentations. Inspected image pages show ground-truth box counts. | Exploratory broken/chalky/foreign-object detection only. | Very small split; inspected pages expose the same group label `rice-quality`; no independent sample grouping or class counts. Not suitable for defensible held-out evaluation as published. |
| Murat Koklu Rice Image Dataset | Kaggle card reports 75,000 real grain images, 15,000 each across Arborio/Basmati/Ipsala/Jasmine/Karadag; CC0 1.0. | Candidate variety/appearance negatives after manual quality review. | Variety labels are not “clean” defect labels. It has no defect annotations or source-group IDs. |
| UCI Rice Cammeo/Osmancik | 3,810 tabular records, seven morphometric features plus variety class, CC BY 4.0; no image files in repository archive. | Tabular variety research only. | Not suitable to train image classifiers or detectors. |
| IIT Indore milled-rice damage dataset (2022) | Institutional record reports 8,048 high-magnification images and seven damage classes; 4.5x acquisition described. | Potential chalky/discoloured/damage classification. | Repository record has no files attached; full taxonomy, license, splits and images are unavailable there. Do not train from abstract claims. |

No candidate has been represented as locally downloaded or training-ready. The GrainDet partial file is named `data/downloads/graindet_rice.zip.partial` and must not be opened as a dataset. Machine-readable candidate records are in `data/manifests/datasets.json`.

## Rice Gate Provenance

Gate regression behavior is retained. The checked-in Logistic Regression gate was generated from 1,200 hand-sampled synthetic feature vectors per class, with a random feature split; its perfect metrics do not validate real-image performance. The live artifact remains in place for compatibility, but registry/output metadata now marks its score experimental and uncalibrated. The legacy trainer refuses to overwrite it without a real manifest. Gate and segmentation decisions were not reworked in this pass.

## ML Model Audit

| Parameter | Model / checkpoint | Dataset, labels, counts and split | Preprocessing, augmentation and inference | Threshold / score | Audit status |
|---|---|---|---|---|---|
| Damaged | VGG-19; `models/damaged/model.pth` (558,336,281 bytes) | `SyntheticGrainCropDataset`; generated ellipse illustrations with artificial color and damage spots. Classes: `0 normal`, `1 damaged`; 80 train crops (40/class), 30 test crops (15/class), no validation split. No source dataset or real grain labels. | Training generated square 224x224 RGB canvases, black outside the ellipse, `ToTensor` + ImageNet normalization; no augmentation. VGG-19 ImageNet backbone when downloadable, feature layers frozen, replacement dropout/linear head trained. Current inference uses a mask-applied, aspect-preserving 224 crop and the same normalization. | Two-class argmax (equivalent to a 0.5 boundary); top-class softmax score is not calibrated. No learned validation threshold. | **B: Synthetic/demo ML; experimental.** Binary labels do not distinguish slightly damaged from damaged. Registry and UI identify the result as experimental/uncalibrated. The old synthetic trainer is guarded against overwriting the checkpoint or emitting demo metrics. |
| Chalky | Logistic Regression + StandardScaler; legacy `models/chalky/model.joblib` and `scaler.joblib` | Legacy script generated 600 feature vectors (300 normal, 300 chalky) from hand-authored distributions; recorded split was 419 train, 91 validation, 90 test. These were not images, grains, or mask-labeled examples. The current trainer requires a real `manifest.csv`; no dataset is present. | Current feature extraction calculates LAB statistics only at grain-mask pixels and GLCM pairs only when both pixels are inside the mask. Training uses this same extractor and a source-group split. No augmentation. | A future real-data trainer selects a threshold on validation F1. Existing synthetic artifact has no valid real-data threshold and its metadata is `unvalidated`; inference returns `Undetermined` with no probability. | **Unavailable for classification.** Synthetic artifacts are not a production source. Real labeled grain-level training is required. |
| Sprouted / Weevilled | ResNet-18; `models/sprouted_weevilled/model.pth` | Checked-in metadata says synthetic demonstration crops. The prior generator script produced 60 training and 24 test illustrations, alternating `normal` and `sprouted_weevilled` (30/12 per class); no validation set. The saved checkpoint is not cryptographically linked to a training run, so those counts describe the recorded trainer, not independently verified checkpoint provenance. Class mapping: `0 normal`, `1 sprouted_weevilled`. | Prior training used 224x224 generated rice ellipses with added green sprouts or dark cavities, ImageNet normalization, and no augmentation; pretrained ResNet-18 backbone frozen with dropout/linear head. Current trainer requires real masked grain images and source groups; no corpus exists. Its mask-applied square crop and ImageNet normalization match current inference. | Inference reads the saved class mapping. Decision threshold is metadata value if present, otherwise 0.5. Displayed confidence is the predicted class softmax score, not calibrated and not a physical grain fraction. | **B: Synthetic/demo ML; experimental / uncalibrated.** Prediction may be shown only with this status and limitation. No real clean or positive validation has been run. |
| Foreign Matter | YOLO11n configured; no `yolo11n_foreign.pt` checkpoint | Manifest lists source names `Rice-Quality 3 Foreign Matter` and `Agricultural Non-Grain Contaminants`, taxonomy `stone`, `inorganic`, `organic`, `other_foreign_matter`, and an intended 80/20 train/validation split. Dataset YAML points to `data/processed/foreign_matter`; that directory is absent/empty. No class counts, annotations, independent test set, or provenance verification are available. | Intended full-image object detection. Ultralytics preprocessing would apply at runtime if a trained weight file existed. Current fallback uses foreground contours and mean-color rules; rice masks are excluded and the YOLO-unavailable warning remains. | YOLO training threshold/metrics unavailable. Heuristic threshold is an engineering setting, not a model threshold. | **C: Heuristic fallback active; YOLO untrained. Dataset provenance unknown.** The source names alone do not prove class coverage. Plastic, other seeds, dirt, and non-rice objects have not been verified in labels. |

### Class and Training Details

- **Damaged:** The class mapping is strictly `normal` vs `damaged`. “Slightly damaged” is not a trained class. Its synthetic test metrics were removed from active metadata because they measured generated illustrations, not rice.
- **Chalky:** Brightness and color features are descriptive only. The former synthetic-feature model must not emit a label or probability. Grain-level classification does not claim pixel-level chalky-region segmentation.
- **Sprouted / Weevilled:** Current class indices are explicitly loaded from `class_mapping.json`, checked against metadata, and logged. The existing artifact is experimental regardless of its architecture or softmax score.
- **Foreign matter:** The configured taxonomy is not evidence of training. Until a YOLO checkpoint and annotated dataset are supplied, the existing heuristic and its warning are the active detector.
- **Other classifiers:** No training scripts or model checkpoints exist for broken, discoloured, red, dehusked, immature/shrunken, or admixture classification. Their current implementations are rules, geometric estimators, color proxies, or statistical proxies.

## Provenance of the 14 Parameters

| # | Parameter | Current method | Provenance category | Current interpretation |
|---:|---|---|---|---|
| 1 | Broken | Length compared with robust whole-kernel estimate using the 0.75 concept threshold | **F** official concept plus computer-vision estimator | `Undetermined` when the sample cannot establish a whole-kernel reference (including N <= 2); not an official weight test. |
| 2 | Damaged | VGG-19 binary demo checkpoint | **B** synthetic/demo ML | Experimental prediction only; no real rice training/evaluation; no slightly-damaged class. |
| 3 | Discoloured | CIE76 LAB distance from sample median reference | **C** computer-vision heuristic | Image color proxy, not validated against laboratory labels. |
| 4 | Chalky | No production classification | **Unavailable** | `Undetermined` until a real labeled grain-level classifier is trained and independently validated. |
| 5 | Red | LAB/HSV surface-color rules | **C** computer-vision heuristic | Image proxy; no labeled performance evaluation. |
| 6 | Dehusked | HSV brown/bran-coverage proxy | **F** official concept, not directly measurable from image | Not equivalent to official chemical staining. |
| 7 | Immature / Shrunken | Relative geometry and solidity rules | **C** computer-vision heuristic | Not a trained classifier; relative evidence is weak for very small samples. |
| 8 | Sprouted / Weevilled | ResNet-18 synthetic checkpoint | **B** synthetic/demo ML | Prediction and uncalibrated class score must be marked experimental. |
| 9 | Foreign Matter | YOLO configured but unavailable; contour/color fallback | **C** computer-vision heuristic | Full-image fallback only; no trained detector evidence. |
| 10 | Admixture | Unsupported; Mahalanobis geometry outlier is a separate diagnostic | **Unavailable** | A geometric outlier is not evidence of lower-class variety. API returns null count/fraction and `admixture_status: unsupported`. |
| 11 | Length | Grain-mask major-axis geometry | **D** mathematical image measurement | Pixels without physical calibration; millimeters only with a valid scale. |
| 12 | Breadth | Grain-mask minor-axis geometry | **D** mathematical image measurement | Pixels without physical calibration; millimeters only with a valid scale. |
| 13 | L/B Ratio | Length / breadth | **D** derived mathematical measurement | Dimensionless. |
| 14 | Total Count | Count of accepted rice instances after the existing gate and segmentation path | **C** computer-vision instance count | Case tests pass; general accuracy remains dependent on image and segmentation quality. Foreign objects must not become grain IDs. |

## Current Usability

- **Usable as scoped image measurements:** Length, breadth and L/B ratio, with pixel units unless calibrated. Total count works for tested inputs but is a computer-vision count, not a learned or universally validated count.
- **Experimental model outputs only:** Damaged and sprouted/weevilled. They are synthetic demonstrations and are not scientifically validated.
- **Unavailable:** Chalky classifier; trained YOLO foreign-matter model.
- **Heuristic image predictions:** Broken estimator, discoloured, red, dehusked visual proxy, immature/shrunken, and foreign-matter fallback. These must be described as proxies, not validated model outputs.
- **Diagnostic only:** Mahalanobis geometry outliers; the admixture parameter remains unsupported with null count/fraction.
- **Official concept not directly measurable from RGB:** Dehusked chemical staining and official weight-based standards. Image fractions are not lot compliance.

## Datasets Required

| Target | Required real data and labels | Rice images? | Directly trainable or adaptation needed? |
|---|---|---|---|
| Chalky | Grain-level image + instance mask + object label. Negative labels must include clean normal rice, naturally bright white rice, translucent rice, and varied normal appearances/varieties/backgrounds. Positive labels must be genuinely chalky grains. Split by source image, acquisition session, or lot to prevent leakage. Pixel masks of chalky subregions are not required for a grain-level classifier. | Yes; every class is real rice. | Adapt current Logistic Regression manifest pipeline directly after assembling the dataset. Train on mask-normalized LAB/GLCM features; select threshold only on validation data; evaluate clean-rice false-positive rate, precision, recall, F1, confusion matrix and calibration. |
| Damaged | Real masked grain crops with source groups and adjudicated labels `normal`, `damaged`, and, if the project reports it separately, `slightly_damaged`. Include visual lesion type/grade annotation guidance and avoid mixing foreign objects into positive labels. | Yes; every class is real rice. | Requires a real-manifest trainer and a reviewed class mapping. Existing synthetic trainer is disabled from artifact-writing; current checkpoint is not a starting validation set. |
| Sprouted / Weevilled | Real masked grain crops with source groups and labels `normal`, `sprouted`, and `weevilled` (or a documented combined positive class). Include several varieties, lighting/backgrounds and confirmed positive examples. | Yes; every class is real rice. | Current manifest trainer supports a combined binary positive class and group splits. Three-way subtype prediction would require mapping/model-head changes. Keep a group-disjoint untouched test split; calibrate probabilities separately if claiming calibration. |
| Foreign Matter | Full images with object bounding boxes and reviewed labels for stones, plastic, organic matter/chaff, other seeds, dirt/soil, and other non-rice objects. Include rice-only and mixed rice/foreign images plus varied backgrounds. | Mixed images should contain rice and non-rice examples; foreign objects themselves are not rice. | Adapt/reconcile labels to the existing four YOLO classes. The named sources in the manifest may help, but their contents, license, class coverage and annotations are not present or verified here. Use a separate source-grouped test set. |
| Total Count / learned segmentation (optional upgrade) | Full rice images with instance masks for every accepted grain, including touching/overlapping grains, plus source-group splits. | Yes. | The manifest describes a COCO rice segmentation dataset, but no train/validation annotation files are present. Current Case behavior remains on the existing classical-CV path; this audit does not change it. |

No minimum number of grains per class is asserted here: the repository does not establish one scientifically. A valid group split must contain both labels in train, validation, and test, with no source image/session shared across splits. Dataset acquisition must preserve source/license and label-review provenance.

## Image-Quality Threshold Audit

`ml/quality.py` uses the following hand-configured thresholds from `configs/thresholds.json`:

- Blur is variance of the Laplacian sampled only at accepted grain-mask pixels; pixels outside the mask are set to a deterministic zero before convolution. Scores: 1.0 at >= 100, 0.7 at >= 50, 0.4 at >= 20, otherwise 0.1. This reduces background dependence but remains uncalibrated.
- Grain size is median instance-mask area in pixels. Scores: 1.0 at >= 400 px, 0.7 at >= 100 px, 0.4 at >= 25 px, otherwise 0.1.
- Uncertain-grain fraction above 0.20 scores 0.5; otherwise 1.0.
- Illumination uses block-mean standard deviation over accepted grain-mask pixels to form `1 - min(1, std/50)`. Uniformity below 0.5 scores 0.5; otherwise the score is `min(1, uniformity + 0.2)`.
- Clipping is measured over accepted grain-mask pixels when available; it scores 0.6 if more than 10% are below gray 5 or above gray 250, otherwise 1.0. Without a valid mask it falls back to the full image.
- Five factor scores are averaged equally. Score bands are GOOD >= 0.80, FAIR >= 0.60, otherwise POOR.
- UNRELIABLE is an explicit hard-failure state if any configured criterion applies: blur below 20 Laplacian variance, median grain area below 25 px, mean segmentation confidence below the existing 0.5 segmentation acceptance threshold, uncertain/merged fraction >= 0.5, poor/unreliable segmentation fraction >= 0.5, clipped grain-pixel fraction >= 0.5, or grain-region illumination uniformity < 0.15.
- Mean segmentation confidence and per-grain `segmentation_quality` now affect the hard-failure decision. Image megapixels still do not reject an image.

The code/configuration does not document empirical calibration or human-rated labels used to choose these values. They are **engineering thresholds requiring calibration**, not scientific criteria. UNRELIABLE is now reachable via explicit hard failures rather than the prior unreachable aggregate-score branch. A regression verifies that the same grain has equal blur, illumination and clipping metrics on black and white backgrounds. Quality tiers are not calibrated probabilities.

## Standards and Sample Size

Thirty grains is the **project screening threshold**, an engineering setting in `configs/thresholds.json`; it is not asserted to be a Government of India minimum sample requirement. For one or two grains, individual predictions and observed image fractions may be shown, but official lot-level compliance remains **Not determinable from this sample**. Image counts/areas do not reproduce official weight-based measurements.

## Regression Evidence

- `tests/test_case3_regressions.py`: clean one-grain and two-grain paths, chalky background invariance/abstention, experimental sprouted output, class-map behavior, small-sample compliance suppression, and training-data guards.
- `tests/test_rice_gate.py`: Case 1 foreign-matter-only input remains NOT RICE with no grain records, and Case 2 rice inputs remain accepted. The gate and segmentation implementation were not edited for this audit.
