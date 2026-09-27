# Dataset Documentation — Rice Quality AI

## Overview

This project builds a rice grain instance segmentation model using multiple datasets, both public and internally curated. Datasets are chosen based on: (1) availability of pixel-level segmentation masks for instance-level learning, (2) coverage of realistic rice quality scenarios (density, touching, overlapping, foreign matter), (3) permissive or academic licensing, and (4) compatibility with COCO-format annotation pipelines.

All datasets flow through a standardized pipeline: `audit → convert to COCO → generate synthetic → create stratified splits → train`.

---

## Datasets Table

| Dataset name | Location | Format | Images (EST) | Annotations | Purpose | Status |
|---|---|---|---:|---|---|---|
| GrainSet Rice v3 | `data/raw/grainset_rice_v3/rice (1)/rice/` | PNG images + paired mask PNGs | ~5,000 (pre-audit placeholder) | Per-image mask (instance or semantic — UNKNOWN until audit) | Primary training data for segmentation | **Needs Audit** |
| GrainDet Rice v2 | `data/raw/graindet_rice_v2/rice/` | 8 class folders, JPG/PNG singles | ~15,000 (pre-audit placeholder) | Image-level classification labels only (8 classes) | Synthetic grain source (0_NOR), FM reference (1_F&S) | **Needs Audit** |
| 27 Hand-Curated Test Set | `a:\grain\27 tests\` | PNG/JPG mixed | 27 | Manual scenarios (no pixel GT yet — GT annotations TBD) | Final held-out qualitative + quantitative evaluation | **Ready (GT pending)** |
| Synthetic Dataset | `datasets/processed/coco/images/synthetic/` + `annotations/instances_synthetic.json` | PNG + COCO JSON | 1,900 (target) | COCO instance polygons (rice_grain + foreign_matter) | Data augmentation for touching/overlap/dense/FM edge cases | **To be generated** |
| Roboflow candidates | TBD — Roboflow Universe search terms: `rice grain segmentation`, `foreign matter rice` | N/A | N/A candidates | Instance segmentation | Backup/extended training if GrainSet v3 is insufficient | **Not Downloaded** |
| Murat Koklu dataset (Kaggle) | TBD — Kaggle: `muratkokludataset/rice-image-dataset` | Classified single-grain images | ~75,000 across 5 varieties | Image-level class labels only | Synthetic single-grain library expansion (variety diversity) | **Not Downloaded** |

---

## GrainSet v3 Deep Dive

### Expected Structure

```
data/raw/grainset_rice_v3/rice (1)/rice/
├── train/      # RGB images (PNG/JPG) — primary training inputs
├── mask/       # Paired mask images (same filenames as train/)
└── test/       # Optional test images (may or may not have masks)
```

### Mask Format (UNKNOWN until audit run)

Each mask file is examined per-pixel by `scripts/audit_dataset.py`:

- **Instance encoding**: Each grain is drawn with a unique non-zero, non-255 pixel value. Conversion to COCO polygons is direct (one polygon per unique ID).
- **Semantic encoding**: All grains are drawn with value 255 (binary). Conversion requires `cv2.connectedComponentsWithStats` to separate touching grains, then one polygon per connected component (area ≥ 50 px²).
- The audit script votes across the first N masks and reports a dominant type in `datasets/inspection/audit_report.json` under `mask_type`.

### Statistics Placeholders (populated by `audit_dataset.py`)

| Metric | Value | Source |
|---|---|---|
| Total train images found | TBD | `audit_report.total_train_images` |
| Valid image/mask pairs | TBD | `audit_report.summary.total_pairs_analyzed` |
| Dominant mask type | TBD (INSTANCE / SEMANTIC) | `audit_report.summary.mask_type_verdict` |
| Mask type votes | instance=?, semantic=?, empty=?, corrupt=? | `audit_report.mask_type_votes` |
| Grain count min / max / mean / median | TBD | `audit_report.summary.grain_count_stats` |
| Grain count distribution (buckets of 5) | TBD | `audit_report.grain_count_distribution` |
| Images with ≥1 touching pair | TBD / fraction | `audit_report.touching_grain_count`, `touching_image_fraction` |
| Corrupt / empty / missing / duplicate | TBD / TBD / TBD / TBD | `summary` fields |
| Recommendation | TBD | `summary.recommendation` |

---

## GrainDet v2 Deep Dive

### 8 Class Folders

```
data/raw/graindet_rice_v2/rice/
├── 0_NOR/          # Normal whole grains — primary synthetic source
├── 1_F&S/          # Foreign & stones — the *only* public FM reference we have
├── 2_BRO/          # Broken
├── 3_CHK/          # Chalky / opaque
├── 4_DIS/          # Discoloured
├── 5_IMM/          # Immature / shrunken
├── 6_SPR/          # Sprouted
└── 7_WEE/          # Weevilled / insect-damaged
```

### Key Notes

- **0_NOR as synthetic source**: All images in `0_NOR/` are single clean grains. `scripts/generate_synthetic.py` extracts each grain via Otsu thresholding on the HSV-V channel and composites them onto textured/blank backgrounds at controlled densities.
- **1_F&S classification only**: This folder provides image-level labels for foreign matter and stones, but **does NOT contain instance-level segmentation annotations for the foreign objects** inside each cropped thumbnail. Therefore it is useful only as a visual reference and for classifier training, NOT as an FM segmentation source.
- Class counts: TBD — populated by `audit_graindet_v2()` → `datasets/inspection/graindet_v2_audit.json`.
- Class name mapping (used downstream for quality sub-classifiers): `0_NOR`=Normal, `1_F&S`=Foreign/Stone, `2_BRO`=Broken, `3_CHK`=Chalky, `4_DIS`=Discoloured, `5_IMM`=Immature, `6_SPR`=Sprouted, `7_WEE`=Weevilled.

---

## 27 Hand-Curated Test Scenarios

Stored at `a:\grain\27 tests\`. These files are also copied into `tests/datasets/{category}/` subdirectories by `scripts/create_splits.py`.

| # | Filename | Scenario Description |
|---:|---|---|
| 1 | `1 stones.png` | Foreign matter — stones visible in a rice sample |
| 2 | `2 No rice - wheat grains.jpg` | No-rice control: non-rice grains only (wheat admixture) |
| 3 | `3 One clean rice grain.png` | Single isolated normal rice grain |
| 4 | `4 Two clean rice grains.png` | Two isolated normal grains separated |
| 5 | `5 Five clean rice grains.png` | Small sparse group, no touching |
| 6 | `6 Many rice grains.png` | Medium-density clean sample (~50 grains area) |
| 7 | `7 Very large sample — ~1000 grains.png` | Extremely dense, ~1000 grains — stress-test counting |
| 8 | `8. Broken rice grain.png` | Broken fragments (fragments < full length) |
| 9 | `9. Chalky rice grain.png` | Opaque / chalky centre grains |
| 10 | `10. Damaged rice grain.png` | Physically cracked / split grains |
| 11 | `11. Red rice.png` | Red-rice variant colour discoloration |
| 12 | `12. Discoloured rice.png` | Yellow / stained grains |
| 13 | `13. Dehusked  partially husked grain.png` | Remaining husk fragments, partial dehulling |
| 14 | `14. Immature  shrunken grain.png` | Small, thin, under-developed grains |
| 15 | `15. Sprouted rice.png` | Germinated / sprouted grain protrusions |
| 16 | `16. Weevilled  insect-damaged rice.png` | Hole / tunnel damage from insects |
| 17 | `17. Foreign matter — stone.png` | Stone fragment + rice |
| 18 | `18. Foreign matter — plastic.png` | Plastic fragment + rice |
| 19 | `19. Foreign matter — other seed.png` | Non-rice seed admixture + rice |
| 20 | `20. Mixed sample — clean + broken + chalky.png` | Multi-defect moderate density sample |
| 21 | `21. Mixed defective sample.png` | Multiple defect types combined |
| 22 | `22. Rice + foreign matter + defects.png` | Rice + FM + grain defects combined |
| 23 | `23. Two grains touching.png` | Minimal touching-pair case |
| 24 | `24. Overlapping grains.png` | Grains with visible pixel overlap (occlusion) |
| 25 | `25. Completely dense rice sample.png` | Dense packing, many touching pairs |
| 26 | `26. Admixture of lower class — sample-level case.png` | Lower-class rice admixture |
| 27 | `27. Mixed everything stress-test image.png` | All cases combined: dense + touching + overlap + defects + FM |

---

## Synthetic Dataset Strategy

### Why Synthetic Is Needed

GrainSet v3 alone is unlikely to adequately represent:
1. Explicit **touching grain pairs** with controlled inter-grain distance
2. Explicit **overlapping / occluded grain** pairs with controlled overlap ratio
3. Very **dense packing** (>100 grains per 640×640 crop)
4. Controlled **foreign matter** instances with precise per-object segmentation GT
5. Balanced density stratification

### Generation Modes & Target Quantities

| Mode | Grain Range Per Image | Touch Pairs | Overlap Pairs | Target Images | Total Grains EST |
|---|---:|---:|---:|---:|---:|
| sparse (isolated) | 10–20 | 0 | 0 | 500 | ~7,500 |
| touching | 4–10 | 2–5 | 0 | 500 | ~4,500 |
| overlapping | 4–10 | 0 | 2–3 | 500 | ~4,500 |
| dense | 50–100 | 0 | 0 | 200 | ~15,000 |
| very_dense | 100–250 | 0 | 0 | 100 | ~17,500 |
| foreign_matter | 10–40 rice + ~25% FM | 0 | 0 | 100 | ~2,500 + FM |
| **TOTAL** | | | | **1,900** | **~51,500 instances EST** |

### Generation Pipeline

1. **Load source grains** from `0_NOR/` folder of GrainDet v2 via Otsu threshold (HSV-V) to extract per-grain mask + tightly cropped RGB.
2. **Synthetic fallback shapes**: If <10 usable real grains exist, generate rotated ellipses (aspect 2.8–3.8 : 1) with rice-colour gradient + per-pixel noise.
3. **Background canvas**: 55% paper-textured + 45% plain, Gaussian vignette, size 640×640.
4. **Per-grain augmentation**: 0–360° rotation, ±20% scale, ±30% brightness, ±20% contrast, σ=0–1.0 Gaussian blur (within-mask only).
5. **Packing + overlap rules**: Max overlap ratio by mode — sparse ≤5%, touching ≤15%, overlapping ≤55%, dense/foreign ≤35%, very_dense ≤55%. Placement retry up to (grains × 80) attempts.
6. **Foreign matter**: 4 synthetic types (stone polygon, straw ellipse, husk splinter, dark speck), each with natural colour noise. Placed in 25% proportion to rice count in `foreign` mode.
7. **COCO output**: Mask → contour → Douglas-Peucker polygon (ε=1.5 px simplification) → `instances_synthetic.json`.

---

## Annotation Format

All training/evaluation uses **COCO Instance Segmentation JSON** (standard 2017 schema). The canonical structure follows:

```jsonc
{
  "info":          { "description": "...", "version": "1.0", "year": 2026 },
  "licenses":      [],
  "categories":    [ /* see COCO Categories table below */ ],
  "images": [
    { "id": 1, "file_name": "img_0001.png", "width": 640, "height": 640 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1,y1, x2,y2, ...]],  // one polygon per instance
      "area": 1234,       // pixel area inside mask
      "bbox": [x, y, w, h],
      "iscrowd": 0
    }
  ]
}
```

- `segmentation` is a flat list of alternating x,y coordinates (relative to image origin, pixel units). One polygon per annotation entry (multi-part polygons split across multiple entries with same `iscrowd=0` is acceptable but not generated by our scripts).
- Min grain area filter: **50 px²** (applied during conversion and synthetic generation).
- Polygon simplification: Douglas-Peucker ε = 2.0 px (convert) / 1.5 px (synthetic).

## COCO Categories

| id | name | supercategory | Notes |
|---:|---|---|---|
| 1 | `rice_grain` | `rice` | All rice varieties and quality grades (normal, broken, chalky, discoloured, immature, sprouted, weevilled, damaged). Defect *type* is determined by post-segmentation classifiers, not the segmentation model itself. |
| 2 | `foreign_matter` | `other` | Stones, straw, husk, plastic, other seeds, dark specks — any non-rice discrete object. |

---

## How to Run

All scripts are invoked from the project root (`a:\grain\rice-quality-ai\`).

### 1. Audit Raw Datasets

```bash
# Full audit of GrainSet v3 + GrainDet v2, 20 preview images
python scripts/audit_dataset.py --dataset all --preview-count 20

# Quick audit on a 500-image sample (fast)
python scripts/audit_dataset.py --dataset grainset_v3 --sample 500 --seed 42

# GrainDet v2 only (classification folder count summary)
python scripts/audit_dataset.py --dataset graindet_v2
```

Outputs:
- `datasets/inspection/audit_report.json` — full GrainSet v3 report
- `datasets/inspection/graindet_v2_audit.json` — GrainDet v2 class counts
- `datasets/inspection/grainset_v3_previews/preview_XXXX_*.jpg` — 4-panel previews

### 2. Convert GrainSet v3 → COCO

```bash
# Auto-detect mask type from audit report; 70/15/15 auto-split
python scripts/convert_to_coco.py

# Force mask type (use after audit confirms the type)
python scripts/convert_to_coco.py --mask-type instance
python scripts/convert_to_coco.py --mask-type semantic

# Process a single split from an externally-generated split list
python scripts/convert_to_coco.py --split-file datasets/splits/val.txt --split val --no-copy-images

# Test the pipeline on a small subset
python scripts/convert_to_coco.py --max-images 200
```

Outputs:
- `datasets/processed/coco/annotations/instances_{train,val,test}.json`
- `datasets/processed/coco/images/{train,val,test}/` (images copied or symlinked)
- `datasets/splits/{train,val,test}.txt` (filename lists)

### 3. Generate Synthetic Dataset

```bash
# Default 1,900-image generation (500+500+500+200+100+100), seed 42
python scripts/generate_synthetic.py

# Custom counts (e.g. smaller dev set)
python scripts/generate_synthetic.py \
  --num-isolated 100 --num-touching 100 --num-overlapping 100 \
  --num-dense 50 --num-very-dense 20 --num-foreign 20 \
  --seed 123 --img-size 640
```

Outputs:
- `datasets/processed/coco/images/synthetic/synthetic_<mode>_XXXXXX.png`
- `datasets/processed/coco/annotations/instances_synthetic.json`

### 4. Create Stratified Splits (Combined Real + Synthetic)

```bash
# Default 70/15/15, stratified by (density, real/synthetic), seed=42, copy images
python scripts/create_splits.py

# Custom ratios
python scripts/create_splits.py --train-ratio 0.75 --val-ratio 0.125 --test-ratio 0.125

# Force re-split (ignore existing split .txt files)
python scripts/create_splits.py --force-resplit --seed 42

# Symlink instead of copy (saves disk; Windows will fall back to copy on failure)
python scripts/create_splits.py --no-copy-images
```

Outputs:
- `datasets/splits/{train,val,test}.txt` + `split_stats.json`
- `datasets/processed/coco/annotations/instances_{train,val,test}.json` (filtered per split, contiguous re-mapped IDs)
- `tests/datasets/{touching,overlapping,dense,foreign_matter,no_rice,single_grain,uncategorized}/` (from 27 hand-curated set)

---

## Known Issues & Limitations

| # | Issue | Impact | Mitigation |
|---|---|---|---|
| 1 | **GrainSet v3 license ambiguity** | Public download URL exists, but no explicit LICENSE file accompanies the dataset. Redistributing derived annotations may be risky. | Use only for internal training; cite the dataset webpage in any writeup; do NOT redistribute the raw images or derived COCO JSONs externally without license review. |
| 2 | **Mask type unknown until audit** | Conversion pipeline behaves differently for instance vs semantic masks (connected components step or not). | Always run `audit_dataset.py` first. `convert_to_coco.py --mask-type auto` reads the audit result before proceeding. |
| 3 | **No foreign_matter annotations in GrainSet v3** | Primary training data contains only rice grains. A segmentation model trained on GrainSet alone will never output FM category=2 predictions. | Synthetic dataset (6 modes, 100 FM images) + manual 27-test FM cases (17,18,19,22,1 stones). After Run 1 evaluation, add human-annotated FM patches. |
| 4 | **GrainDet v2 1_F&S is classification, not instance** | Thumbnail crops of FM objects have image-level labels but no pixel-level FM segmentation. Cannot be used for segmentation GT directly. | Use only as synthetic FM texture reference + classifier training data. |
| 5 | **27 hand-curated set has no pixel GT yet** | Cannot compute quantitative per-instance metrics on the most interesting stress cases. | Manually annotate using CVAT/LabelMe (COCO export) as a post-Run-1 step; auto-populate GT JSON sidecar files under `tests/datasets/*/`. |
| 6 | **Synthetic domain gap** | Rendered composites do not perfectly match real rice lighting / shadows / background textures. | Heavy real-image augmentation pipeline (mosaic, mixup, copy-paste, HSV jitter) in YOLO training + domain randomisation in synthetic generator. |

---

## Data Quality Checklist (Run Before Every Training Run)

Every dataset combined into the training split must pass these checks:

- [ ] **No corruption**: 0 mask files return `None` on `cv2.imread` (audit `corrupt_files` list empty)
- [ ] **No blank masks**: 0 masks with 0 foreground pixels (audit `empty_masks` list empty)
- [ ] **No missing masks**: Every `train/*.png` has a corresponding `mask/*.png` (or alternate extension)
- [ ] **No duplicates**: Perceptual hash (aHash, 8×8) uniqueness check (audit `duplicate_pairs` list empty — or duplicates manually removed)
- [ ] **Mask consistency**: When instance, unique IDs are contiguous per-image and non-overlapping in pixel space; when semantic, connected-components count matches visual count on preview grids
- [ ] **Valid polygons**: All COCO `segmentation` entries have ≥6 points / ≥3 vertices after simplification; `bbox` encloses all polygon points; `area` matches `np.sum(mask)` within 1%
- [ ] **Stratification preserved**: `split_stats.json` shows density buckets and real/synthetic ratios are balanced (no bucket missing in val or test)
- [ ] **No cross-split leakage**: No filename appears in more than one `train.txt` / `val.txt` / `test.txt`; synthetic variants of the same source grain are grouped into one split by source hash

---

## License / Citation Placeholders

### GrainSet Rice v3
- **Source URL**: TBD (populate after confirming download origin)
- **License**: *UNKNOWN — pending audit* (assume non-commercial / academic-only until verified)
- **Citation format**: `@dataset{grainset_rice_v3, title={GrainSet Rice v3 — Instance Segmentation Dataset}, year={TBD}, author={TBD}}`

### GrainDet Rice v2
- **Source URL**: TBD
- **License**: *UNKNOWN — pending audit*
- **Citation format**: `@dataset{graindet_rice_v2, title={GrainDet Rice v2 — 8-Class Rice Quality Classification}, year={TBD}, author={TBD}}`

### Murat Koklu Rice Image Dataset
- **Source**: Kaggle `muratkokludataset/rice-image-dataset`
- **Citation**: Köklü, M., Ozkan, I. A., & Sarigil, S. (2021). *A Public and Effective Rice Image Dataset for Variety Classification and Quality Determination*.
- **License**: CC0 / Public domain (as published on Kaggle — verify at download time)

### Synthetic Dataset (generated by this project)
- **License**: Proprietary / internal use only (same as the project)
- **Seed determinism**: All outputs reproducible for given `--seed` and source grain folder contents.
