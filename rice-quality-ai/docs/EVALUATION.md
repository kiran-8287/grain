# Evaluation Documentation — Rice Quality AI

## Evaluation Philosophy

**Honesty over headline numbers.** This project separates all reported metrics into two clearly-marked categories:

1. **Measured metrics** — produced by `scripts/evaluate_model.py` from real predictions on real held-out data. These are the only numbers that count. They are reproducible.
2. **Pre-training literature ESTIMATES** — sourced from comparable rice-segmentation papers and similar model architectures on similar tasks. These are explicitly labeled `PRE-TRAINING LITERATURE ESTIMATES, NOT MEASURED`. They are planning targets only, not claims.

Any report, graph, or table that mixes measured and estimated values *without clear labeling* is considered incorrect. When in doubt, leave a cell blank and mark it "TBD — pending Run N evaluation."

Additional principles:
- **Same test set, same metrics, same hardware** for every model being compared (YOLO vs Mask R-CNN vs Classical CV).
- **Predictions are matched to ground truth once** per IoU threshold; no double-counting.
- **Failure images are generated for *every* evaluated image** when `--generate-failure-images` is on — not just bad ones.
- **Reproducibility**: `evaluate_model.py` is deterministic given the same weights, same COCO JSON, same test images, same `--conf` threshold.

---

## 9 Test Categories A–I

Every test image is assigned to exactly one category by `categorize_image()` in `evaluate_model.py:104` based on its *ground-truth properties* (not model predictions). The category is stored in `per_image_results.json` under the `"category"` key and used for per-category aggregation.

| Code | Name | Definition | Examples from 27-Test Set |
|---|---|---|---|
| **A** | `single_isolated` | gt_count == 1; no touching; no overlap; no FM | `3 One clean rice grain.png` |
| **B** | `few_separated` | 2 ≤ gt_count ≤ 10; no touching; no overlap; no FM | `4 Two clean rice grains.png`, `5 Five clean rice grains.png` |
| **C** | `touching` | ≥1 GT touching pair (dilated mask intersection); no overlap; no FM | `23. Two grains touching.png` |
| **D** | `overlapping` | ≥1 GT overlap pair (mask IoU ≥ 0.15); no FM | `24. Overlapping grains.png` |
| **E** | `dense` | gt_count ≥ 50; no touching/overlap flags sufficient to bump to C/D; no FM | `6 Many rice grains.png`, `7 Very large sample — ~1000 grains.png`, `25. Completely dense rice sample.png` |
| **F** | `rice_plus_foreign` | gt_count (rice) > 0 AND ≥1 foreign_matter GT instance present | `1 stones.png`, `17. Foreign matter — stone.png`, `18. Foreign matter — plastic.png`, `19. Foreign matter — other seed.png`, `22. Rice + foreign matter + defects.png` |
| **G** | `no_rice` | gt_count == 0 AND 0 foreign_matter instances (pure background / non-rice only) | `2 No rice - wheat grains.jpg` |
| **H** | `foreign_only` | gt_count == 0 AND ≥1 foreign_matter instances present | (no direct file in 27-set; added via synthetic foreign-only mode) |
| **I** | `mixed_difficult` | Does not cleanly fit A–H. Fallback: (a) defects mixed + density combined, or (b) gt_count 11–49 non-touching, or (c) multiple overlapping flags | `8. Broken…`–`16. Weevilled…` (single-defect close-ups without FM GT bucket here), `20. Mixed sample — clean + broken + chalky.png`, `21. Mixed defective sample.png`, `26. Admixture of lower class.png`, `27. Mixed everything stress-test image.png` |

Category mapping is *exclusive* — `categorize_image()` checks in a strict priority order: H/G first (no rice), then F (rice+FM), then D (overlap), then C (touching), then E (dense ≥50), then A (exactly 1), then B (2–10), else I.

---

## Metric Definitions

All metrics are computed per-image first, then aggregated (mean or sum) per category and overall.

| Metric | Short Name | Range | Definition |
|---|---|---|---|
| Box mAP@50 | `ap50_box` | 0.0–1.0 | Average recall across 10 IoU thresholds? No — **box AP @ IoU=0.50 only** (using greedy match on box IoU ≥ 0.50). In evaluate_model.py this is approximated as `recall@TP/FP/FN match` per image then averaged. |
| Box mAP@50:95 | `ap50_95_box` | 0.0–1.0 | Mean recall across 10 equally-spaced IoU thresholds: 0.50, 0.55, …, 0.95. Box matching. Approximation (not full 11-point PR interpolation) when `pycocotools` is unavailable. |
| Mask AP@50 | `ap50_mask` | 0.0–1.0 | Same as AP@50_box but using **mask IoU** for the TP/FP/FN match instead of box IoU. The primary accuracy metric for segmentation quality. |
| Mask AP@50:95 | `ap50_95_mask` | 0.0–1.0 | Mean mask recall across IoU 0.50–0.95, 10 thresholds. The strictest segmentation-quality metric. |
| Count Error % | `count_error_pct` | 0.0–100+% | `100 * abs(pred_count - gt_count) / max(gt_count, 1)`. Measures the rice-counter usability. Lower is better. |
| Missed Grain Rate | `missed_rate` | 0.0–1.0+ | `FN / max(gt_count, 1)` — fraction of GT rice grains not matched by any prediction at IoU≥0.50. |
| Merged Grain Rate | `merged_rate` | 0.0–1.0+ | `count(pred masks that overlap ≥2 GT masks at IoU≥0.30) / max(gt_count, 1)`. Measures oversegmentation failure where the model lumps multiple grains into one polygon. |
| Split Grain Rate | `split_rate` | 0.0–1.0+ | `count(GT masks that overlap ≥2 pred masks at IoU≥0.30) / max(gt_count, 1)`. Measures the model incorrectly splitting one real grain into multiple predictions. |
| Duplicate Rate | `duplicate_rate` | 0.0–1.0+ | `(total matched preds − total unique matched GTs) / max(gt_count, 1)`. Extra predictions that matched the same GT (post-NMS duplicates). |
| False Foreign Rate | `false_foreign_rate` | 0.0–1.0+ | `count(pred-FM whose mask IoU ≥0.5 any GT rice mask) / max(gt_count, 1)`. Rice grains that the model *incorrectly labeled as foreign matter* — the most dangerous false positive for quality standards. |
| Inference Time | `inference_ms` | ≥0 ms | Wall-clock milliseconds for a single image's forward pass (preprocess + model predict, *excludes* post-processing). Measured with `time.perf_counter()`; hardware-dependent. Reported as mean across evaluated images. |

All failure types + counts (MISSED_GRAIN … LOW_CONFIDENCE) are recorded per-image in `per_image_results.json[].failures` and aggregated in `eval_report.json.overall.failure_counts`.

---

## How to Run

Run from project root `a:\grain\rice-quality-ai\`.

### Basic Full Evaluation (YOLO primary)

```bash
# Auto-detect weights path from models/yolo_seg/run1_baseline/weights/best.pt; test split
python scripts/evaluate_model.py \
  --model-type yolo \
  --split test \
  --conf 0.25 \
  --generate-failure-images \
  --failures-only
```

### Explicit Weights / Other Models

```bash
# Specific YOLO checkpoint
python scripts/evaluate_model.py \
  --model-type yolo \
  --weights models/yolo_seg/run2_synth_finetune/weights/best.pt \
  --generate-failure-images

# Mask R-CNN comparison baseline
python scripts/evaluate_model.py \
  --model-type maskrcnn \
  --weights models/maskrcnn/best/model.pth \
  --max-images 100

# Classical CV watershed baseline (no ML model — pure OpenCV)
python scripts/evaluate_model.py \
  --model-type classical \
  --generate-failure-images
```

### Auto Mode

```bash
# Picks: YOLO if best.pt exists → else MaskRCNN if model.pth → else Classical
python scripts/evaluate_model.py --model-type auto
```

### Useful Flags

| Flag | Default | Purpose |
|---|---|---|
| `--model-type` | `auto` | `auto \| yolo \| maskrcnn \| classical` |
| `--weights` | `None` | Override path to `.pt` (YOLO) or `.pth` (MaskRCNN) |
| `--split` | `test` | COCO subdirectory for image lookup (`images/test/`) |
| `--max-images` | `None` | Cap images evaluated from the COCO split (useful for smoke tests) |
| `--max-images-per-category` | `None` | Cap per-category from `tests/datasets/*/` folders |
| `--conf` | `0.25` | Confidence threshold for raw predictions before post-processing |
| `--generate-failure-images` | off | Write 4-panel diagnostic PNGs to `results/evaluation/failures/<cat>/` |
| `--failures-only` | off | With `--generate-failure-images`, skip images that have zero FP/FN/count error (save disk space on large evals) |
| `--device` | `""` (auto) | `""`, `"cpu"`, `"cuda:0"` for YOLO/MaskRCNN device override |
| `--coco-ann` | `datasets/processed/coco/annotations/instances_test.json` | Override COCO GT JSON path |
| `--coco-img-dir` | `datasets/processed/coco/images` | Parent directory containing `{train,val,test}/` subdirs |
| `--tests-datasets-dir` | `tests/datasets` | Structured test folders (A–I subdirs or named subdirs) |
| `--output-dir` | `results/evaluation` | Where eval_report.json, summary.txt, failures/ are written |

Evaluation records come from two sources concatenated:
1. `load_coco_split()` — the held-out COCO test split, matched by `instances_{split}.json`.
2. `load_tests_datasets()` — `tests/datasets/{A,B,…,I}/` or `tests/datasets/{single_isolated,touching,…}/` named subdirectories. Sidecar `.json` files with annotations are loaded as GT if present.

---

## Evaluation Pipeline Diagram (ASCII)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    RICE SEGMENTATION EVALUATION PIPELINE                   │
│                                                                            │
│  ┌──────────────────┐                                                       │
│  │ Load COCO GT     │  instances_test.json + test/ images                  │
│  │ (per_image GT)   │  ──► {image_id, file_name, w, h, gt_anns[], category} │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌────────────────────────────┐      ┌──────────────────────────────┐      │
│  │ For each image:            │      │ Load predictor:              │      │
│  │  1. Read BGR image         │◄────►│  YOLO / MaskRCNN / Classical │      │
│  │  2. Run model.predict()    │      └──────────────────────────────┘      │
│  │  3. Record inference_ms    │                                              │
│  └────────┬───────────────────┘                                              │
│           │  pred_anns[]: {bbox, mask, conf, cat_id}                        │
│           ▼                                                                 │
│  ┌────────────────────────────┐   IoU thresholds: 0.50, 0.55 … 0.95         │
│  │ Match Predictions → GT     │   Greedy highest-IoU first.                 │
│  │  (Box IoU + Mask IoU)      │   Per pred: matched GT index or None (FP).  │
│  └────────┬───────────────────┘   Per GT:  matched pred or None (FN).       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌────────────────────────────┐                                             │
│  │ Compute Per-Image Metrics  │  AP50_box, AP50:95_box,                     │
│  │  + Failure Types           │  AP50_mask, AP50:95_mask,                    │
│  │  (8 failure classes)       │  count_error_pct, rates,                    │
│  │                           │  MISSED_GRAIN / MERGED / SPLIT / etc.        │
│  └────────┬───────────────────┘                                             │
│           │                                                                 │
│           ▼                                                                 │
│  ┌────────────────────────────┐   Group per_image_results by "category"     │
│  │ Aggregate by Category A–I  │   (A … I) → means, sums, counts.            │
│  │  + Overall Aggregate       │   Compute failure rates / GT-rice counts.   │
│  └────────┬───────────────────┘                                             │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ Outputs:                                                            │    │
│  │  1. eval_report.json    (overall + per_category structured data)    │    │
│  │  2. per_image_results.json (all per-image raw metrics)              │    │
│  │  3. summary.txt         (ASCII human-readable table)                │    │
│  │  4. failures/<cat>/*.png (4-panel: Input | GT | Pred | Diff)        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Training ESTIMATES (Literature Targets)

> **⚠️ PRE-TRAINING LITERATURE ESTIMATES, NOT MEASURED.**
> These ranges are compiled from comparable rice-segmentation papers (YOLOv8-seg / Mask R-CNN on rice datasets of similar scale, instance-annotated) and from the authors' experience with comparable tasks. They are planning targets only. **Do not cite these as project results.** They are useful for sanity-checking a Run before you commit to 10 hours of GPU training — if a measured number is far outside these ranges, something is wrong (too high: data leakage; too low: bug in COCO loader / class mapping).

### Expected Single-Model Performance Ranges (Before Iterative Training)

| Category | Scenario | Box mAP@50 EST | Mask mAP@50 EST | Count Error % EST |
|---|---|---:|---:|---:|
| A | Single isolated | 0.90 – 0.97 | 0.85 – 0.95 | < 1% |
| B | Few separated | 0.85 – 0.95 | 0.80 – 0.92 | < 5% |
| C | Touching pairs | 0.65 – 0.82 | 0.55 – 0.75 | 10 – 25% |
| D | Overlapping | 0.55 – 0.75 | 0.45 – 0.65 | 15 – 35% |
| E | Dense ≥50 | 0.60 – 0.80 | 0.50 – 0.70 | 5 – 20% |
| F | Rice + FM | 0.55 – 0.75 (rice) | 0.45 – 0.65 (rice) | 10 – 25% |
| G | No rice (negatives) | N/A (count-based) | N/A | FP rice count ≤ 2 |
| H | FM only | N/A | N/A | FM FN rate ≤ 20% |
| I | Mixed difficult | 0.50 – 0.70 | 0.40 – 0.60 | 15 – 35% |
| **OVERALL** | Weighted mean | **0.70 – 0.85** | **0.60 – 0.78** | **8 – 20%** |

### After Iterative Training Targets (Round 3 complete — aspirational ESTIMATES)

| Metric | Target (after 3 rounds) |
|---|---:|
| Overall box mAP@50 | ≥ 0.88 |
| Overall mask mAP@50 | ≥ 0.80 |
| Mean count error % overall | ≤ 8% |
| Mask mAP@50 on worst category (C or D) | ≥ 0.70 |
| False Foreign Rate on clean rice (A, B, E) | ≤ 0.5% |
| Missed Grain Rate on dense (E) | ≤ 10% |
| Inference ms / image, CPU (modern laptop, 640²) | ≤ 800 ms (YOLOv8l-seg) |
| Inference ms / image, 8 GB GPU (640²) | ≤ 40 ms (YOLOv8l-seg) |

---

## Model Comparison Protocol

For a head-to-head comparison of YOLO vs Mask R-CNN vs Classical CV:

1. **Fix the test data.** One COCO test split JSON + one `tests/datasets/` folder. Do not re-split between runs.
2. **Fix the confidence threshold.** Use `--conf 0.25` for YOLO, `--conf 0.50` for Mask R-CNN, classical uses its internal defaults. Report the thresholds used.
3. **Fix the device.** Run all three models on the same machine (e.g. a single RTX 3070 laptop). Report GPU vs CPU inference ms separately.
4. **Run `evaluate_model.py` three times** with only `--model-type` and `--weights` changed:
   ```bash
   python scripts/evaluate_model.py --model-type yolo      --weights $YOLO_PT  --generate-failure-images --output-dir results/eval_yolo
   python scripts/evaluate_model.py --model-type maskrcnn  --weights $MRCNN_PTH --generate-failure-images --output-dir results/eval_maskrcnn
   python scripts/evaluate_model.py --model-type classical --generate-failure-images --output-dir results/eval_classical
   ```
5. **Compare columns in `summary.txt` side-by-side.** The 3 key columns to look at: `AP50_m` (mask), `CntE%` (count error), `ms` (inference time). A table like:
   | Model | AP50_mask | Count Err % | Inference ms (GPU) | Verdict |
   |---|---:|---:|---:|---|
   | YOLOv8l-seg | ? | ? | ? | |
   | Mask R-CNN R50 | ? | ? | ? | |
   | Classical CV | ? | ? | ? | |
6. **Hardware note**: Always prefix reported `ms` values with the hardware, e.g. "GPU: RTX 3070 laptop, CPU: i7-12700H". Inference ms are not meaningful without it.

---

## Failure Images Output

Generated when `--generate-failure-images` is passed. Stored under:

```
results/evaluation/failures/
├── A/                          # One subdirectory per category code A–I
│   ├── 3 One clean rice grain.png
│   └── …
├── B/
├── …
└── I/
    └── 27. Mixed everything stress-test image.png
```

Filename convention: exactly the original image filename, extension normalised to `.png` (PNG is lossless and preserves the diagnostic panels).

### 4-Panel Format

```
┌───────────────────────┬───────────────────────────┐
│ 1. Input              │ 2. GT (green overlay)     │
│    (raw RGB image)    │    + green contour lines  │
├───────────────────────┼───────────────────────────┤
│ 3. Prediction         │ 4. Diff:                  │
│    (orange overlay)   │    GREEN  = GT only (miss)│
│    + orange contours  │    RED    = Pred only (FP)│
│                       │    WHITE  = Both (agree)  │
└───────────────────────┴───────────────────────────┘
```

Each panel has a 28 px black title bar with white text legend.

Visual reading rules for the Diff panel:
- Green blobs → model **missed** a grain (FN, MISSED_GRAIN).
- Red blobs → model **hallucinated** a grain or FM (FP, FALSE_RICE / FALSE_FM).
- White blobs → agreement (TP).
- Pattern: mostly white → good. Green edges around touching grains → merged-rate failures. Red duplicate red blobs near each other → duplicate/SPLIT failures.

---

## How to Read the Report

### `eval_report.json` Structure

```json
{
  "evaluation_timestamp": "2026-09-28T12:00:00+0000",
  "model_type": "yolo",
  "weights_path": "models/yolo_seg/run1_baseline/weights/best.pt",
  "split": "test",
  "pycocotools_used": true,
  "yolo_available": true,
  "maskrcnn_available": true,

  "overall": {
    "num_images": 123,
    "total_gt_instances": 4567,
    "total_gt_rice": 4400,
    "total_pred_instances": 4500,
    "mean_ap50_box": 0.82,
    "mean_ap50_95_box": 0.58,
    "mean_ap50_mask": 0.73,
    "mean_ap50_95_mask": 0.51,
    "mean_count_error_pct": 10.4,
    "mean_missed_rate": 0.08,
    "mean_merged_rate": 0.06,
    "mean_split_rate": 0.03,
    "mean_duplicate_rate": 0.01,
    "mean_false_foreign_rate": 0.005,
    "mean_inference_ms": 32.7,
    "failure_counts": {
      "MISSED_GRAIN": 250, "MERGED_GRAINS": 120, "SPLIT_GRAIN": 60,
      "FALSE_RICE": 40, "FALSE_FOREIGN_MATTER": 10, "DUPLICATE": 25,
      "BAD_MASK": 55, "LOW_CONFIDENCE": 90
    },
    "failure_rates_total_gt": { /* each failure / total_gt_rice */ },
    "num_categories_with_data": 9
  },

  "per_category": {
    "A": {
      "category_name": "single_isolated",
      "num_images": 5,
      "total_gt_rice": 5,
      "mean_ap50_box": 0.97,
      "...": "...",
      "failure_counts": { "...": "..." },
      "composite_failure_score_mean": 0.3
    }
    /* B … I ditto */
  }
}
```

### `summary.txt` Columns

One header row, then 9 category rows (only categories with ≥1 image shown), then ALL row, then failure count breakdown:

| Column Header | Meaning | Unit |
|---|---|---|
| `Cat` | 1-letter code A–I / `ALL` | code |
| `Name` | human name `single_isolated`… | string |
| `N` | number of evaluated images in this category | count |
| `AP50_b` | mean box AP@0.50 | 0–1 float |
| `mAP_b` | mean box AP@0.50:0.95 | 0–1 float |
| `AP50_m` | mean mask AP@0.50 | 0–1 float |
| `mAP_m` | mean mask AP@0.50:0.95 | 0–1 float |
| `CntE%` | mean count error % | percent |
| `Miss%` | mean missed grain rate × 100 | percent |
| `Mrg%` | mean merged grain rate × 100 | percent |
| `Spl%` | mean split grain rate × 100 | percent |
| `Dup%` | mean duplicate rate × 100 | percent |
| `FF%` | mean false foreign rate × 100 | percent |
| `ms` | mean inference ms per image | ms |

Failure count footer: each `FAILURE_TYPE` with absolute count and `% of GT rice`.

### `failures_by_category.csv` Columns (from `analyze_failures.py`)

Written to `results/evaluation/failures_by_category.csv` after running `analyze_failures.py`. One row per category (A–I) with ≥1 image:

```
category, category_name, num_images, total_gt_rice,
mean_ap50_box, mean_ap50_95_box, mean_ap50_mask, mean_ap50_95_mask,
mean_count_error_pct, composite_failure_score_mean,
MISSED_GRAIN, MERGED_GRAINS, SPLIT_GRAIN, FALSE_RICE, FALSE_FOREIGN_MATTER, DUPLICATE, BAD_MASK, LOW_CONFIDENCE,
MISSED_GRAIN_rate, MERGED_GRAINS_rate, SPLIT_GRAIN_rate, FALSE_RICE_rate, FALSE_FOREIGN_MATTER_rate, DUPLICATE_rate, BAD_MASK_rate, LOW_CONFIDENCE_rate
```

Each `*_rate` column is the absolute failure count / `total_gt_rice` for that category (0–1 float).
