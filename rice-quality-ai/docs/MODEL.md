# Model Documentation — Rice Quality AI

## Scope

Phase 1 scope only: **Detection + Instance Segmentation + Foreign Matter detection.**  
Grading, 14-parameter classification, and defect analysis are explicitly out of scope  
until every Phase 1 gate in the implementation plan is checked.

---

## 1. Candidate Architecture Evaluation

Every candidate was scored on five axes that matter for rice grain quality work in an
analyst-facing lab setting:

| Axis | Why it matters |
|---|---|
| **Touching grain separation** | Real samples have 2–8 grain clusters; watershed alone fails here. |
| **Overlapping grain separation** | Broken, chalky, or piled grains overlap; the model must recover boundaries. |
| **Dense scene (100+ grains)** | Bulk sample trays are the standard case; dense recall must not collapse. |
| **CPU inference speed** | Analyst workstations often lack GPUs; > 500 ms feels sluggish. |
| **Training & deployment ease** | The project is maintained by non-ML engineers; `pip install` + a single `.pt` file is the bar. |

### Scored Comparison

| Model | Touching | Overlap | Dense (100+) | Speed (CPU est.) | Train Ease | Verdict |
|---|---|---|---|---|---|---|
| **YOLOv8l-seg** | Good | Good | Excellent | 30–80 ms / 640px | Very easy | **PRIMARY — Phase 1 production model** |
| **Mask R-CNN ResNet50-FPN-v2** | Good | Good | Fair | 200–500 ms / 800px | Moderate | **COMPARISON BASELINE only** |
| YOLOv9-seg | Good | Good | Excellent | 40–90 ms | Easy | Alternative if YOLOv8 onnx-export blocks |
| SAM (Segment Anything) | Excellent | Excellent | Excellent | Very slow (> 2 s) | Complex | Too heavy; rejected on speed + license |
| SOLO v2 | Good | Good | Good | ~150 ms | Hard | Not practical; rejected on deploy complexity |
| Cascade Mask R-CNN | Good | Good | Fair | 400–800 ms | Hard | No CPU path; rejected |
| Classical CV (Watershed + Distance Transform) | Poor | Poor | Poor | 10–40 ms | N/A | **EMERGENCY FALLBACK only — never deleted** |

**Baseline justification:** Mask R-CNN is kept as a comparison baseline for two reasons:
(1) the existing `train_segmentation.py` manifest referenced it, so the comparison
satisfies the original intent, and (2) published rice-segmentation papers often use it,
giving a literature-comparable number. It is **never** the production deployment target.

---

## 2. Final Architecture Stack (Phase 1)

```
                 ┌──────────────────────────────────────────────┐
                 │         ml.inference.analyze_image()          │
                 │  (public entrypoint in ml/inference.py)       │
                 └──────────────────────┬───────────────────────┘
                                        │ priority cascade
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌─────────────────────┐   ┌─────────────────────────┐   ┌───────────────────────┐
│ 1. YOLOv8l-seg      │   │ 2. Mask R-CNN (baseline)│   │ 3. Classical CV       │
│    .pt weights      │──▶│    .pth weights         │──▶│    Watershed fallback  │
│ PRIMARY (when .pt)  │   │ COMPARISON only         │   │ EMERGENCY always      │
└─────────┬───────────┘   └────────────┬────────────┘   └──────────┬────────────┘
          │                             │                           │
          └─────────────────────────────┼───────────────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │   PostProcessor                  │
                       │   ml/postprocessing.py           │
                       │   1. Confidence filter (0.25)    │
                       │   2. YOLO built-in box NMS       │
                       │   3. Mask-level IoU NMS (0.5)    │
                       │   4. Tiling (640 tile, 128 ovlp) │
                       │   5. Global ID assignment        │
                       │   6. Area >= 50 px² verify       │
                       │   7. Touching detection          │
                       └───────────────┬──────────────────┘
                                       ▼
                    Structured dict output (see ml/inference.py docstring)
```

### Model variants on disk

| Variant | Path | When used |
|---|---|---|
| YOLOv8l-seg (large) | `models/yolo_seg/run1_baseline/weights/best.pt` | Default production; best accuracy on dense/touching |
| YOLOv8n-seg (nano) | `models/yolo_seg/run_nano/weights/best.pt` | CPU-only fallback if large model is too slow; selected automatically if filename contains `nano` or `yolov8n` |
| Mask R-CNN baseline | `models/maskrcnn/best/model.pth` | Loaded when YOLO `.pt` is missing; comparison metrics only |
| Classical CV fallback | (code in `ml/segmentation.py`) | Always available; zero-dependency; retained forever |

---

## 3. YOLOv8l-seg Training Specification

Primary training script: `training/train_yolo_seg.py`
Dataset config: `configs/rice_seg_yolo.yaml`

| Hyperparameter | Value | Rationale |
|---|---|---|
| Backbone / head | YOLOv8l-seg (P3–P5 FPN + decoupled head) | Anchor-free; best accuracy/size tradeoff on grains |
| Pretrained init | `yolov8l-seg.pt` (COCO) | Transfer learning cuts training 5–10×; grains share enough with COCO "person/car/bird" contours that the early backbone transfers well |
| Image size | `imgsz=640` | Fits ~16 large grains per side; covers 95 % of lab samples; good CPU speed |
| Batch size | `batch=16` (GPU), `batch=4` (CPU OOM fallback) | Tuner in `train_yolo_seg.py` halves batch on RuntimeError OOM |
| Optimizer | AdamW, `lr0=0.001`, `weight_decay=5e-2` | Literature standard for object-detection fine-tune; beats SGD on small-to-medium datasets |
| Final LR factor | `lrf=0.01` (cosine decay to 1 % of peak) | Gentle decay avoids forgetting fine-grained grain boundaries |
| Epochs | 100 with `patience=20` early stop | Typical rice-seg runs converge 40–70 epochs; 20-epoch patience is safe |
| Degrees | ±45° | Grains lie in any orientation; full rotation equivariance needed |
| Flip up-down | 0.5 prob | Rice has no top/bottom |
| Flip left-right | 0.5 prob | Same |
| Mosaic | 1.0 (on by default) | Mixes 4 images → forces the model to count dense clusters correctly |
| Copy-paste | 0.3 | **Most important augmentation for this project.** Literally pastes grains onto existing grains → creates synthetic touching/overlap pairs the model must learn to split. Without this, dense touching mAP collapses. |
| Mixup | 0.1 | Light blend robustness; kept low because it blurs fine boundaries |
| HSV H | 0.015 | Small hue shift (variety color differences are small) |
| HSV S | 0.70 | Large saturation range (phone cameras vs flatbed scanners) |
| HSV V | 0.40 | Brightness range for lab vs overhead lighting |
| Box gain | 7.5 (default) | Balances the multi-task loss |
| Segmentation gain | 1.2 (default) | Emphasizes mask quality slightly over boxes |
| Save period | Every 10 epochs | Resume from intermediate if run is killed |

---

## 4. Mask R-CNN Baseline Specification

Training script: `training/train_maskrcnn.py`

| Parameter | Value |
|---|---|
| Backbone | ResNet50-FPN-v2 (torchvision DEFAULT pretrained) |
| Image size | min 800, max 1333 (fixed by MaskRCNN class) |
| RPN / RoI | Defaults (512 RPN, 2000 train / 1000 test NMS) |
| Optimizer | SGD, lr=0.005, momentum=0.9, weight_decay=5e-4 |
| Scheduler | StepLR every 3 epochs, gamma 0.1 |
| Epochs | 50, patience 10 early stop |
| Batch size | 4 (CPU), 8 (GPU) |
| Gradient clip | 1.0 norm |
| Save best on | `val_mask_ap_50` |

This baseline is **not** the production target. Its only job is to produce a comparison
value so the YOLOv8 choice can be justified ("YOLOv8 was X mAP better at Y× speed").

---

## 5. Model Cascade / Fallback Behaviour (Production)

`_load_model()` in `ml/inference.py` enforces a strict priority order:

```
1. Does models/yolo_seg/run1_baseline/weights/best.pt exist
   AND `from ultralytics import YOLO` imports cleanly?
   → Use YOLOv8l-seg.

2. Else, does models/maskrcnn/best/model.pth exist
   AND torch + torchvision are importable?
   → Use Mask R-CNN.

3. Else (no weights, no GPU, missing packages, corrupted files)?
   → Classical CV watershed in ml/segmentation.segment_grains.
   → Also appends a warning string to analysis.warnings.
```

None of the fallback tiers are ever deleted. Classical CV stays because it has zero
external dependencies and is the only layer guaranteed to work on a fresh offline
install.

---

## 6. Output Contract (what the model promises to return)

Defined in the docstring of `ml.inference.analyze_image()`. Copy:

```
{
  "rice_detected": bool,
  "rice_count": int,
  "foreign_matter_count": int,
  "unresolved_cluster_count": int,
  "grains": [
    {
      "id": 1,
      "confidence": 0.97,
      "confidence_label": "HIGH",        # HIGH >= 0.80, MEDIUM 0.50–0.80, LOW < 0.50
      "bbox": [x, y, w, h],
      "mask_polygon": [[x1,y1],[x2,y2], ...],
      "centroid": [cx, cy],
      "area_pixels": 1234,
      "is_touching": false,
      "segmentation_method": "yolov8l-seg"
    }
  ],
  "foreign_matter": [
    { "id": 1, "class": "stone", "confidence": 0.89, "bbox": [x, y, w, h] }
  ],
  "unresolved_clusters": [],
  "processing": { "inference_ms": 340, "total_ms": 397 },
  "method": "yolov8l-seg",
  "model_version": "1.0.0"
}
```

Additional invariants:

- `rice_count == len(grains)` always.
- `confidence_label` is derived, not stored. Thresholds are constants at the top
  of `ml/inference.py` and the frontend `Phase1Demo.tsx` (values: HIGH >= 0.80,
  MEDIUM >= 0.50, LOW otherwise).
- No foreign matter entry is ever created from an `unresolved_cluster`.
- `rice_detected == False` iff `rice_count == 0` AND the rice gate agrees (image
  passes the "background-only / definitely-no-rice" heuristics — see
  `ml/rice_gate.py`). A dense scene where detection was poor still returns
  `rice_detected = True` if at least one high-confidence grain survived.

---

## 7. Pre-Training Literature Estimates (HONEST — NOT MEASURED YET)

These values are copied from comparable rice-segmentation papers. They are
**planning targets only**, not claims. Actual measured values will be written by
`scripts/evaluate_model.py` into `results/evaluation/eval_report.json` and
compared against these rows.

### Run 1 baseline (before any synthetic augmentation or hard-example mining)

| Category | Expected mAP@50 | Expected Count Error % |
|---|---|---|
| A — single isolated | 0.90 – 0.97 | < 2 % |
| B — few separated | 0.85 – 0.95 | < 5 % |
| C — touching (2–3 grains) | 0.65 – 0.80 | 5 – 15 % |
| D — overlapping | 0.55 – 0.75 | 10 – 25 % |
| E — dense (50+ grains) | 0.60 – 0.80 | 5 – 15 % |
| F — rice + foreign matter | 0.80 – 0.90 | < 10 % |
| G — no rice | N/A (image-level) | **0 % false-rice rate required** |
| H — foreign only | 0.70 – 0.90 (FM AP) | N/A |
| I — mixed difficult | 0.50 – 0.70 | 15 – 30 % |

### After iterative training (Run 2 / Run 3 — targets, not promises)

| Category | Target mAP@50 | Target Count Error % |
|---|---|---|
| A — single isolated | >= 0.95 | < 1 % |
| C — touching | >= 0.80 | < 8 % |
| D — overlapping | >= 0.70 | < 15 % |
| E — dense (50+) | >= 0.80 | < 10 % |
| G — no rice | **still 0 % false rice** | 0 % |

The iterative protocol is:

1. Run `evaluate_model.py`.
2. Run `analyze_failures.py` → inspect `docs/FAILURE_ANALYSIS.md` section.
3. If C fails → add 500+ touching synthetic images and fine-tune from best.pt.
4. If E fails → add 200+ dense synthetic images.
5. If D fails → add 500+ overlapping synthetic images.
6. Re-evaluate. Repeat max 3 rounds, then stop and report honestly.

---

## 8. Known Failure Modes the Model Cannot Fix

Documenting up-front so users set correct expectations:

1. **Grain-through-grain physical overlap** (one grain actually sitting on top of
   another, covering > 40 % of the area underneath). No segmentation model —
   including human annotators — can recover the boundary of a completely hidden
   grain. The pipeline returns the top grain only and marks the lower one as an
   `unresolved_cluster` if the shape suggests occlusion.
2. **Chalky grains that blend to the background color.** If a chalky grain is
   photographed against a near-white plate with no side lighting, even humans
   struggle to place the boundary. In these cases confidence_label will be LOW
   and the analyst is expected to re-photograph with better lighting.
3. **Foreign matter that is the exact size/shape/colour of a grain.** Small
   white pebbles and some seed admixtures will be mis-classified as rice some of
   the time. `false_foreign_rate` is tracked explicitly in the evaluation
   framework because this is the safety-relevant failure.
4. **No-rice images that *contain* tiny rice-coloured dust speckles.** These
   trigger false positives occasionally. Confidence is usually < 0.30, so the
   0.25 default threshold catches most, but the no-rice test protocol includes
   a strict 0-false-rice bar that forces the threshold to be re-tuned if any
   dust speck is mis-labelled as a grain.

---

## 9. Run Log (Training Runs Section)

See `training/run_log.md`, auto-appended by `train_yolo_seg.py` and
`train_maskrcnn.py` after every completed run. No manual edits here.

---

## 10. Model Weights Provenance & Honesty Policy

- **Never commit weights to git.** The `models/*/` folders contain only
  `.gitkeep`, metadata JSON, and class-mapping JSON files. Weights are distributed
  via a separate artifact store (out of scope for Phase 1).
- **Never copy-paste mAP numbers from another paper into a run log.** Only
  values produced by `scripts/evaluate_model.py` on the project's own held-out
  test set count. Literature values go only in Section 7 of this file and are
  labelled "PRE-TRAINING LITERATURE ESTIMATES."
- **If a run fails or produces worse numbers than the previous run, record it.**
  Deleting bad runs poisons the honesty of the comparison baseline.
- **Report the hardware and exact version.** Each run log entry records GPU/CPU
  model, PyTorch version, and ultralytics/torchvision version so the next person
  can reproduce the numbers.
