# Training Documentation — Rice Quality AI

## Prerequisites

### Python Version
Requires **Python ≥ 3.10** (for modern typing + torch/ultralytics compatibility).

### pip Install Commands

From the project root:

```bash
# Core ML stack (PyTorch with CUDA support — install the wheel matching your CUDA version)
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# YOLOv8 + COCO tools + CV
pip install ultralytics>=8.1.0
pip install opencv-python>=4.8.0
pip install pycocotools>=2.0.6
pip install pyyaml>=6.0.0

# Numerics + utilities (also required)
pip install numpy>=1.24.0 scipy>=1.10.0 scikit-image>=0.21.0
pip install pillow>=10.0.0 imagehash>=4.3.1 matplotlib>=3.7.0
pip install tqdm>=4.65.0
```

Alternatively, the backend requirements file covers most of these:

```bash
pip install -r backend/requirements.txt
```

### Hardware Requirements

| Tier | GPU VRAM | CPU RAM | Storage | Notes |
|---|---:|---:|---:|---|
| **Recommended** | ≥ 8 GB (NVIDIA, CUDA 11.8+) | ≥ 16 GB | ≥ 20 GB SSD | yolov8l-seg batch=16 at imgsz=640 comfortably fits; 100 epochs in hours, not days |
| **Minimum GPU** | 4–6 GB (NVIDIA) | ≥ 8 GB | ≥ 15 GB | Reduce batch to 4–8; expect ~2× training time; `train_with_fallback` auto-reduces batch on OOM |
| **CPU only** | N/A | ≥ 16 GB | ≥ 10 GB | Supported but slow: 100 epochs may take **ESTIMATE 10–20× longer** than equivalent 8 GB GPU (literature estimate, not measured) |

Windows note: DataLoader `num_workers` is hard-set to 0 on Windows (multiprocessing spawn issues with OpenCV/PyTorch). Linux/macOS uses `num_workers=4` on ≥8 GB GPUs.

---

## Training Order (Step-by-Step Protocol)

Complete each step before proceeding. A step cannot be skipped if its outputs feed the next.

| Step | Action | Script / Command | Outputs |
|---|---|---|---|
| 1 | **Audit datasets** — verify mask type, file integrity, grain-count distribution | `python scripts/audit_dataset.py --dataset all` | `datasets/inspection/audit_report.json`, previews |
| 2 | **Convert GrainSet v3 → COCO** — instance or semantic mask handling per audit result | `python scripts/convert_to_coco.py --mask-type auto` | `datasets/processed/coco/annotations/instances_{train,val,test}.json` |
| 3 | **Generate synthetic data** — 1,900 images across 6 density/FM modes | `python scripts/generate_synthetic.py` | `datasets/processed/coco/images/synthetic/*.png` + `instances_synthetic.json` |
| 4 | **Create stratified splits** — merge real + synthetic, 70/15/15, density-stratified | `python scripts/create_splits.py` | `datasets/splits/*.txt`, filtered per-split COCO JSONs, `tests/datasets/*/` subdirs |
| 5 | **Train YOLO Run 1** — baseline on real+synth, 100 epochs, default hyperparams | `python training/train_yolo_seg.py --name run1_baseline` | `models/yolo_seg/run1_baseline/weights/best.pt` + `last.pt` |
| 6 | **Evaluate Run 1** — full evaluation on test split + 27 hand-curated scenarios | `python scripts/evaluate_model.py --model-type yolo --generate-failure-images` | `results/evaluation/{eval_report.json, summary.txt, failures/*/}` |
| 7 | **Iterative retraining** (if needed) — failure analysis → add data / tune → Run 2 / Run 3 fine-tuning | `python training/train_yolo_seg.py --name run2_synth_finetune --resume` | `models/yolo_seg/run2_synth_finetune/weights/best.pt`, appended run log |

---

## YOLOv8l-seg Training

### Model Choice: yolov8l-seg (large)
Trained via `ultralytics.YOLO("yolov8l-seg.pt")` — **COCO-pretrained initialization by default** (auto-downloaded on first run if `~/.cache/torch/hub/checkpoints/` is empty).

### Config Details

All defaults live in `training/train_yolo_seg.py:199-225` and are overridable by CLI flags.

| Parameter | Default | CLI Flag | Rationale |
|---|---:|---|---|
| epochs | 100 | `--epochs 100` | COCO pretrained converges well within 100; `patience=20` early-stops if plateaued |
| imgsz | 640 | `--imgsz 640` | Matches synthetic generator tile size; native YOLO stride multiples; good speed/accuracy tradeoff for rice grains |
| batch | 16 | `--batch 16` | Fits 8 GB VRAM at 640²; OOM handler halves to 8 then retries once |
| optimizer | AdamW | `--optimizer AdamW` | Weight decay stabilizes fine-tuning; SGD available via `--optimizer SGD` |
| lr0 (initial LR) | 0.001 | `--lr0 0.001` | Standard AdamW learning rate for fine-tuning pretrained backbones |
| lrf (final LR factor) | 0.01 | hardcoded | Cosine decay to lr0×0.01 |
| patience | 20 | hardcoded | Early stop after 20 epochs without `best_fitness` improvement |
| save_period | 10 | hardcoded | Save checkpoint every 10 epochs (in addition to best + last) |

### Augmentation Pipeline (Ultralytics built-in)

Enabled by default in the training kwargs; intensities are tuned for rice grain morphology (not aggressive on colour since grain colour matters for defect classification).

| Augmentation | Value | Effect |
|---|---:|---|
| degrees | 45.0° | Random rotation ±45° (rice orientation is arbitrary) |
| flipud | 0.5 | Vertical flip probability |
| fliplr | 0.5 | Horizontal flip probability |
| mosaic | 1.0 | 4-image mosaic enabled every batch |
| copy_paste | 0.3 | Copy-paste instance augmentation 30% of samples — **critical for FM and dense grains** |
| mixup | 0.1 | Mild image blending to reduce overfitting |
| hsv_h | 0.015 | Very mild hue shift (preserve rice vs FM colour distinction) |
| hsv_s | 0.7 | Larger saturation jitter |
| hsv_v | 0.4 | Brightness jitter (simulates scanner lamp variation) |

### Training Command

```bash
# Run 1 baseline — COCO pretrained, auto hardware detect
python training/train_yolo_seg.py \
  --name run1_baseline \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --optimizer AdamW \
  --notes "Baseline run: GrainSet v3 + synthetic 1900 images, default augs, seed 42 data split"

# Explicit device override (useful for multi-GPU or forced CPU)
python training/train_yolo_seg.py --name run1_baseline --device cuda:0
python training/train_yolo_seg.py --name run1_baseline --device cpu
```

Dataset YAML path defaults to `configs/rice_seg_yolo.yaml`. Contents:

```yaml
path: ../datasets/processed/coco
train: images/train
val:   images/val
test:  images/test
nc:    2
names: ['rice_grain', 'foreign_matter']
```

### Checkpoint Locations

```
models/yolo_seg/{run_name}/
├── weights/
│   ├── best.pt          # Best validation fitness — USE THIS for inference / evaluation
│   └── last.pt          # Last epoch checkpoint — used for resume
├── results.csv
├── results.png
├── confusion_matrix.png
├── events.out.tfevents.*  # TensorBoard log
└── args.yaml              # Reproducible training arguments
```

The script appends every run to `training/run_log.md` with: run number, date, epochs / best epoch, best val mAP@50 and mAP@50:95, duration, hardware, notes.

### Fine-Tuning / Resuming

Run 2+ protocols use the previous run's `best.pt` as initialization:

```bash
# Implicit fine-tune — automatically finds best.pt from the named run and uses it as init
python training/train_yolo_seg.py --name run1_baseline --epochs 60 --notes "Run 2: +200 FM-heavy synthetic images"

# Explicit resume (continues a run log entry; same as fine-tune but restores optimizer state)
python training/train_yolo_seg.py --name run1_baseline --resume --epochs 40

# Run a 2nd-round training that weights synthetic data more heavily (requires re-generating splits with adjusted ratios)
python scripts/generate_synthetic.py --num-foreign 300 --num-touching 800 --seed 99
python scripts/create_splits.py --force-resplit
python training/train_yolo_seg.py --name run2_synth_heavy --resume --epochs 80 \
  --weights models/yolo_seg/run1_baseline/weights/best.pt
```

---

## Mask R-CNN Baseline

**When to use it**: comparison only — used once during the architecture comparison phase to justify the YOLOv8l-seg primary choice. Not used for production after the decision.

### Config

| Parameter | Default | CLI Flag |
|---|---:|---|
| Backbone | ResNet50-FPN-v2 (ImageNet pretrained) | fixed |
| Epochs | 50 | `--epochs 50` |
| Batch size | 4 | `--batch-size 4` |
| Optimizer | SGD (momentum=0.9, wd=5e-4) | fixed |
| LR | 0.005 | `--lr 0.005` |
| Scheduler | StepLR step=20, γ=0.1 | fixed |
| Gradient clip norm | 5.0 | `--grad-clip 5.0` |
| Early stop patience | 10 (val mask mAP@50) | `--patience 10` |
| Train augs | RandomHorizontalFlip(0.5) + ToTensor + ImageNet norm | fixed |
| Min/max image size | 800 / 800 (built-in FPN resizer) | fixed |

### Training Command

```bash
python training/train_maskrcnn.py \
  --epochs 50 \
  --batch-size 4 \
  --lr 0.005 \
  --data-root datasets/processed/coco \
  --patience 10 \
  --grad-clip 5.0
```

Checkpoints saved to:
- `models/maskrcnn/best/model.pth` — best val mask mAP@50
- `models/maskrcnn/last/model.pth` — last epoch (resumable via `--resume path/to/.pth`)

Run entry also appended to `training/run_log.md`.

---

## Iterative Training Protocol (3-Round Recipe)

This is the *recommended* schedule after the first baseline evaluation returns a usable but imperfect model:

| Round | Data | Init | Objective | Success Criterion (placeholder targets — literature ESTIMATES) |
|---|---|---|---|---|
| **Round 1** | GrainSet v3 + default synthetic 1,900 | COCO pretrained yolov8l-seg | Establish baseline performance across all 9 test categories | mAP@50 box ≥ 0.80 overall; no category has mask AP < 0.50 |
| **Round 2** | Round 1 split + *failure-informed* additions: more synthetic for the 2 worst failure categories (typically touching=C or overlapping=D or dense=E); re-split keeping val/test fixed | `best.pt` from Round 1 (resume / fine-tune, lower LR=0.0005, 60 epochs) | Reduce top-2 failure types by ≥30% in absolute counts | MERGED_GRAINS + MISSED_GRAINS counts drop by at least 30% each vs Round 1 eval on same val/test set |
| **Round 3** | Round 2 split + hard example mining: top-50 worst images from Round 2 eval get human-verified GT + new dense/FM patches; add 200 very_dense synthetic images | `best.pt` from Round 2, LR=0.0002, 40 epochs (small decay) | Converge performance; no regressions; prepare for deployment | All categories A–I have mask AP ≥ 0.65; count error% mean < 10 overall; 0 false-FM on clean-rice images |

Between every round, run:
1. `python scripts/evaluate_model.py ... --generate-failure-images`
2. `python scripts/analyze_failures.py --append-docs --run-info '{"run_id":"runN_...","notes":"..."}'`
3. Manually inspect `results/evaluation/failures/*/` worst images.
4. Design data additions / hyperparameter changes for the next round.

Do NOT skip the failure analysis step — without it, retraining is blind.

---

## Hyperparameter Reference (Full Tunable Table)

All trainable knobs with their defaults and when you would want to change them.

| Parameter | Default | Range to Try | Change When… |
|---|---:|---|---|
| `epochs` | 100 | 30–200 | Training loss still dropping at 100 → increase; already overfitting at epoch 40 → decrease and rely on patience |
| `imgsz` | 640 | 512, 640, 832, 1024 | Mask boundary quality too coarse on small rice fragments → 832 or 1024 (requires ≥12 GB VRAM or batch=4 at 640-equivalent) |
| `batch` | 16 | 4, 8, 16, 32 | GPU VRAM headroom > 3 GB → 32; OOM → 8 or 4; keep effective batch ≥16 (accumulate grads if needed; Ultralytics doesn't do this by default — use smaller batch instead) |
| `optimizer` | AdamW | SGD, Adam, AdamW, RMSProp | Production with long schedule → SGD often generalizes slightly better but needs 2× epochs to converge. AdamW for fast iteration. |
| `lr0` | 0.001 (AdamW) / 0.01 (SGD) | 1e-4 … 1e-2 (AdamW); 1e-3 … 0.1 (SGD) | Fine-tuning from an existing rice checkpoint → 0.5×–0.2× default. Diverging loss → halve LR. Not learning → double LR. |
| `lrf` (final LR factor) | 0.01 | 0.001 … 0.05 | Linear/step schedule — 0.01 means 100:1 ratio; tune with scheduler shape if using custom |
| `patience` | 20 | 5 … 40 | Small dataset / aggressive augs → 30–40; large dataset, quick convergence → 10 |
| `degrees` | 45.0 | 0 … 180 | Rice orientation in lab is arbitrary → keep 45–90; never set to 0 unless data is intentionally oriented |
| `mosaic` | 1.0 | 0.0, 0.5, 1.0 | Very small samples → keep 1.0 to increase effective variance; already dense beyond 100 grains / image → 0.5 (less noise during final fine-tune epochs) |
| `copy_paste` | 0.3 | 0.0 … 0.5 | Foreign-matter or rare-grain counts are low → raise to 0.5; spurious false-rice on background → drop to 0.1 |
| `mixup` | 0.1 | 0.0 … 0.4 | Overfitting → 0.3–0.4; already label noise in GT → 0.0 |
| `hsv_h / hsv_s / hsv_v` | 0.015 / 0.7 / 0.4 | h: 0–0.05; s: 0–1.0; v: 0–0.8 | Colour-based defect classifier later uses HSV → keep `hsv_h` SMALL (≤0.02); FM vs rice discrimination needs saturation range → keep s/v high |
| `conf` (inference threshold) | 0.25 | 0.10 … 0.50 | High recall missed grains dominant → 0.10; high FP/FM on clean → 0.40–0.50 |
| `iou` (box NMS threshold) | 0.7 (Ultralytics default) | 0.3 … 0.7 | Merged grains dominant → tighten to 0.4–0.5; many duplicates → tighten; post `mask_iou_nms` at 0.5 handles the rest |

---

## Hardware Notes

### GPU Detection
- `detect_hardware()` in `train_yolo_seg.py:39` checks `torch.cuda.is_available()`. If CUDA is present: uses GPU, reports device name and VRAM, scales workers accordingly. Else CPU.
- Override with `--device cuda:0`, `--device cuda:0,1` (DDP multi-GPU — Ultralytics handles DDP internals), or `--device cpu`.

### Windows num_workers = 0
Multiprocessing DataLoader workers on Windows spawn new Python processes that re-import the training script; combined with OpenCV's fork-unsafety this reliably deadlocks or crashes. Workers forced to 0 on `platform.system() == "Windows"`. Training will be loader-bound but correctness-first.

### Batch Size vs VRAM (ESTIMATES from literature)

Approximate yolov8l-seg memory consumption per-image at imgsz=640 with mixed-precision enabled (Ultralytics default on CUDA):

| Batch Size | VRAM Used EST (GB) | Works On |
|---:|---:|---|
| 4 | ~3.5 | 4 GB+ GPUs |
| 8 | ~5.5 | 6 GB+ GPUs |
| 16 | ~8.5 | 8 GB+ GPUs (recommended default) |
| 32 | ~15.5 | 16 GB+ GPUs (A100 / 3090 / 4090 class) |

These are **literature / architectural ESTIMATES, not measured benchmarks**. The `train_with_fallback` wrapper catches RuntimeError OOM and retries once with `batch // 2`.

### Estimated Training Time Placeholders (PRE-TRAINING LITERATURE ESTIMATES, NOT MEASURED)

Hardware: 8 GB consumer NVIDIA GPU (e.g. RTX 3070 / 4070 class), imgsz=640, batch=16, ~5,000 train images + 1,900 synthetic = ~6,900 training images total.

| Epochs | Estimated Wall Time (EST) | Notes |
|---:|---:|---|
| 30 | ~1.5–3 h | Quick smoke-test / prototype run |
| 100 (baseline, Round 1) | ~6–12 h | Typical overnight run |
| 60 (fine-tune, Round 2) | ~4–8 h | Converges faster because init is better |
| 40 (hard example, Round 3) | ~3–5 h | Small LR changes, mostly refinement |

CPU equivalent: multiply all estimates by **10–20×** (rough literature ratio for comparable PyTorch workloads). Mask R-CNN baseline is **2–3× slower per epoch** than YOLOv8l-seg on the same GPU (FPN + two-stage architecture cost).

---

## Reproducibility

Reproducibility is a hard requirement — every training run must be exactly reproducible given the same raw data on disk.

| Surface | Control |
|---|---|
| Dataset splitting | `random.seed(42)` in `create_splits.py` and `convert_to_coco.py` auto-split paths. Use `--force-resplit` only when you intentionally want to change the split. |
| Synthetic generation | `np.random.default_rng(seed)` + `cv2.setRNGSeed(seed & 0xFFFFFFFF)`. Default seed=42. |
| Data order in DataLoader | `shuffle=True` seeded by torch seed (Ultralytics sets global torch/numPy/random seeds internally) |
| Augmentation randomness | Ultralytics trainer sets deterministic seeds when `deterministic=True` mode is enabled via its args; for strict bit-exact reproduction, also set `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA 10.2+ environments (documented for reproducibility, not defaulted) |
| Model init weights | First run: `yolov8l-seg.pt` (COCO pretrained SHA from Ultralytics release) — recorded in `args.yaml` per-run. Round 2+: explicitly resume from a specific `best.pt`. |
| Post-training eval | Use same `evaluate_model.py` invocation with same `--conf 0.25 --split test` flags. |
| Run log | All runs are appended to `training/run_log.md` with date, hardware, args, and final metrics. Include a unique `--notes` string. |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| **RuntimeError: CUDA out of memory** | Batch size too large for GPU VRAM + imgsz combination | Retry with `--batch 8` or `--batch 4`; reduce `--imgsz 512` if still failing. `train_with_fallback` already retries once automatically. |
| **ImportError: No module named 'ultralytics'** | pip install not run or wrong environment | `pip install ultralytics>=8.1.0`; confirm `python --version` ≥ 3.10 and matches the env where packages were installed. |
| **torch / torchvision import error or `torch.cuda.is_available()==False`** | CPU-only PyTorch wheel installed instead of CUDA build | Reinstall with the CUDA wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` (match cuDNN/CUDA to driver). |
| **Dataset config not found: `configs/rice_seg_yolo.yaml` missing** | `convert_to_coco.py` and `create_splits.py` not run before training | Run steps 2–4 in order. The YAML references `../datasets/processed/coco` which must contain at least `images/train/` and `annotations/instances_train.json`. |
| **0 images found in COCO loader** | COCO annotation JSON has wrong `images[].file_name` entries OR image files not copied into `images/{train,val,test}/` | Run `convert_to_coco.py` WITHOUT `--no-copy-images`; re-run `create_splits.py --copy-images`; verify `images/train/` is populated. |
| **Training stalls at epoch 0, 0% progress** | DataLoader workers deadlock (Windows symptom) or corrupt image file | On Windows, workers are already set to 0 — check for corrupt PNG/JPG in train/ by re-running audit; `audit_dataset.py` flags corrupt_files. |
| **Loss = NaN on epoch 1** | LR too high, or bad annotations (zero-area polygons) in COCO JSON | Halve `--lr0`; inspect `instances_train.json` for entries where `"area" < 50` or `"bbox"` has zero w or h. |
| **val mAP stays exactly 0.0 for ≥10 epochs** | Category ID mismatch between dataset YAML `names` and COCO `categories` | Verify YAML `nc: 2, names: ['rice_grain', 'foreign_matter']` exactly matches COCO JSON `categories[].name` order (rice_grain=1 maps to YAML index 0 in Ultralytics convention). |
| **pycocotools build fail on Windows** | Missing Visual C++ build tools | `pip install pycocotools-windows` or use conda-forge distribution; `evaluate_model.py` falls back to built-in simplified mAP when pycocotools is absent (flagged in eval_report JSON as `"pycocotools_used": false`). |
