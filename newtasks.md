Yes. For **this phase**, I would completely separate the problem from the 14-parameter grading system.

Your target should be:

> **Image → detect rice + foreign matter → detect every individual rice grain → produce one mask per grain → correctly separate touching/overlapping grains → assign a unique ID → count them.**

And importantly, **do not train a model only on clean, separated rice images**. The dataset must deliberately contain isolated, touching, partially overlapping, heavily overlapping, dense, rotated, broken, partially occluded, different lighting/backgrounds, and foreign-matter cases.

Recent work specifically on dense overlapping rice reports that severe adhesion/occlusion/overlap is a major segmentation problem; one 2026 study used 1,046 images with 78,422 annotated instances for this exact kind of situation. ([ScienceDirect][1])

Below is the plan I would give to your coding agent.

---

# 1. Final model we are trying to build

The system should ultimately behave like this:

```text
                    INPUT IMAGE
                         │
                         ▼
                ┌─────────────────┐
                │ Image Validation│
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
        NO RICE PRESENT        RICE PRESENT
              │                     │
              ▼                     ▼
        "No rice detected"   Rice + FM Detection
                                    │
                                    ▼
                         Instance Segmentation
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
          Grain #1              Grain #2              Grain #N
              │                     │                     │
         individual mask      individual mask      individual mask
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                         Foreign Matter Objects
                                    │
                                    ▼
                         Final Annotated Image
```

For example:

```text
Input:
    150 rice grains
    3 stones
    2 wheat kernels
    1 plastic piece

Output:

Rice:
    Grain #001 → mask
    Grain #002 → mask
    ...
    Grain #150 → mask

Foreign matter:
    FM #001 → stone
    FM #002 → stone
    FM #003 → wheat
    ...

Total rice = 150
Total foreign matter = 6
```

---

# 2. Most important requirement: INSTANCE segmentation

Do **not** train a semantic segmentation model where all rice pixels become one big mask.

You need:

```text
Semantic segmentation:

████████████████
████ RICE ██████
████████████████

= one rice region
```

but you need:

```text
Instance segmentation:

   Grain 1      Grain 2
    ████         ████
   █████        █████
    ████         ████

   Grain 3      Grain 4
    ████        █████
   █████        █████
```

Each grain must have:

```text
instance_id
mask
bounding_box
confidence
```

For example:

```json
{
    "id": 37,
    "class": "rice",
    "confidence": 0.97,
    "mask": "...",
    "bbox": [x1, y1, x2, y2]
}
```

This is absolutely critical for touching/overlapping grains.

---

# 3. Dataset strategy

I would **not rely on one dataset**.

You need a **dataset mixture**.

Think of it as:

```text
                  MASTER TRAINING DATASET
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  Clean/isolated       Touching/adhesive    Dense/overlap
      rice                  rice                rice
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                  Foreign matter
                           │
                           ▼
                 Your own real images
```

---

# 4. Dataset A — general rice instance segmentation

A strong starting point is the **Rice-Variety-Classification-System-Dataset** on Mendeley.

It was published in June 2026 and specifically describes multiple grains per image with manual **instance segmentation annotations** intended for YOLOv8 segmentation. ([Mendeley Data][2])

[Rice-Variety-Classification-System-Dataset — Mendeley Data](https://data.mendeley.com/datasets/jcgmmpbv6g/1?utm_source=chatgpt.com)

Use it for:

```text
✓ basic rice detection
✓ individual grain segmentation
✓ different grain orientations
✓ multiple grains/image
✓ baseline instance masks
```

But don't assume this dataset alone solves touching/overlap.

The agent must inspect the actual annotations and quantify:

```text
number of images
number of instances
instances/image
touching instances
overlapping instances
isolated instances
mask format
image resolution
rice varieties
backgrounds
```

---

# 5. Dataset B — dense/touching/overlapping rice

This is probably the **most important dataset category for your problem**.

Recent research specifically constructed a dense rice dataset containing:

> 1,046 images
> 78,422 annotated instances

for densely overlapping rice grains. The paper reports severe adhesion, occlusion and overlap as the central challenge. ([ScienceDirect][1])

The paper also describes SAM-assisted annotation and soft-NMS-style inference improvements for overlapping grains. ([ScienceDirect][1])

However:

**Do not tell the coding agent to blindly download this paper's dataset.**

The agent must first verify whether the dataset is actually publicly downloadable and under a usable license.

If it isn't publicly available:

```text
DO NOT fabricate access.
DO NOT scrape illegally.
DO NOT pretend it is part of training.
```

Instead, reproduce the same **data characteristics** ourselves.

---

# 6. Dataset C — adhesive/touching grains

Another important research direction is the STR-900 dataset.

Recent work describes STR-900 as a rice instance-segmentation dataset containing:

* multiple rice varieties
* varying illumination
* different adhesion levels

and specifically targets segmentation of adhesive rice grains. ([Henan University of Technology Journal][3])

Again, the agent should verify:

```text
Is dataset downloadable?
What license?
Are masks available?
Are masks instance-level?
How many touching cases?
How many overlapping cases?
```

Only then include it.

---

# 7. Dataset D — Grainalyze

There is also a publicly visible Roboflow **Grainalyze Instance Segmentation** dataset.

It is listed as:

```text
Task: Instance Segmentation
License: CC BY 4.0
Images: 2,470
Train: 2,280
Validation: 95
Test: 95
```

and provides a downloadable dataset version. ([Roboflow][4])

[Grainalyze Instance Segmentation Dataset](https://universe.roboflow.com/rice-6hvoz/grainalyze/dataset/11?utm_source=chatgpt.com)

But again:

**don't automatically trust the dataset just because it says instance segmentation.**

The agent should inspect actual samples and calculate:

```text
isolated %
touching %
overlapping %
dense %
background types
```

If 95% are isolated grains, it is not sufficient for your goal.

---

# 8. Dataset E — foreign matter

This is a separate problem.

You need foreign-object images containing things such as:

```text
stone
pebble
soil/clod
wheat
corn
other seeds
plastic
metal
insect
plant material
other impurities
```

A 2025 rice foreign-object study used approximately 5,000 images and included objects such as stones, clods, metal fragments, screws, corn and wheat. ([MDPI][5])

Another 2026 study on rice-processing environments specifically investigated foreign objects including stone, plastic, bag-line material and insects. ([ScienceDirect][6])

But your model needs to operate on **your actual image domain**.

So don't simply train on isolated pictures of stones.

You want:

```text
rice + stone
rice + wheat
rice + plastic
rice + insect
rice + seed
rice + soil
rice + mixed FM
```

inside the same type of scene as your rice images.

---

# 9. Dataset F — YOUR OWN DATASET

This is extremely important.

Even if we download 10 datasets, the model will still encounter a domain shift between:

```text
Internet dataset
        ↓
controlled laboratory image
        ↓
your actual smartphone image
```

Therefore, create your own dataset.

I would target at least:

### Stage 1

```text
300–500 images
```

### Stage 2

```text
1,000+ images
```

if possible.

Don't just photograph 1 grain.

Create scenes such as:

### Case 1 — one grain

```text
○
```

### Case 2 — two separated

```text
○       ○
```

### Case 3 — touching

```text
○○
```

### Case 4 — several touching

```text
○○○
 ○○
```

### Case 5 — partial overlap

```text
  ○
 ○○
```

### Case 6 — heavy overlap

```text
 ○○○
○○○○
 ○○○
```

### Case 7 — dense

```text
████████████████
████████████████
████████████████
```

### Case 8 — rice + FM

```text
rice rice rice
rice stone rice
rice rice wheat
```

---

# 10. Create a dedicated difficulty matrix

This is something I strongly recommend putting into the project.

Every image should be assigned difficulty metadata.

```text
density:
    low
    medium
    high
    extreme

interaction:
    isolated
    touching
    overlapping
    heavily_overlapping

occlusion:
    none
    partial
    severe

background:
    white
    black
    tray
    natural
    smartphone

lighting:
    controlled
    bright
    dark
    shadow
    uneven

foreign_matter:
    none
    stone
    seed
    plastic
    mixed
```

Then your test set can explicitly ask:

```text
How well does the model segment touching grains?

How well does it segment overlapping grains?

How well does it work at 1,000 grains?

How well does it work with foreign matter?
```

---

# 11. Annotation format

For every rice grain:

```text
class = rice
instance_id = unique
polygon/mask = exact boundary
```

For foreign matter:

```text
class = foreign_matter
subclass = stone / wheat / plastic / insect / etc.
instance_id = unique
mask = exact boundary
```

Example:

```text
Image 001

Rice:
    R001 mask
    R002 mask
    R003 mask
    R004 mask

Foreign matter:
    FM001 stone mask
    FM002 wheat mask
```

### CRITICAL

For touching grains:

```text
     A
   ████
  █████
     ████
       B
```

you **must have two masks**.

Not:

```text
one combined mask
```

This is the single most important annotation rule.

---

# 12. Train/validation/test split

Do **not** randomly split near-identical augmented images.

Instead:

```text
70% TRAIN
15% VALIDATION
15% TEST
```

But split **by original image/session/source**, not after augmentation.

Otherwise:

```text
original image
     ↓
augmentation 1 → TRAIN
augmentation 2 → TEST
```

would create data leakage.

The model would effectively see the same image during training and testing.

---

# 13. Test set must be special

Don't make the test set merely:

```text
random 15%
```

Create a **challenge test set**.

For example:

```text
TEST-01
10 isolated grains

TEST-02
100 isolated grains

TEST-03
touching grains

TEST-04
2 overlapping grains

TEST-05
5 overlapping grains

TEST-06
dense 100+ grains

TEST-07
500+ grains

TEST-08
1000+ grains

TEST-09
rice + stone

TEST-10
rice + wheat

TEST-11
rice + plastic

TEST-12
mixed foreign matter

TEST-13
dark background

TEST-14
bright background

TEST-15
shadows

TEST-16
phone camera

TEST-17
low-resolution phone image

TEST-18
partial occlusion
```

This is how we actually discover whether the model works.

---

# 14. Model architecture

I would **not blindly choose YOLO or Mask R-CNN before benchmarking**.

Train at least:

```text
Model A:
YOLO instance segmentation

Model B:
Mask R-CNN
```

Then compare them on the exact same held-out test set.

Why?

Because your requirement is not:

> fastest detector

Your requirement is:

> correctly separate every grain, especially touching and overlapping grains.

Mask R-CNN is a natural baseline for this because it directly predicts an instance mask for each detected object.

YOLO segmentation is attractive because it is faster and easier to deploy.

Recent rice research has demonstrated YOLO-based instance segmentation for touching/overlapping grains, while the recent dense-overlap work also explicitly discusses the difficulty of adhesion and overlap. ([ScienceDirect][1])

So **benchmark both rather than deciding from theory**.

---

# 15. Very important: don't resize everything blindly

This matters a LOT for your project.

Suppose the original image is:

```text
4000 × 3000
```

and contains:

```text
1000 grains
```

If you resize directly to:

```text
640 × 640
```

each grain may become tiny.

Instead investigate:

### Tiled inference

```text
4000 × 3000
       ↓
┌────────┬────────┐
│ tile 1 │ tile 2 │
├────────┼────────┤
│ tile 3 │ tile 4 │
└────────┴────────┘
```

with overlap:

```text
tile overlap = 10–25%
```

Then:

```text
tile predictions
      ↓
merge masks
      ↓
remove duplicate instances
      ↓
final image
```

This is especially important for dense scenes.

---

# 16. Augmentation

The model should see realistic variation.

Use:

```text
rotation
horizontal flip
vertical flip
scale
crop
translation
brightness
contrast
gamma
blur
noise
shadow
background variation
slight perspective
color variation
```

But don't destroy the grain geometry.

Avoid unrealistic transformations such as extreme warping unless justified.

---

# 17. Synthetic touching/overlap training

This can be VERY useful.

Start with individually segmented grains.

Then programmatically compose them:

```text
grain A
    +
grain B
    ↓
touching scene
```

Then:

```text
grain A
     +
grain B
     +
grain C
```

with controlled overlap:

```text
0%
10%
20%
30%
40%
50%
```

Now you automatically know the ground truth masks because you created them.

For example:

```text
Synthetic image

A mask = known
B mask = known
C mask = known
```

This can dramatically increase the number of difficult training examples.

But synthetic images should **supplement**, not replace, real touching/overlapping images.

---

# 18. Training curriculum

Don't immediately train everything together.

Use stages.

## Stage 1 — baseline

Train:

```text
clean isolated rice
```

Goal:

```text
model understands rice
```

---

## Stage 2 — touching

Add:

```text
touching grains
```

Goal:

```text
A + B touching
```

must become:

```text
A
B
```

not:

```text
A+B
```

---

## Stage 3 — overlap

Add:

```text
partial overlap
heavy overlap
occlusion
```

---

## Stage 4 — density

Add:

```text
50 grains
100 grains
300 grains
500 grains
1000+ grains
```

---

## Stage 5 — foreign matter

Add:

```text
rice + stone
rice + wheat
rice + plastic
rice + other seed
rice + insect
```

---

## Stage 6 — real smartphone images

Finally:

```text
your actual images
```

---

# 19. Important negative examples

The model must also see images containing:

```text
no rice
only stones
only wheat
only plastic
only seeds
empty tray
random objects
human hand
cloth
background
```

Otherwise it can learn:

> everything looks like rice.

You want:

```text
Input → no rice
Output → NO_RICE
```

rather than:

```text
Input → random objects
Output → 37 rice grains
```

---

# 20. Foreign matter architecture

I would initially test two approaches.

### Approach A

Single instance segmentation model:

```text
rice
stone
wheat
plastic
insect
...
```

This is simplest.

### Approach B

Two-stage:

```text
                 Image
                   │
             Rice segmentation
                   │
          ┌────────┴────────┐
          ▼                 ▼
        Rice              Remaining
       regions             regions
                            │
                            ▼
                    Foreign matter model
```

The agent should benchmark both.

---

# 21. Do NOT classify unresolved rice as foreign matter

This bug already appeared in your earlier classical pipeline.

Suppose:

```text
████████
████████
```

is actually two touching rice grains.

If the model cannot separate them, the system must say:

```text
UNRESOLVED_RICE_CLUSTER
```

NOT:

```text
FOREIGN MATTER
```

Use states like:

```text
RICE_INSTANCE
FOREIGN_MATTER
UNRESOLVED_RICE_CLUSTER
LOW_CONFIDENCE
```

This is extremely important for debugging.

---

# 22. Post-processing

After model prediction:

```text
raw predictions
      ↓
confidence filtering
      ↓
mask cleanup
      ↓
duplicate removal
      ↓
overlap resolution
      ↓
tile merging
      ↓
instance IDs
      ↓
final result
```

Do not use aggressive morphological operations that merge nearby grains.

---

# 23. Evaluation metrics

Do NOT evaluate only with accuracy.

Use:

### Detection

```text
Precision
Recall
F1
mAP50
mAP50:95
```

### Segmentation

```text
Mask IoU
Mask AP
AP50
AP75
mAP50:95
```

### Instance counting

```text
Predicted count
Ground-truth count
Absolute count error
Count MAE
Count percentage error
```

### Special metrics

Most importantly:

```text
Touching separation accuracy
Overlap separation accuracy
Dense-scene recall
False merge rate
False split rate
```

For example:

```text
Ground truth:

A B C D E

Prediction:

A+B C D E
```

This is a **merge error**.

Another:

```text
Ground truth:

A

Prediction:

A1 A2
```

This is a **split error**.

Your agent should explicitly measure both.

---

# 24. The model should have an uncertainty mechanism

For example:

```text
confidence >= 0.90
    → accepted

0.60–0.90
    → uncertain

<0.60
    → rejected/flagged
```

But **do not hard-code these values as truth**.

Tune them using the validation set.

---

# 25. Final inference output

For every image, return something like:

```json
{
    "rice_detected": true,
    "rice_count": 147,
    "foreign_matter_count": 4,
    "unresolved_clusters": 2,
    "instances": [
        {
            "id": 1,
            "class": "rice",
            "confidence": 0.98,
            "mask": "...",
            "bbox": [...]
        }
    ],
    "foreign_matter": [
        {
            "id": 1,
            "class": "stone",
            "confidence": 0.94,
            "mask": "..."
        }
    ]
}
```

---

# 26. What I want the AI coding agent to do

Here is the **master prompt**. You can paste this directly into Antigravity/Cursor/Codex.

---

## MASTER TRAINING PROMPT

```text
You are the ML lead for my Rice Grain Instance Segmentation project.

IMPORTANT:
This is a focused milestone.

DO NOT implement the 14 rice-quality parameters yet.
DO NOT implement official rice grading yet.
DO NOT implement chalkiness/damage/red/dehusked classification unless required as a dataset class.
DO NOT spend time building the final grading system.

Our ONLY current objective is to build and fully train a robust model that can:

1. Detect whether rice is present in an image.
2. Detect foreign matter / non-rice objects.
3. Detect every individual rice grain.
4. Generate one separate instance mask for every individual rice grain.
5. Correctly separate touching grains.
6. Correctly separate partially overlapping grains.
7. Correctly separate heavily overlapping/occluded grains as far as the visual information allows.
8. Work with sparse images and extremely dense images.
9. Work with images containing hundreds or approximately 1000+ grains.
10. Assign a unique ID to every detected grain.
11. Count the grains.
12. Flag unresolved rice clusters instead of incorrectly classifying them as foreign matter.
13. Work on realistic smartphone-style images as well as controlled images.
14. Detect common foreign matter such as stones, other seeds/cereal grains, plastic, insects, soil/clods, and other available contamination classes.

The goal is HIGH VALIDATED ACCURACY, NOT a fake claim of 100% accuracy.
Never claim 100% accuracy unless it is actually measured on a clearly defined test set.
Never fabricate training results, metrics, datasets, model weights, annotations, or test results.

==================================================
PHASE 0 — AUDIT THE EXISTING REPOSITORY
==================================================

First inspect the entire repository.

Determine:

- current frontend
- current backend
- current ML code
- current segmentation code
- current dataset directories
- existing models
- existing annotation formats
- existing tests
- existing scripts
- existing configuration
- existing inference pipeline
- existing regression cases

Do NOT assume that old classical OpenCV segmentation is sufficient.

If classical segmentation exists, keep it only as:
- baseline
- debugging/reference
- fallback if useful

The primary solution must be a TRAINED INSTANCE SEGMENTATION MODEL.

Do not destroy working code until the new system has been validated.

Create:

docs/ML_TRAINING_PLAN.md

containing the complete training plan and current repository status.

==================================================
PHASE 1 — DATASET DISCOVERY
==================================================

Find and audit publicly available datasets for:

A. rice instance segmentation
B. touching rice grains
C. overlapping rice grains
D. densely packed rice grains
E. rice grain detection/counting
F. rice + foreign matter
G. foreign object detection in rice
H. smartphone/real-world rice images

Candidate sources to investigate include:

1. Rice-Variety-Classification-System-Dataset
   Mendeley:
   https://data.mendeley.com/datasets/jcgmmpbv6g/1

2. Grainalyze Instance Segmentation Dataset
   Roboflow:
   https://universe.roboflow.com/rice-6hvoz/grainalyze/dataset/11

3. STR-900 / adhesive rice instance segmentation datasets
   Verify actual availability, license, and annotations.

4. Recent dense-overlapping rice datasets reported in academic papers.
   Verify whether the actual dataset is publicly downloadable.
   Do not claim access if it is not.

5. Other legitimate public rice instance-segmentation datasets.

6. Rice foreign-object datasets containing:
   stone
   wheat
   corn
   plastic
   insect
   soil/clod
   metal
   other foreign materials

For every candidate dataset create:

datasets/DATASET_AUDIT.md

with:

- dataset name
- URL
- DOI if available
- license
- number of images
- resolution
- number of instances if available
- annotation type
- instance segmentation or semantic segmentation
- bounding boxes or masks
- number of isolated grains
- number of touching grains
- number of overlapping grains
- density
- rice varieties
- background types
- lighting variation
- foreign matter classes
- whether raw images are available
- whether annotations are downloadable
- whether redistribution is allowed
- whether commercial/non-commercial restrictions exist
- suitability for our project
- exact download instructions

Do not include a dataset merely because a paper mentions it.

Verify the actual downloadable files.

==================================================
PHASE 2 — DATASET QUALITY AUDIT
==================================================

After downloading candidate datasets, write scripts that inspect them automatically.

For every dataset calculate:

- image count
- annotation count
- instance count
- instances/image
- image dimensions
- mask dimensions
- class distribution
- object size distribution
- percentage isolated
- percentage touching if detectable
- percentage overlapping if detectable
- density distribution
- corrupted files
- missing annotations
- invalid polygons
- duplicate images
- near-duplicate images
- empty annotations

Create dataset reports.

Do not blindly merge datasets.

Normalize annotation formats first.

==================================================
PHASE 3 — BUILD A MASTER DATASET
==================================================

Create:

data/

    raw/
    processed/
    annotations/
    train/
    val/
    test/
    challenge_test/

Use a common annotation representation.

Preferred internal representation:

COCO-style instance segmentation.

Every object must contain:

- image_id
- category_id
- segmentation mask
- bbox
- area
- iscrowd
- instance_id

Classes should initially be:

0 = rice
1 = stone
2 = wheat/other cereal
3 = plastic
4 = insect
5 = soil/clod
6 = other_foreign_matter

If the available data cannot support all foreign-matter subclasses, collapse them to:

foreign_matter

but preserve original subclass metadata where possible.

==================================================
PHASE 4 — CRITICAL INSTANCE ANNOTATION RULE
==================================================

For touching or overlapping rice grains:

EVERY PHYSICAL GRAIN MUST HAVE ITS OWN INSTANCE MASK.

Example:

Grain A touches Grain B.

Correct annotation:

A = mask 1
B = mask 2

Incorrect annotation:

A+B = one combined mask

This rule must be automatically checked wherever possible.

Create annotation visualization tools that display:
- original image
- all instance masks
- instance IDs
- bounding boxes
- class names

==================================================
PHASE 5 — ADD OUR OWN DATASET
==================================================

Create a collection procedure for our own real images.

Capture:

1. isolated grains
2. 2 grains
3. 5 grains
4. 10 grains
5. 20–50 grains
6. 100+ grains
7. 300+ grains
8. 500+ grains
9. 1000+ grains

Also deliberately capture:

- touching grains
- partially overlapping grains
- heavily overlapping grains
- dense piles
- different orientations
- rotated grains
- partially occluded grains
- broken grains
- rice mixed with foreign matter
- shadows
- bright lighting
- dark lighting
- uneven lighting
- black background
- white background
- tray background
- realistic smartphone images

Create metadata for every image:

density:
isolated / low / medium / high / extreme

interaction:
isolated / touching / overlapping / heavily_overlapping

lighting:
controlled / bright / dark / shadow / uneven

background:
black / white / tray / natural / smartphone

foreign_matter:
none / stone / wheat / plastic / insect / mixed

==================================================
PHASE 6 — SYNTHETIC DATA GENERATION
==================================================

Build a synthetic scene generator using individually segmented real rice grains.

Generate scenes containing:

- isolated grains
- touching grains
- 2-grain overlap
- 3-grain overlap
- multiple overlaps
- dense clusters
- random rotations
- scale variations
- different grain orientations
- realistic spacing
- controlled occlusion

The generator must preserve exact ground-truth instance masks.

Generate overlap levels approximately:

0%
5%
10%
20%
30%
40%
50%

Also generate different densities.

Synthetic data is supplemental.
Do not allow synthetic data to replace real validation/test images.

==================================================
PHASE 7 — DATA SPLITTING
==================================================

Create:

70% training
15% validation
15% test

BUT:

Split by original source/session/image group.

Do not place augmented versions of the same source image in different splits.

Prevent data leakage.

The challenge_test set must be kept completely separate from training.

==================================================
PHASE 8 — CHALLENGE TEST SET
==================================================

Create a manually verified challenge set containing:

A. no rice
B. one rice grain
C. two grains
D. five grains
E. ten grains
F. 50 grains
G. 100 grains
H. 300+ grains
I. 500+ grains
J. 1000+ grains

Touching:
K. 2 touching
L. 5 touching
M. many touching

Overlap:
N. 2 overlapping
O. several overlapping
P. severe overlap

Foreign matter:
Q. rice + stone
R. rice + wheat
S. rice + plastic
T. rice + insect
U. rice + mixed foreign matter

Real-world:
V. smartphone image
W. low lighting
X. shadows
Y. uneven lighting
Z. cluttered background

This test set must never be used for training.

==================================================
PHASE 9 — MODEL BENCHMARK
==================================================

Do not assume one architecture is best.

Train at least:

MODEL A:
YOLO instance segmentation

MODEL B:
Mask R-CNN

If computationally feasible, evaluate one additional modern instance segmentation baseline.

Use exactly the same:
- training split
- validation split
- challenge test split

for fair comparison.

Evaluate:

- precision
- recall
- F1
- mAP50
- mAP50:95
- mask AP
- mask IoU
- AP75
- count MAE
- count percentage error
- false merge rate
- false split rate
- touching separation rate
- overlap separation rate
- dense-scene recall

==================================================
PHASE 10 — HIGH-RESOLUTION / TILED INFERENCE
==================================================

Investigate whether full-image resizing loses small grain information.

Support tiled inference for large images.

Example:

Original:
4000x3000

Split into overlapping tiles.

Use configurable:

tile_size
tile_overlap

Run instance segmentation on each tile.

Then merge tile predictions.

Handle duplicate detections at tile boundaries.

Evaluate:

1. full-image inference
2. tiled inference

and determine which produces better grain-level segmentation.

Do not assume 640x640 is sufficient.

==================================================
PHASE 11 — TRAINING CURRICULUM
==================================================

Use staged training.

Stage 1:
clean/isolated rice

Stage 2:
touching rice

Stage 3:
overlapping rice

Stage 4:
dense rice

Stage 5:
foreign matter

Stage 6:
mixed rice + foreign matter

Stage 7:
real smartphone images

Stage 8:
fine-tuning on difficult failure cases

Keep validation/test sets fixed.

Do not contaminate validation/test data during hard-example mining.

==================================================
PHASE 12 — HARD EXAMPLE MINING
==================================================

After the first model:

Run inference on training/validation development data.

Collect failure categories:

- missed grain
- merged grains
- split grain
- bad boundary
- false rice
- false foreign matter
- missed foreign matter
- duplicate detection
- low-confidence grain
- unresolved cluster

Add difficult examples back into training ONLY from the allowed training pool.

Do not add test images.

Retrain.

Repeat until improvement plateaus.

Track every experiment.

==================================================
PHASE 13 — SPECIAL TOUCHING/OVERLAP ANALYSIS
==================================================

Create a dedicated evaluation script:

evaluate_touching.py

and:

evaluate_overlap.py

For each scene determine:

ground_truth_instances
predicted_instances

Then calculate:

missed instances
merged instances
split instances
correctly separated instances

Example:

GT = 10 grains
prediction = 9 instances

If two GT grains became one predicted mask:

false_merge = 1

If one GT grain became two predicted masks:

false_split = 1

Produce visual reports.

==================================================
PHASE 14 — FOREIGN MATTER
==================================================

Foreign matter must never be used as a garbage class for unresolved rice.

Explicit states:

RICE
FOREIGN_MATTER
UNRESOLVED_RICE_CLUSTER
LOW_CONFIDENCE

If the model sees a rice-looking cluster but cannot separate it:

classify it as:

UNRESOLVED_RICE_CLUSTER

not FOREIGN_MATTER.

This is essential for correct system behavior.

==================================================
PHASE 15 — NO-RICE DETECTION
==================================================

Create explicit negative samples:

- stones only
- wheat only
- plastic only
- empty background
- random objects
- non-rice grains
- foreign matter only

The system must be able to return:

{
    "rice_detected": false
}

when appropriate.

Do not infer rice merely because some object was detected.

==================================================
PHASE 16 — CONFIDENCE CALIBRATION
==================================================

Do not hard-code arbitrary confidence thresholds.

Use validation data to determine useful thresholds.

Investigate:

- confidence threshold
- mask quality
- duplicate removal
- overlapping instances
- low-confidence cases

Store thresholds in configuration.

==================================================
PHASE 17 — FINAL OUTPUT FORMAT
==================================================

The inference API must return:

{
    "rice_detected": true,
    "rice_count": 147,
    "foreign_matter_count": 4,
    "unresolved_cluster_count": 2,

    "rice_instances": [
        {
            "id": 1,
            "confidence": 0.98,
            "bbox": [...],
            "mask": ...
        }
    ],

    "foreign_matter": [
        {
            "id": 1,
            "class": "stone",
            "confidence": 0.94,
            "bbox": [...],
            "mask": ...
        }
    ],

    "unresolved_clusters": [
        {
            "id": 1,
            "confidence": 0.61,
            "bbox": [...],
            "mask": ...
        }
    ]
}

==================================================
PHASE 18 — VISUALIZATION
==================================================

Create a visualization function that produces:

Original image

+

individual colored masks

+

unique grain IDs

+

confidence

+

foreign matter labels

+

unresolved cluster labels

Example:

Rice #001
Rice #002
Rice #003
FM #001 Stone
FM #002 Wheat
Cluster #001

The visualization must make it immediately obvious whether touching grains were separated correctly.

==================================================
PHASE 19 — COUNT VALIDATION
==================================================

For every challenge image compare:

ground truth count
predicted count

Create a table:

Image
GT Count
Predicted Count
Absolute Error
Percentage Error
Merged Instances
Split Instances
Missed Instances
False Positives

Also create aggregate metrics.

==================================================
PHASE 20 — MODEL SELECTION
==================================================

Do NOT choose the model based only on:

- FPS
- mAP50
- model size

The primary priority is:

1. correct instance separation
2. mask quality
3. touching separation
4. overlap separation
5. dense-scene performance
6. foreign matter detection
7. count accuracy
8. robustness
9. inference speed

Report all metrics.

Do not produce a subjective "best model" claim without showing the measurements.

==================================================
PHASE 21 — EXPERIMENT TRACKING
==================================================

Every experiment must record:

experiment ID
date
dataset version
dataset size
model architecture
pretrained weights
image size
batch size
epochs
optimizer
learning rate
augmentation
training time
GPU
best checkpoint
validation metrics
test metrics
touching metrics
overlap metrics
dense metrics
foreign matter metrics

Save:

experiments/<experiment_id>/

with:

config
metrics
plots
confusion matrix
sample predictions
failure examples
best checkpoint path

==================================================
PHASE 22 — FINAL REGRESSION TEST
==================================================

After training the final model, run all regression cases.

Minimum cases:

1. no rice
2. one grain
3. two grains
4. five grains
5. ten grains
6. isolated grains
7. touching grains
8. overlapping grains
9. dense grains
10. 100+ grains
11. 500+ grains
12. 1000+ grains
13. rice + stone
14. rice + wheat
15. rice + plastic
16. rice + insect
17. mixed foreign matter
18. smartphone image
19. dark image
20. shadow image

Save:

reports/final_regression_report.json

and visual examples.

==================================================
PHASE 23 — FRONTEND/API INTEGRATION
==================================================

Only after the ML model is validated:

Connect the trained model to the existing application.

Frontend should allow:

Upload image
    ↓
Run model
    ↓
Show original image
    ↓
Show segmented image
    ↓
Show individual masks
    ↓
Show IDs
    ↓
Show rice count
    ↓
Show foreign matter count
    ↓
Show unresolved clusters
    ↓
Allow toggling:
    masks
    IDs
    bounding boxes
    confidence

Do NOT add the 14 parameter calculations yet.

==================================================
PHASE 24 — PERFORMANCE
==================================================

Benchmark:

CPU inference
GPU inference

and, where applicable:

full image
tiled image

Measure:

inference time
preprocessing time
postprocessing time
total time
memory usage

Do not sacrifice segmentation quality merely to obtain high FPS.

==================================================
PHASE 25 — FAILURE ANALYSIS
==================================================

Before declaring completion, generate:

reports/failure_analysis.md

Group failures into:

1. missed grains
2. merged grains
3. split grains
4. poor boundaries
5. false rice
6. false foreign matter
7. missed foreign matter
8. dense-scene failures
9. severe-overlap failures
10. low-resolution failures
11. lighting failures
12. background failures

For every failure include:

image
ground truth
prediction
failure category
likely cause
possible solution

==================================================
PHASE 26 — IMPORTANT RULES
==================================================

RULE 1:
Never fabricate metrics.

RULE 2:
Never claim 100% accuracy.

RULE 3:
Never fabricate a dataset.

RULE 4:
Never claim a dataset contains touching/overlapping masks unless verified.

RULE 5:
Never train on the challenge test set.

RULE 6:
Never classify an unresolved rice cluster as foreign matter.

RULE 7:
Never combine touching grains into one ground-truth instance.

RULE 8:
Never use semantic segmentation when instance segmentation is required.

RULE 9:
Do not depend entirely on classical watershed segmentation.

Watershed may be retained as a fallback/baseline, but the primary solution must be learned instance segmentation.

RULE 10:
Do not implement the 14 parameters yet.

RULE 11:
Do not implement official rice grading yet.

RULE 12:
Do not modify the challenge test images based on model failures.

RULE 13:
Keep every experiment reproducible.

RULE 14:
If a dataset cannot be legally/publicly downloaded, document it but do not pretend that it is available.

RULE 15:
If a dataset's annotations are insufficient for instance segmentation, do not use it as an instance-mask training dataset.

==================================================
PHASE 27 — DEFINITION OF DONE
==================================================

This milestone is complete only when:

[ ] Rice/no-rice detection works
[ ] Individual rice instances are detected
[ ] Every detected grain has an individual mask
[ ] Touching grains are evaluated and separated
[ ] Overlapping grains are evaluated and separated
[ ] Dense scenes are evaluated
[ ] 100+ grain scenes are evaluated
[ ] 500+ grain scenes are evaluated if available
[ ] 1000+ grain scenes are evaluated if available
[ ] Foreign matter is detected
[ ] Foreign matter is not confused with unresolved rice clusters
[ ] Unique IDs are generated
[ ] Counts are generated
[ ] Challenge test set exists
[ ] Test set is never used for training
[ ] Mask metrics are reported
[ ] Count metrics are reported
[ ] Merge/split metrics are reported
[ ] Failure analysis exists
[ ] Best checkpoint is saved
[ ] Inference API works
[ ] Frontend visualization works
[ ] Complete training documentation exists

Only after all of the above is complete should we move on to the 14 rice-quality parameters.
```

---

# 27. One thing I would change from the prompt above

I would tell the agent **not to spend days searching for a mythical perfect dataset**.

Your project doesn't need:

> one dataset containing everything.

Instead:

```text
Dataset A
general rice instances
       +
Dataset B
touching/overlap
       +
Dataset C
dense grains
       +
Dataset D
foreign matter
       +
YOUR DATA
smartphone/domain adaptation
       +
synthetic difficult scenes
       ↓
MASTER DATASET
       ↓
TRAIN
```

That is much more realistic.

The recent literature itself supports this direction: dense overlapping rice is treated as a specialized instance-segmentation problem, while other work uses controlled multi-grain images and instance segmentation for morphology. ([ScienceDirect][1])

---

# 28. The biggest thing I want you to understand

**The model does not become good at overlapping grains just because we choose Mask R-CNN or YOLO.**

The real equation is:

```text
Good model
     +
Correct instance annotations
     +
Enough touching examples
     +
Enough overlap examples
     +
Enough dense examples
     +
Different imaging conditions
     +
Real smartphone examples
     +
Hard-negative examples
     +
Proper evaluation
     +
Iterative retraining
     =
Robust segmentation
```

If you train on:

```text
2,000 images
all containing nicely separated grains
```

you can have an excellent-looking mAP score and still get:

```text
████████████
= 1 giant grain
```

when 10 grains are touching.

That's exactly why your **training dataset composition and challenge test set are more important right now than blindly picking an architecture**.

And for your specific requirement of **~1000+ grains in one image**, I would make **dense-scene/tiled inference a first-class part of the system**, not an afterthought. Recent dense-rice work demonstrates that this is a distinct technical challenge rather than ordinary object detection. ([ScienceDirect][1])

[1]: https://www.sciencedirect.com/science/article/pii/S0263224126029349?utm_source=chatgpt.com "Edge-deployable instance segmentation framework for densely overlapping embryo-retained rice grains in industrial processing - ScienceDirect"
[2]: https://data.mendeley.com/datasets/jcgmmpbv6g/1?utm_source=chatgpt.com "Rice-Variety-Classification-System-Dataset - Mendeley Data"
[3]: https://xuebaozk.haut.edu.cn/en/article/doi/10.16433/j.1673-2383.202501110002?utm_source=chatgpt.com "An improved YOLOv8 model for segmentation of adhesive rice images"
[4]: https://universe.roboflow.com/rice-6hvoz/grainalyze/dataset/11?utm_source=chatgpt.com "Grainalyze Instance Segmentation Model (v11, Final) by rice"
[5]: https://www.mdpi.com/2077-0472/15/13/1354?utm_source=chatgpt.com "YOLOv-MA: A High-Precision Foreign Object Detection Algorithm for Rice"
[6]: https://www.sciencedirect.com/science/article/pii/S0956713525006735?utm_source=chatgpt.com "Deep learning for extraneous material detection in a rice processing factory - ScienceDirect"
