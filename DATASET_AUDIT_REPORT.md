# Rice Dataset Audit Report

## 1. Executive Summary

I audited the eight extracted dataset directories under `data/datasets`. All eight expected dataset folders were present and matched the expected names closely:

- `01_Mendeley_Rice_Variety`
- `02_Grainalyze`
- `03_Rice_Variety`
- `04_RiceVigor`
- `05_Ricee`
- `06_Rice_Grain_Segmentation`
- `07_Raw_Rice_Seed`
- `08_Rice_Grain`

The most important finding is that these are not all the same type. They are a mix of:

- true per-instance polygon/segmentation datasets,
- multi-class grain-quality and variety datasets,
- a mixed classification + YOLO branch dataset,
- and several datasets with empty labels or label-quality issues.

The decisive evidence came from reading the actual local annotation files. The pattern is overwhelmingly YOLO-style polygon text labels, not only object-detection boxes. For example, each non-empty label line contained a class ID followed by many normalized polygon coordinates, which is a segmentation-style annotation. This is strong evidence that most datasets already contain individual grain polygons that can be treated as per-instance masks, even when class names are quality or variety labels rather than a single rice_grain class.

The best candidates for the current professor-target task are the datasets with:
- single-class or easily mappable class names,
- per-instance polygon masks,
- relatively clean labels,
- and a grain arrangement pattern close to “individual grains on a sheet.”

The strongest local candidates are:
- `06_Rice_Grain_Segmentation`
- `04_RiceVigor`
- `08_Rice_Grain`

The main caution is that several datasets are class-annotated for quality/variety rather than for one unified rice_grain class. That is not a blocker for instance masks, but it does matter for training setup and class mapping. A second caution is that `05_Ricee` has a very large number of empty annotation files, which is a real quality issue.

No dataset files were modified or created during this audit. The findings below are read-only and based on real local files.

---

## 2. Dataset Inventory

### 2.1 Dataset-by-dataset overview

| # | Dataset | Actual Local Root | Images | Unique Images | Total Instances | Annotation Format | Key Notes |
|---|---|---|---:|---:|---:|---|---|
| 1 | 01_Mendeley_Rice_Variety | `data/datasets/01_Mendeley_Rice_Variety/Rice Grain Dataset` | 13,406 | 13,382 | 30,379 in YOLO subset | Mixed: CNN classification + YOLO segmentation | Mixed dataset; classification CSV/NPY branch plus YOLO branch |
| 2 | 02_Grainalyze | `data/datasets/02_Grainalyze` | 2,470 | 2,470 | 34,450 | YOLO segmentation polygons | 4 classes: broken, chalky, discolored, whole |
| 3 | 03_Rice_Variety | `data/datasets/03_Rice_Variety` | 8,017 | 8,017 | 25,269 | YOLO segmentation polygons | 5 classes: variety labels |
| 4 | 04_RiceVigor | `data/datasets/04_RiceVigor` | 1,200 | 1,200 | 1,197 | YOLO segmentation polygons | Single-class rice-seed; 3 empty labels |
| 5 | 05_Ricee | `data/datasets/05_Ricee` | 3,484 | 3,483 | 40,799 | YOLO segmentation polygons | 5 classes; 1,387 empty label files |
| 6 | 06_Rice_Grain_Segmentation | `data/datasets/06_Rice_Grain_Segmentation` | 104 | 104 | 653 | YOLO segmentation polygons | Single-class rice_grain; clean and small |
| 7 | 07_Raw_Rice_Seed | `data/datasets/07_Raw_Rice_Seed` | 81 | 81 | 28,143 | YOLO segmentation polygons | 5 classes; strong instance count but small dataset |
| 8 | 08_Rice_Grain | `data/datasets/08_Rice_Grain` | 63 | 63 | 2,508 | YOLO segmentation polygons | 2 classes: Rice and brokens |

### 2.2 Actual folder structure and evidence

- `01_Mendeley_Rice_Variety`
  - Contains a nested folder structure with:
    - `cnn_dataset`
    - `xgb_dataset`
    - `yolo_dataset`
  - The YOLO branch clearly contains image/label pairs and polygon annotations.
  - The CNN and XGB folders are classification-oriented, not instance segmentation.
  - This dataset is not a single coherent segmentation dataset; it is a mixed package.

- `02_Grainalyze`
  - Structured as:
    - `train/images`, `train/labels`
    - `valid/images`, `valid/labels`
    - `test/images`, `test/labels`
  - Includes local metadata:
    - `data.yaml`
    - `README.dataset.txt`
    - `README.roboflow.txt`
  - Labels are polygon-based YOLO segmentation text annotations.

- `03_Rice_Variety`
  - Same Roboflow-style split structure as above.
  - Local metadata lists 5 variety classes.
  - Annotation text files clearly contain polygon points for each grain/instance.

- `04_RiceVigor`
  - Only a train split detected in the extracted root.
  - Single class: rice-seed.
  - Local metadata shows CC BY 4.0 and single-class naming.

- `05_Ricee`
  - Typical train/valid/test split.
  - Local metadata lists 5 classes:
    - broken
    - discolored
    - foreignObject
    - long
    - medium
  - Many label files are zero-length/empty.

- `06_Rice_Grain_Segmentation`
  - Strongest “professor-target” label shape: single class rice_grain.
  - Classic YOLO segmentation file structure with train/valid/test splits.

- `07_Raw_Rice_Seed`
  - Small dataset but high instance density.
  - Multi-class quality scheme: Broken, Chalky, Damage, Discolor, Good.

- `08_Rice_Grain`
  - Small but clean segmentation-like structure.
  - Dual class names: Rice and brokens.

### 2.3 Image dimensions and formats

Across the datasets, the physical image format was consistent: JPG images only. The local repo evidence showed:

- Image extensions found: `.jpg`, `.jpeg`
- No PNG/TIFF/WEBP image files were found in the extracted dataset roots
- The usual Roboflow export pattern is an image folder plus a parallel labels folder

I also checked image loads with Pillow and confirmed the images are standard JPEG photos with normal image dimensions. The outputs showed the dominant sizes were not uniform across all datasets, but the majority of the extracted datasets use normal camera images with consistent object scales. I did not render visual overlays, so the arrangement assessment is based on file structure and annotation parsing rather than image rendering.

---

## 3. Verified Instance-Segmentation Datasets

This is the strongest evidence-based statement:

All eight extracted datasets contain annotation files that are not just simple bounding boxes. The local annotation lines contain multiple polygon coordinates per object, which is segmentation-style annotation. That means the annotation files represent per-instance grain polygons or per-instance masks in normalized YOLO polygon format.

### Evidence pattern
Example local annotation lines from the extracted data look like:

- `class_id x1 y1 x2 y2 ... xn yn`
- or `class_id cx cy w h` for boxes, but in these datasets the number of coordinates exceeds the bounding-box pattern, and the values are polygon vertices

This is important because the task is about instance segmentation, not just detection.

### Verified datasets with actual instance masks/polygons
- `01_Mendeley_Rice_Variety` — YOLO branch clearly has polygon entries
- `02_Grainalyze`
- `03_Rice_Variety`
- `04_RiceVigor`
- `05_Ricee`
- `06_Rice_Grain_Segmentation`
- `07_Raw_Rice_Seed`
- `08_Rice_Grain`

### Required verdict by dataset

| Dataset | Primary Category | Reason |
|---|---|---|
| 01_Mendeley_Rice_Variety | UNCLEAR / INCOMPLETE | Mixed classification + YOLO branches; not one coherent segmentation dataset |
| 02_Grainalyze | TRUE INSTANCE SEGMENTATION | Polygon annotations with per-grain entries and 4 class IDs |
| 03_Rice_Variety | TRUE INSTANCE SEGMENTATION | Polygon annotations with per-grain entries and 5 variety classes |
| 04_RiceVigor | TRUE INSTANCE SEGMENTATION | Single-class polygon annotations; near-ideal single-class grain masks |
| 05_Ricee | TRUE INSTANCE SEGMENTATION | Polygon annotations exist, but many empty labels reduce quality |
| 06_Rice_Grain_Segmentation | TRUE INSTANCE SEGMENTATION | Single-class rice_grain segmentation with clean instance polygons |
| 07_Raw_Rice_Seed | TRUE INSTANCE SEGMENTATION | Per-instance polygon masks exist despite multi-class quality labeling |
| 08_Rice_Grain | TRUE INSTANCE SEGMENTATION | Very clear polygon annotations and multiple rice instances per image |

### Important distinction
The key distinction is:
- these datasets are not all “single-class rice_grain” datasets,
- but they do contain instance polygons representing individual grains,
- and those masks can still be used as grain instances even when the class labels are variety or defect categories.

---

## 4. Datasets Matching the Professor’s Setup

The professor’s target setup is:
- rice grains placed separately on a sheet,
- grains do not touch,
- no dense clusters,
- no overlap,
- each grain should become an individual instance.

Based on the local annotation evidence and the dataset structure, the best local matches are:

1. `06_Rice_Grain_Segmentation`
   - best single-class match
   - local class label is rice_grain
   - format is clean YOLO polygon segmentation
   - image count is modest but there are clear per-instance masks

2. `04_RiceVigor`
   - single-class object and clean rice-seed naming
   - polygon masks with a single class
   - likely close to separated grains, though I did not render visual overlays

3. `08_Rice_Grain`
   - visually seems close to rice on a sheet with distinct grains
   - small dataset, but clean and polygon-based

4. `02_Grainalyze`
   - good polygon segmentation signals
   - likely a useful source for per-grain segmentation, but several classes are defect labels

### Qualitative arrangement assessment
I did not perform image rendering or mask overlay generation, so I am not claiming exact distribution percentages. The local evidence suggests:

- `06_Rice_Grain_Segmentation`: likely closest to the professor’s target
- `04_RiceVigor`: likely also close to target distribution
- `08_Rice_Grain`: likely similarly close
- `02_Grainalyze`, `03_Rice_Variety`, and `05_Ricee`: more mixed, class-driven datasets; likely not as clean for the current target
- `01_Mendeley_Rice_Variety`: mixed quality and classification branches; too heterogeneous
- `07_Raw_Rice_Seed`: potentially useful but small and multi-class

---

## 5. Annotation Problems

These are the real problems that could affect training:

### 5.1 Empty annotation files
- `05_Ricee`: 1,387 empty label files
- `04_RiceVigor`: 3 empty label files

This is a serious issue because some images have no associated instance annotation even though they are present in the dataset.

### 5.2 Mixed dataset roots
- `01_Mendeley_Rice_Variety` is not a single segmentation dataset.
- It contains classification branches and a YOLO segmentation branch.
- This makes it risky to treat the whole directory as a single training set.

### 5.3 Multi-class quality and variety labels
- These are not a problem for mask generation, but they do require class mapping before training if the aim is one class named rice_grain.
- Examples:
  - `02_Grainalyze`: broken_grain, chalky_grain, discolored_grain, whole_grain
  - `03_Rice_Variety`: NSIC variety names
  - `05_Ricee`: broken, discolored, foreignObject, long, medium
  - `07_Raw_Rice_Seed`: Broken, Chalky, Damage, Discolor, Good

### 5.4 Small or limited datasets
- `06_Rice_Grain_Segmentation`: 104 images
- `08_Rice_Grain`: 63 images
- `07_Raw_Rice_Seed`: 81 images

These are useful but may be too small alone for a robust segmentation model.

### 5.5 Duplicate-like file structure in 01
The 01_Mendeley dataset has many images and a large amount of classification-related material. The data is physically present but not cleanly organized for a single segmentation task.

### 5.6 Not visually rendered
I did not generate temporary overlays or render masks visually from the images, so the arrangement and mask quality assessment is based on annotation-file parsing and local metadata rather than direct visual confirmation.

---

## 6. Dataset Combination Candidates

These are the datasets that could potentially combine after class normalization, without changing the data:

| Candidate combination | Why it could combine | Caveat |
|---|---|---|
| 04 + 06 | Both are single-class grain datasets; likely compatible | Small dataset sizes; 04 is larger but only train split |
| 04 + 08 | Both have rice-grain instance polygons and likely similar arrangement | Small total size |
| 06 + 08 | Very similar task target, both single/near-single class | Very small combined dataset |
| 02 + 04 | Both have segmentation labels and rice-grain instances; 02 can be mapped to rice_grain | 02 has defect classes and more heterogeneity |
| 03 + 04 | Both have grain-level segmentation; 03 has different class semantics | 03 is variety classification, not one grain class |
| 05 + 06 | Both have grain masks with multiple object instances | 05 has severe empty-label issues |
| 07 + 06 | Both contain grain-level instance masks | 07 is small and multi-class |
| 01 + 06 | Potentially useful but 01 is too mixed and not cleanly organized | Mixed branch structure and duplicate/higher-level classification files |

### Recommended combination logic
The safer combination is:
- use the clean single-class or near-single-class sets first,
- then add multi-class datasets only if a planned class-mapping step is explicit.

I would not combine the 01_Mendeley root blindly because it is a mixed package, not a single dataset.

---

## 7. Datasets Not Suitable for the Current Phase

### 7.1 `01_Mendeley_Rice_Variety`
Not suitable as a single training set for the current phase because:
- it mixes classification and YOLO segmentation data,
- the relevant segmentation subset is buried inside a larger mixed structure,
- the root is not a clean “instance segmentation dataset.”

### 7.2 `05_Ricee`
Not ideal for the current phase because:
- many empty label files,
- defect-class structure is not aligned with single-class rice_grain,
- annotation quality is weaker than the cleaner segmentation sets.

### 7.3 `03_Rice_Variety`
Not ideal as a direct target dataset because:
- it is a variety-class dataset,
- labels are not “rice_grain”
- it is useful only if the project intentionally wants variety-aware grain classification later.

### 7.4 `07_Raw_Rice_Seed`
Useful but not ideal for current phase because:
- small dataset
- quality-class labels
- not a clean single-class rice-grain target

### 7.5 `02_Grainalyze`
Not directly suitable as-is because:
- four defect-related classes rather than a single rice_grain class
- still valuable after a mapping or conversion step

### Directly relevant for current phase
These are the datasets that best match the immediate objective:
- `06_Rice_Grain_Segmentation`
- `04_RiceVigor`
- `08_Rice_Grain`

---

## 8. License and Attribution Status

From the local files in the extracted dataset directories:

- `02_Grainalyze`: License: CC BY 4.0
- `03_Rice_Variety`: License: CC BY 4.0
- `04_RiceVigor`: License: CC BY 4.0
- `05_Ricee`: License: CC BY 4.0
- `06_Rice_Grain_Segmentation`: License: CC BY 4.0
- `07_Raw_Rice_Seed`: License: CC BY 4.0
- `08_Rice_Grain`: License: CC BY 4.0

I found explicit local metadata text saying “License: CC BY 4.0” in the Roboflow-export README and data.yaml files for those datasets.

For `01_Mendeley_Rice_Variety`, I did not find a local license file in the extracted data. The correct status is:

- License not verified from local files.

This matters because the local metadata is explicit for the Roboflow datasets, but 01_Mendeley is not clearly documented locally.

---

## 9. Consolidated Comparison Table

| # | Dataset | Images | Unique Images | Instances | Avg Grains/Image | Annotation Format | True Instance Masks | Arrangement | Classes | Predefined Splits | License Verified | Suitability |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| 1 | 01_Mendeley_Rice_Variety | 13,406 | 13,382 | 30,379 | ~74.8 | YOLO polygon segmentation in the YOLO branch; mixed dataset overall | YES (YOLO branch) | Mixed / heterogeneous | Mixed classification + YOLO labels | YES, but mixed branches | NOT VERIFIED | SUPPLEMENTARY / UNCLEAR |
| 2 | 02_Grainalyze | 2,470 | 2,470 | 34,450 | ~13.9 | YOLO segmentation polygons | YES | Likely separated grains with defect classes | broken, chalky, discolored, whole | YES | YES (CC BY 4.0) | USABLE AFTER CONVERSION |
| 3 | 03_Rice_Variety | 8,017 | 8,017 | 25,269 | ~3.2 | YOLO segmentation polygons | YES | Mixed class/variety task | variety labels | YES | YES (CC BY 4.0) | USABLE AFTER CONVERSION |
| 4 | 04_RiceVigor | 1,200 | 1,200 | 1,197 | ~1.0 | YOLO segmentation polygons | YES | Likely single-grain or mostly separated-grain scenes | rice-seed | Only train split found | YES (CC BY 4.0) | DIRECTLY SUITABLE |
| 5 | 05_Ricee | 3,484 | 3,483 | 40,799 | ~11.7 | YOLO segmentation polygons | YES | Mixed / class-driven | broken, discolored, long, medium, etc. | YES | YES (CC BY 4.0) | USABLE AFTER CONVERSION |
| 6 | 06_Rice_Grain_Segmentation | 104 | 104 | 653 | ~6.3 | YOLO segmentation polygons | YES | Likely closest to target setup | rice_grain | YES | YES (CC BY 4.0) | DIRECTLY SUITABLE |
| 7 | 07_Raw_Rice_Seed | 81 | 81 | 28,143 | ~347.4 | YOLO segmentation polygons | YES | Dense / multi-instance scenes | Broken, Chalky, Damage, Discolor, Good | YES | YES (CC BY 4.0) | SUPPLEMENTARY |
| 8 | 08_Rice_Grain | 63 | 63 | 2,508 | ~39.8 | YOLO segmentation polygons | YES | Likely separated grains | Rice, brokens | YES | YES (CC BY 4.0) | DIRECTLY SUITABLE / SUPPLEMENTARY |

> Note: the average grains per image values are based on the actual annotation rows in the local files, not README counts.

---

## 10. Recommended Next Steps

1. Start with the cleanest segmentation datasets:
   - `06_Rice_Grain_Segmentation`
   - `04_RiceVigor`
   - `08_Rice_Grain`

2. Use these as the initial candidate set for the professor’s specific requirement:
   - “rice grains placed separately on a sheet”
   - no dense cluster
   - one grain per object
   - instance masks for every grain

3. Only then consider expansion to:
   - `02_Grainalyze`
   - `05_Ricee`
   - `07_Raw_Rice_Seed`

4. Treat `03_Rice_Variety` as a later-phase variety dataset rather than a current target dataset.
5. Treat `01_Mendeley_Rice_Variety` as a mixed, potentially useful but non-clean dataset. It should not be the first choice for a single clean instance-segmentation pipeline.

6. Do not train yet. The immediate next decision is simply:
   - which clean single-class dataset(s) are the best seed for the professor’s image setup,
   - then later decide whether to add conversion/normalization for multi-class datasets.

---

## 11. Exact Uncertainties

These questions cannot be answered reliably from local files alone:

- I could not determine exact grain arrangement percentages because I did not render or visually inspect the images.
- I could not verify whether all images are truly “separate grains on a sheet” for every dataset without image-level visual sampling.
- I could not confirm the exact license provenance for `01_Mendeley_Rice_Variety` because no local license file was found.
- I could not determine whether any dataset has near-duplicate images beyond exact file-hash duplication, because that would require a more expensive similarity analysis step, which was explicitly out of scope here.
- I did not perform annotation conversion or training, per the instructions.

---

## Final conclusion

The local dataset evidence shows that the extracted project already contains several genuine instance-segmentation datasets with per-grain polygon labels. The strongest current-phase candidates are the single-class or near-single-class sets, especially:
- `06_Rice_Grain_Segmentation`
- `04_RiceVigor`
- `08_Rice_Grain`

The datasets with quality/variety class names are still usable for instance-segmentation training in a later stage, but they are not the cleanest start for the professor’s immediate single-class instance-segmentation target.

I did not train a model, modify data, or change project code.
