# Viva & Oral Examination Explanations Guide

This document is prepared for explaining the engineering design choices, mathematical foundations, and scientific rigor of this system to a Teaching Assistant (TA) or Academic Examiner.

---

### Q1: Why not use Machine Learning for all 14 parameters?
**Answer**:
Machine learning should only be used where linear/deterministic mathematical formulations fail.
- **Pure Geometry (Length, Breadth, L/B Ratio, Area)**: These are deterministic spatial quantities best computed via Moments, fitted ellipses, and Minimum Area Bounding Rectangles. Applying ML here introduces unnecessary latency, opacity, and approximation error.
- **Broken Grain**: The definition ($< 0.75 \times \text{whole kernel}$) is purely mathematical. The challenge is estimating the whole-kernel reference when broken grains are present. An iterative median estimator converges deterministically without needing training data.
- **Discolouration**: CIELAB $\Delta E$ is the international standard for perceptual color difference. Comparing against the image's median grain profile adapts to varying illumination automatically.
- **Damaged & Sprouted Grains**: Fungal decay, insect boring holes, and sprouts exhibit complex spatial textures and irregular morphology where deep CNN feature hierarchies (VGG-19, ResNet-18) significantly outperform heuristic rules.
- **Chalkiness**: Brightness alone is not a chalky signal. The grain-level feature extractor computes LAB statistics and GLCM pairs only within the instance mask. The checked-in model used synthetic features, so inference abstains until a real labeled grain corpus is available.

---

### Q2: Why does the system refuse to assign an official Government of India Grade?
**Answer**:
Scientific honesty. The configured limits are historical/reference values; the current KMS 2026-27 primary document has not been independently verified in this repository. In addition, an RGB image cannot fulfill required official measurements:
1. **Weight Percentages**: Official standards mandate weight percentages (e.g. broken grains $\le 25\%$ by weight). An optical camera measures surface area and instance counts, not grain density or mass.
2. **Dehusked Chemical Staining**: The official test for dehusked kernels requires alkaline chemical dye staining to reveal residual bran. An RGB camera can only provide a visual bran-color coverage proxy.

Therefore, the system provides **Image-Based Standard Screening** against historical/reference limits, labels observed image fractions separately from official weight-based limits, and suppresses compliance conclusions for small samples or unreliable images.

---

### Q3: Why is there no fixed 12-Megapixel image rejection threshold?
**Answer**:
Total image megapixels is not equivalent to pixels-per-grain.
- A 2-megapixel macro photograph of 1 grain yields over 100,000 pixels across that grain, providing exceptional detail for defect inspection.
- Conversely, a 12-megapixel wide-angle photo of 2,000 grains might yield only 30 pixels per grain.
The system therefore rejects images only when genuinely unreadable, and measures quality **after segmentation** via median grain pixel dimensions and Laplacian blur variance.

---

### Q4: How are touching and overlapping grains handled?
**Answer**:
1. Merged blobs are identified using area ratios ($> 2.5 \times$ median area), aspect ratios, and contour concavity.
2. Watershed segmentation on the Euclidean distance transform splits instances at constriction points.
3. If an instance cannot be split with high confidence, it is flagged as `uncertain` and reported in diagnostics rather than silently counted as one giant grain.

---

### Q5: How is single-grain ($N=1$) analysis handled without distorting sample statistics?
**Answer**:
For $N=1$, the single grain is fully segmented, measured, and classified. However:
1. The broken kernel reference is marked `undetermined` because a population reference requires multiple grains.
2. A prominent warning informs the user: *"Individual-grain analysis is possible, but sample-level quality percentages are not representative of a larger rice lot."*
3. Percentages are labeled as `"Observed sample fraction"` rather than batch quality.
