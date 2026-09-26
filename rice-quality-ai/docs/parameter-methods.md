# Quality Parameter Methodologies (14 Parameters)

This document provides complete technical specifications for how each of the exactly 14 project parameters is calculated, its scientific basis, mathematical formulation, and limitations.

---

### 1. Broken Grain
- **Scope**: Per-Grain
- **Definition**: Kernels whose length is below three-fourths (0.75) of the robust whole-kernel length reference.
- **Why this method**: Simple mean of all grains is corrupted when broken grains exist in the sample. An iterative median estimator filters out broken grains and converges on true whole-kernel length.
- **Algorithm / Formula**:
  1. $L_{\text{whole}}^{(0)} = \text{median}(\{L_i\})$
  2. Repeat 3 times:
     $$\text{Selected} = \{L_i \mid L_i > 0.85 \cdot L_{\text{whole}}^{(k)}\}$$
     $$L_{\text{whole}}^{(k+1)} = \text{median}(\text{Selected})$$
  3. $\text{Broken}_i = L_i < 0.75 \cdot L_{\text{whole}}$
  4. Small broken pieces ($L_i < 0.25 \cdot L_{\text{whole}}$) are tracked separately.
- **Small-Sample Behavior**: For $N \le 2$, returns `broken_label: "undetermined"` because a reference population cannot be computed from $\le 2$ grains.
- **Scientific Classification**: Official concept with image-count proxy; configured limit is historical/reference only, not verified as current KMS 2026-27.

---

### 2. Damaged / Slightly Damaged Grain
- **Scope**: Per-Grain
- **Definition**: Kernels with visible discolouration, insect bites, or fungal lesions affecting kernel integrity.
- **Why ML**: Visual patterns of rot, cracks, and mold are spatially complex and vary across varieties. Transfer learning on deep representations captures these textures better than brittle color slicing.
- **Architecture**: VGG-19 transfer learning fine-tuned on grain crops.
- **Input Preprocessing**: Bounding box crop padded to square ($224 \times 224$) via `pad_to_square` preserving aspect ratio (never stretched).
- **Fallback**: Color variance and dark-spot ratio heuristic when weights are absent.
- **Scientific Classification**: Literature-Supported ML.

---

### 3. Discoloured Grain
- **Scope**: Per-Grain
- **Definition**: Kernels displaying uniform or localized deviation in color from the typical translucent white/cream rice kernel.
- **Why Non-ML**: CIE $L^*a^*b^*$ DeltaE measures perceptual color distance directly and adapts to ambient illumination variations by comparing against the median grain profile of the image.
- **Formula**:
  $$\Delta E = \sqrt{(L_i^* - L_{\text{ref}}^*)^2 + (a_i^* - a_{\text{ref}}^*)^2 + (b_i^* - b_{\text{ref}}^*)^2}$$
  Kernel is discoloured if $\Delta E > 15.0$ over $> 20\%$ of grain area.
- **Scientific Classification**: Engineering Heuristic; configured limit is historical/reference only, not verified as current KMS 2026-27.

---

### 4. Chalky Grain
- **Scope**: Per-Grain
- **Definition**: Grain-level classification of the opaque, milky-white chalky appearance; the classifier does not segment chalky subregions or infer brittleness from RGB.
- **Why ML**: Grain-level color and internal texture features can support this distinction when trained and evaluated on labeled rice grains. Brightness alone is not chalkiness.
- **Input Features**:
  - $L^*$, $a^*$, $b^*$ mean and standard deviation
  - Mask-only brightness distribution and descriptive bright-pixel fraction
  - Mask-only GLCM texture; co-occurrence pairs require both pixels inside the instance mask
  - GLCM Contrast, Homogeneity, Energy, and Correlation
- **Model**: Grain-level Logistic Regression is enabled only for artifacts trained on real labeled images and masks. Current synthetic-feature artifact is disabled; chalkiness is `Undetermined` until a validated model is available.
- **Scientific Classification**: Experimental grain-level ML; raw model scores are not calibrated probabilities.

---

### 5. Red Grain
- **Scope**: Per-Grain
- **Definition**: Kernels having more than one-fourth of their surface area covered with red cuticle/bran layer.
- **Method**: Color distribution in LAB ($a^* > 132$) and HSV ($H \in [0, 20] \cup [165, 180]$ with moderate saturation).
- **Formula**:
  $$\text{Red Fraction} = \frac{\sum \text{pixels}_{\text{red}}}{\sum \text{pixels}_{\text{grain}}}$$
  Flagged as red if fraction $\ge 0.25$.
- **Scientific Classification**: Official Concept Threshold implemented via Image-Based Color Proxy.

---

### 6. Dehusked Grain
- **Scope**: Per-Grain
- **Definition**: Kernels retaining brown pericarp/bran over more than one-fourth of the surface.
- **Important Scientific Limitation**: The official government test for dehusked grain relies on alkaline chemical staining in a laboratory. An RGB camera cannot perform chemical staining. Therefore, the output is explicitly documented and labeled: `"Image-based visual estimate — not equivalent to the official staining test."`
- **Method**: HSV brown hue band ($H \in [10, 30], S \in [40, 255], V \in [40, 200]$) measuring surface coverage fraction.
- **Scientific Classification**: Experimental Visual Proxy.

---

### 7. Immature / Shrunken / Shrivelled
- **Scope**: Per-Grain
- **Definition**: Grains that are underdeveloped, thin, or shrunken before maturity.
- **Method**: Relative breadth ratio ($< 0.70 \times$ median breadth), low area ratio ($< 0.50 \times$ median area), low solidity ($< 0.85$), and high L/B ($> 5.0$).
- **Scientific Classification**: Experimental Geometry Proxy.

---

### 8. Sprouted / Weevilled
- **Scope**: Per-Grain
- **Definition**: Grains exhibiting germinated shoots at the embryo tip or bored holes/tunnels caused by insects (weevils).
- **Method**: ResNet-18 transfer learning on padded crops ($224 \times 224$).
- **Fallback**: Laplacian texture variance and deep cavity detection.
- **Scientific Classification**: Experimental ML; checked-in checkpoint was trained on synthetic demonstration crops and is not validated on real rice grains.

---

### 9. Foreign Matter
- **Scope**: Sample-Level (Evaluated across the entire image)
- **Definition**: All organic (chaff, weed seeds, straw) and inorganic (stones, mud, sand) matter other than rice kernels.
- **Method**: Full-image object detection using fine-tuned YOLO11n.
- **Weight Limitation**: Official limits are by weight ($\le 0.5\%$). Camera images measure area and count. Output is explicitly labeled: `"Image-based count/area estimate — not official laboratory weight percentage."`
- **Scientific Classification**: Literature-Supported ML.

---

### 10. Admixture of Lower Class
- **Scope**: Sample-Level (NOT assigned to individual grains)
- **Definition**: Rice of another variety/class mixed into the sample (e.g., Common rice mixed into Grade A).
- **Method**:
  1. Filter out broken grains and uncertain segmentations.
  2. Compute multivariate geometry vector: $[L, B, L/B, \text{Area}, \text{Solidity}]$.
  3. Calculate robust covariance and Mahalanobis distance:
     $$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
  4. Kernels with $D_M > 3.0$ are counted as statistical admixture outliers.
- **Scientific Classification**: Statistical Distribution Proxy; configured limit is historical/reference only, not verified as current KMS 2026-27.

---

### 11. Length
- **Scope**: Per-Grain
- **Method**: Fitted ellipse major axis dimension cross-checked against `cv2.minAreaRect`.
- **Unit**: Reported in millimeters when calibrated (ArUco or manual), otherwise reported strictly in pixels.
- **Scientific Classification**: Direct Physical Measurement.

---

### 12. Breadth
- **Scope**: Per-Grain
- **Method**: Fitted ellipse minor axis dimension.
- **Unit**: Millimeters (calibrated) or pixels.
- **Scientific Classification**: Direct Physical Measurement.

---

### 13. Length/Breadth (L/B) Ratio
- **Scope**: Per-Grain
- **Formula**:
  $$L/B = \frac{\text{Length}}{\text{Breadth}}$$
- **Scale Invariance**: $L/B$ is dimensionless and identical in pixel units and metric units.
- **Division-by-Zero Guard**: Handled safely with fallback to `None` if breadth $\le 0$.
- **Scientific Classification**: Direct Geometric Ratio.

---

### 14. Total Count
- **Scope**: Sample-Level
- **Definition**: Number of accepted, valid segmented rice grain instances.
- **Diagnostics**: Concurrently reports raw detected instances, accepted grains, uncertain/merged instances, and rejected noise blobs.
- **Scientific Classification**: Count Metric.
