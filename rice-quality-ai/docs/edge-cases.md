# Edge Case Handling and Robustness

This document details how the rice quality analysis system handles complex real-world edge cases.

---

## 1. Grain Sample Size Range ($N = 0$ to $N > 5000$)

| Sample Count | System Behavior | User Interface Indication |
|---|---|---|
| **$N = 0$** | Halts processing early. No fake grain measurements are generated. | Displays `"No rice grains detected. Please upload an image containing rice grains."` |
| **$N = 1$** | Fully segments and analyzes the single grain. Computes all geometric dimensions, classifications, and color profiles. Broken kernel distribution reference marked as `undetermined`. | Shows `"Sample size: 1 grain"` warning: *"Individual-grain analysis is possible, but sample-level quality percentages are not representative of a larger rice lot."* Percentage labeled as `"Observed sample fraction"`. |
| **$N = 2$ to $9$** | Segments each grain, assigns individual IDs `#1..#N`, computes geometry and defect labels. Broken status marked undetermined if whole reference cannot be established. | Shows small-sample advisory banner: *"Observed sample size is below statistical threshold (30 grains)."* |
| **$N = 10$ to $29$** | Segments each grain. Admixture analysis runs with limited confidence. | Small-sample advisory displayed. |
| **$N = 30$ to $999$** | Full statistical population modeling, robust whole-kernel length estimation, and Mahalanobis admixture outlier detection. | Full confidence dashboard. |
| **$N \ge 1000$** | High throughput memory-safe processing, vectorized feature extraction, and cached model inference. | Full dashboard with processing time benchmark. |

---

## 2. No Fixed 12-Megapixel Gate

The system **does not reject images with less than 12 megapixels**.
- Total image resolution is uncoupled from grain resolution. A 2 MP macro close-up of 5 grains provides hundreds of pixels across each grain, whereas a 12 MP camera from 2 meters away may yield grains occupying only 10 pixels.
- Quality is evaluated **after segmentation** via:
  1. Median grain pixel area
  2. Bounding box width/height
  3. Laplacian blur variance ($> 50$)
  4. Illumination uniformity

---

## 3. Touching and Overlapping Grains
- **Detection**: Merged instances are detected using contour concavity, aspect ratio anomalies, and area ratio $> 2.5 \times \text{median area}$.
- **Splitting**: Watershed transform on Euclidean distance transform peaks splits touching grains along natural pinch points.
- **Uncertain Instances**: When a merged blob cannot be split with high confidence, it is tagged as `segmentation_quality: "uncertain"` and flagged in the UI rather than counted as a single giant grain.

---

## 4. Arbitrary Backgrounds
- Thresholding utilizes adaptive Gaussian thresholding and Otsu inverted background comparison.
- Evaluates foreground-to-background ratio to automatically invert polarity on black vs white vs textured backgrounds.

---

## 5. Calibration vs No Calibration
- **Mode A (Calibrated)**: ArUco marker detection or manual scale factor provided. Converts length, breadth, and area to metric millimeters ($mm$).
- **Mode B (Uncalibrated)**: No reference marker detected. Measurements reported strictly in pixels ($px$). L/B ratio is reported normally since it is dimensionless. The UI clearly displays: *"Metric calibration unavailable — measurements shown in pixels."*
