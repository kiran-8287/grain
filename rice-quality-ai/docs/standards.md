# Official Standards Engine & Limitations

## 1. Primary Reference Specification
The system uses the current **Government of India Kharif Marketing Season 2026-27 (KMS 2026-27) Uniform Specification for Grade A and Common Raw Milled Rice** as its official reference.

### Official Reference Limits for Raw Milled Rice

| Ref Parameter | Grade A Limit | Common Limit | Assessment Basis in Government Lab | Image Pipeline Equivalent |
|---|---|---|---|---|
| **Broken** | 25.0% max | 25.0% max | Weight percentage (sieving) | Kernel count/area estimate ($< 0.75 L_{\text{whole}}$) |
| **Foreign Matter** | 0.5% max | 0.5% max | Weight percentage | Bounding box area / object count |
| **Damaged / Slightly Damaged** | 3.0% max | 3.0% max | Visual inspection by weight | VGG-19 CNN on grain crops |
| **Discoloured** | 3.0% max | 3.0% max | Visual inspection by weight | Adaptive CIELAB $\Delta E$ |
| **Chalky** | 5.0% max | 5.0% max | Visual inspection by weight | Logistic Regression on LAB/GLCM features |
| **Red Grains** | 3.0% max | 3.0% max | Visual inspection by weight | Red cuticle coverage fraction $\ge 25\%$ |
| **Admixture of Lower Class** | 6.0% max | Not Applicable | Manual varietal separation | Robust Mahalanobis outlier detection |
| **Dehusked Grains** | 13.0% max | 13.0% max | Chemical/alkaline staining test | Visual bran coverage proxy |
| **Moisture** | 14.0% max | 14.0% max | Oven/moisture meter | **Excluded — Not measurable from RGB** |

---

## 2. No Invented Official Grades

A critical requirement of this project is that the system **DOES NOT invent an official Government of India grade** (such as "Grade A Certified").

The system explicitly provides TWO distinct outputs:

### Output A: Image-Based Standard Screening
A parameter-by-parameter screening comparison:
- Observed value
- KMS 2026-27 reference limit
- Difference
- Status: `WITHIN REFERENCE LIMIT`, `EXCEEDS REFERENCE LIMIT`, or `NOT ASSESSABLE`
- Basis: Explicit notice stating *"Image-based screening; not an official laboratory test."*

### Output B: Formal Official Grade Status
A scientific statement explaining why an official grade is not established from the image alone:
> *"Formal Grade A / Common determination is not established from this image alone. Government procurement standards mandate laboratory procedures that cannot be fully replicated with an RGB image, including weight-based determinations, moisture measurement, and official chemical staining for dehusked kernels."*

---

## 3. Paddy Classification L/B Safeguard
The Government of India specification notes an L/B ratio threshold of $\ge 2.5$ for classifying paddy varieties as Grade A. **The system explicitly does not transfer this paddy classification rule to milled rice**. L/B ratio is reported as an independent geometric parameter.
