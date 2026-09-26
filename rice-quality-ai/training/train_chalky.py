"""
Chalky Grain Classification Training Script.

Uses Logistic Regression on extracted LAB, GLCM, and brightness features.
Follows strict train/val/test splitting, feature scaling, and comprehensive evaluation
(Precision, Recall, F1, ROC-AUC, Confusion Matrix).
Saves model, scaler, feature schema, class labels, and metadata to models/chalky/.
"""

import json
import logging
import csv
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import cv2
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from ml.texture import extract_chalky_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean_l_star",
    "std_l_star",
    "mean_a_star",
    "mean_b_star",
    "bright_pixel_fraction",
    "chalky_pixel_fraction",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]


def generate_synthetic_chalky_data(n_samples: int = 500, random_state: int = 42):
    """
    Generate realistic synthetic feature distributions based on rice literature
    when raw image dataset is not locally downloaded.
    
    Class 0 (Normal / Translucent): Lower L*, lower bright fraction, lower chalky fraction, higher homogeneity.
    Class 1 (Chalky / Opaque): Higher L*, higher bright fraction, higher chalky fraction, higher contrast.
    """
    np.random.seed(random_state)
    n_per_class = n_samples // 2

    # Normal grains
    l_star_norm = np.random.normal(loc=65.0, scale=4.0, size=n_per_class)
    std_l_norm = np.random.normal(loc=6.0, scale=1.5, size=n_per_class)
    a_star_norm = np.random.normal(loc=-1.0, scale=1.0, size=n_per_class)
    b_star_norm = np.random.normal(loc=12.0, scale=2.0, size=n_per_class)
    bright_frac_norm = np.clip(np.random.beta(a=1.5, b=8.0, size=n_per_class), 0, 1)
    chalky_frac_norm = np.clip(np.random.beta(a=1.0, b=10.0, size=n_per_class), 0, 1)
    contrast_norm = np.random.normal(loc=12.0, scale=3.0, size=n_per_class)
    homo_norm = np.clip(np.random.normal(loc=0.85, scale=0.05, size=n_per_class), 0, 1)
    energy_norm = np.clip(np.random.normal(loc=0.35, scale=0.05, size=n_per_class), 0, 1)
    corr_norm = np.clip(np.random.normal(loc=0.80, scale=0.05, size=n_per_class), -1, 1)

    X_norm = np.column_stack([
        l_star_norm, std_l_norm, a_star_norm, b_star_norm,
        bright_frac_norm, chalky_frac_norm,
        contrast_norm, homo_norm, energy_norm, corr_norm,
    ])
    y_norm = np.zeros(n_per_class, dtype=int)

    # Chalky grains
    l_star_chalky = np.random.normal(loc=82.0, scale=5.0, size=n_per_class)
    std_l_chalky = np.random.normal(loc=11.0, scale=2.5, size=n_per_class)
    a_star_chalky = np.random.normal(loc=-0.5, scale=1.2, size=n_per_class)
    b_star_chalky = np.random.normal(loc=10.0, scale=2.5, size=n_per_class)
    bright_frac_chalky = np.clip(np.random.beta(a=5.0, b=2.0, size=n_per_class), 0, 1)
    chalky_frac_chalky = np.clip(np.random.beta(a=4.5, b=2.5, size=n_per_class), 0, 1)
    contrast_chalky = np.random.normal(loc=24.0, scale=5.0, size=n_per_class)
    homo_chalky = np.clip(np.random.normal(loc=0.65, scale=0.08, size=n_per_class), 0, 1)
    energy_chalky = np.clip(np.random.normal(loc=0.20, scale=0.05, size=n_per_class), 0, 1)
    corr_chalky = np.clip(np.random.normal(loc=0.65, scale=0.08, size=n_per_class), -1, 1)

    X_chalky = np.column_stack([
        l_star_chalky, std_l_chalky, a_star_chalky, b_star_chalky,
        bright_frac_chalky, chalky_frac_chalky,
        contrast_chalky, homo_chalky, energy_chalky, corr_chalky,
    ])
    y_chalky = np.ones(n_per_class, dtype=int)

    X = np.vstack([X_norm, X_chalky])
    y = np.concatenate([y_norm, y_chalky])
    return X, y


def _load_chalky_manifest(data_dir: Path):
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Supply real labeled grain images and masks; "
            "synthetic feature vectors are not accepted for model training."
        )

    label_ids = {"clean": 0, "normal": 0, "not_chalky": 0, "chalky": 1}
    feature_rows, labels, source_groups = [], [], []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = csv.DictReader(manifest_file)
        required = {"image_path", "mask_path", "label", "source_group"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(
                "Chalky manifest requires image_path, mask_path, label, source_group columns"
            )
        for row in rows:
            label_name = row["label"].strip().lower()
            if label_name not in label_ids or not row["source_group"].strip():
                raise ValueError(f"Invalid chalky label or source_group in row: {row}")
            image_path = (data_dir / row["image_path"]).resolve()
            mask_path = (data_dir / row["mask_path"]).resolve()
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image_bgr is None or mask is None or image_bgr.shape[:2] != mask.shape:
                raise ValueError(f"Could not load matching image/mask pair: {image_path}")
            features = extract_chalky_features(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mask)
            feature_rows.append([features[name] for name in FEATURE_NAMES])
            labels.append(label_ids[label_name])
            source_groups.append(row["source_group"].strip())

    if not feature_rows or set(labels) != {0, 1}:
        raise ValueError("The real chalky dataset must contain both clean and chalky grains")
    return np.asarray(feature_rows, dtype=float), np.asarray(labels), np.asarray(source_groups)


def _group_split(X, y, groups, test_size, seed):
    for random_state in range(seed, seed + 100):
        train_indices, test_indices = next(
            GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            ).split(X, y, groups)
        )
        if set(y[train_indices]) == {0, 1} and set(y[test_indices]) == {0, 1}:
            return train_indices, test_indices
    raise ValueError(
        "Cannot create group-disjoint train/validation/test splits containing both "
        "classes. Add more independent clean and chalky source groups."
    )


def _select_threshold(y_true, probabilities):
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scored = [
        (f1_score(y_true, probabilities >= threshold, zero_division=0), threshold)
        for threshold in candidates
    ]
    return float(max(scored, key=lambda item: (item[0], item[1]))[1])


def _evaluate(y_true, probabilities, threshold):
    predictions = probabilities >= threshold
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_clean = int(matrix[0].sum())
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "clean_false_positive_rate": float(matrix[0, 1] / true_clean) if true_clean else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def train_chalky_model(data_dir: Path = None):
    """Train from real labeled grains, keeping source images in a single split."""
    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "processed" / "chalky"
    X, y, groups = _load_chalky_manifest(data_dir)

    train_val_indices, test_indices = _group_split(X, y, groups, 0.15, 42)
    train_indices, val_indices = _group_split(
        X[train_val_indices], y[train_val_indices], groups[train_val_indices], 0.1765, 142
    )
    train_indices = train_val_indices[train_indices]
    val_indices = train_val_indices[val_indices]
    if set(y[train_indices]) != {0, 1} or set(y[val_indices]) != {0, 1}:
        raise ValueError("Train and validation splits must each contain clean and chalky grains")

    scaler = StandardScaler().fit(X[train_indices])
    classifier = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    classifier.fit(scaler.transform(X[train_indices]), y[train_indices])
    val_probabilities = classifier.predict_proba(scaler.transform(X[val_indices]))[:, 1]
    threshold = _select_threshold(y[val_indices], val_probabilities)
    test_probabilities = classifier.predict_proba(scaler.transform(X[test_indices]))[:, 1]
    test_metrics = _evaluate(y[test_indices], test_probabilities, threshold)
    validation_metrics = _evaluate(y[val_indices], val_probabilities, threshold)

    out_dir = PROJECT_ROOT / "models" / "chalky"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    with (out_dir / "feature_schema.json").open("w", encoding="utf-8") as schema_file:
        json.dump({"features": FEATURE_NAMES, "n_features": len(FEATURE_NAMES), "scaling": "StandardScaler", "mask_usage": "instance-mask-only GLCM and color features"}, schema_file, indent=2)

    metadata = {
        "model_name": "chalky_logistic_regression",
        "version": "2.0.0",
        "algorithm": "LogisticRegression(class_weight='balanced')",
        "classes": {"0": "normal", "1": "chalky"},
        "dataset": str(data_dir),
        "source_group_split": True,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "test_samples": int(len(test_indices)),
        "decision_threshold": threshold,
        "threshold_source": "validation F1; independent source-group split",
        "validation_metrics": validation_metrics,
        "metrics": test_metrics,
        "probability_calibration": "Raw Logistic Regression scores; calibration not established.",
        "status": "trained",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    logger.info("Chalky test metrics: %s", test_metrics)
    return metadata


if __name__ == "__main__":
    train_chalky_model()
