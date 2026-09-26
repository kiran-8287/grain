"""
Rice-vs-non-rice gate training.

Trains a lightweight binary classifier that distinguishes valid rice grains from
hard negatives such as stones, pebbles, dirt, leaves, plastic, and other debris.
The model is intentionally small and feature-based so it can be used as a fast,
robust gate before the full grain analysis pipeline is executed.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean_l_star",
    "chroma",
    "aspect_ratio",
    "surface_roughness",
    "mean_gray",
    "background_gap_ratio",
]


def _sample_rice_features(n_samples: int, rng: np.random.Generator):
    mean_l = rng.normal(72.0, 7.0, size=n_samples)
    chroma = rng.normal(9.0, 4.0, size=n_samples)
    aspect_ratio = rng.normal(2.8, 0.8, size=n_samples)
    roughness = rng.uniform(0.01, 0.045, size=n_samples)
    mean_gray = rng.normal(130.0, 38.0, size=n_samples)
    gap_ratio = rng.uniform(0.02, 0.18, size=n_samples)
    return np.column_stack([mean_l, chroma, aspect_ratio, roughness, mean_gray, gap_ratio])


def _sample_non_rice_features(n_samples: int, rng: np.random.Generator):
    mean_l = rng.normal(55.0, 18.0, size=n_samples)
    chroma = rng.normal(26.0, 16.0, size=n_samples)
    aspect_ratio = rng.normal(1.35, 0.55, size=n_samples)
    roughness = rng.uniform(0.04, 0.15, size=n_samples)
    mean_gray = rng.normal(100.0, 55.0, size=n_samples)
    gap_ratio = rng.uniform(0.08, 0.42, size=n_samples)
    return np.column_stack([mean_l, chroma, aspect_ratio, roughness, mean_gray, gap_ratio])


def generate_hard_negative_dataset(n_samples_per_class: int = 1200, random_state: int = 42):
    """Create a synthetic but realistic rice vs hard-negative dataset."""
    rng = np.random.default_rng(random_state)
    X_rice = _sample_rice_features(n_samples_per_class, rng)
    X_non_rice = _sample_non_rice_features(n_samples_per_class, rng)

    X = np.vstack([X_rice, X_non_rice]).astype(np.float32)
    y = np.concatenate([
        np.ones(len(X_rice), dtype=int),
        np.zeros(len(X_non_rice), dtype=int),
    ])
    return X, y


def train_rice_gate_model():
    """Train and save the binary rice/no-rice gate model."""
    out_dir = PROJECT_ROOT / "models" / "rice_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_hard_negative_dataset(n_samples_per_class=1200, random_state=42)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "false_positive_rate_non_rice": round(
            float(np.sum((y_pred == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1)), 4
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info("Rice gate evaluation: %s", metrics)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    class_mapping = {"0": "non_rice", "1": "rice"}
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    feature_schema = {
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "scaling": "StandardScaler",
        "label_order": ["non_rice", "rice"],
    }
    with open(out_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    metadata = {
        "model_name": "rice_gate_logistic_regression",
        "version": "1.0.0",
        "algorithm": "LogisticRegression(class_weight='balanced')",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "features": FEATURE_NAMES,
        "classes": class_mapping,
        "metrics": metrics,
        "status": "trained",
        "hard_negatives": [
            "stone",
            "pebble",
            "dirt",
            "sand",
            "leaf",
            "plastic",
            "other_seed",
        ],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved rice gate artifacts to %s", out_dir)
    return metadata


if __name__ == "__main__":
    train_rice_gate_model()
