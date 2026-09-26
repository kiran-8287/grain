"""Legacy synthetic feature demo for the rice gate; training is disabled.

The checked-in gate behavior is retained, but the available feature generator is
not image data and cannot establish real-image performance. Replace this with a
source-grouped, real-image feature manifest before retraining or reporting metrics.
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
    """Refuse to overwrite the functioning gate with synthetic feature metrics."""
    manifest_path = PROJECT_ROOT / "data" / "processed" / "rice_gate" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Real labeled object crops/features are required; "
            "synthetic gate training is disabled."
        )
    raise NotImplementedError(
        "A reviewed real-image rice gate training/evaluation pipeline is not implemented. "
        "Existing gate inference artifacts are left untouched."
    )


if __name__ == "__main__":
    train_rice_gate_model()
