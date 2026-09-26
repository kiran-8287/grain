"""
Sprouted / Weevilled Grain Classification Training Script.

Uses ResNet-18 transfer learning.
Input crops are padded to square (224x224) with aspect ratio preservation.
Evaluates with accuracy, precision, recall, F1, and confusion matrix.
Saves model weights, class mapping, and training metadata to models/sprouted_weevilled/.
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from ml.preprocessing import masked_grain_square_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ManifestSproutedDataset(Dataset):
    """Real labeled grain/mask pairs; source_group prevents image leakage."""

    LABEL_IDS = {
        "normal": 0,
        "clean": 0,
        "no": 0,
        "sprouted_weevilled": 1,
        "sprouted": 1,
        "weevilled": 1,
        "yes": 1,
    }

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Supply real labeled grain images and masks; "
                "synthetic crops are not accepted for model training."
            )

        self.samples = []
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            rows = csv.DictReader(manifest_file)
            required = {"image_path", "mask_path", "label", "source_group"}
            if not rows.fieldnames or not required.issubset(rows.fieldnames):
                raise ValueError(
                    "Sprouted manifest requires image_path, mask_path, label, source_group columns"
                )
            for row in rows:
                label_name = row["label"].strip().lower()
                group = row["source_group"].strip()
                if label_name not in self.LABEL_IDS or not group:
                    raise ValueError(f"Invalid label or source_group in row: {row}")
                self.samples.append((
                    (self.data_dir / row["image_path"]).resolve(),
                    (self.data_dir / row["mask_path"]).resolve(),
                    self.LABEL_IDS[label_name],
                    group,
                ))

        self.labels = np.asarray([sample[2] for sample in self.samples], dtype=int)
        self.groups = np.asarray([sample[3] for sample in self.samples])
        if set(self.labels) != {0, 1}:
            raise ValueError("The real sprouted dataset must contain normal and positive grains")
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, label, _ = self.samples[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None or image_bgr.shape[:2] != mask.shape:
            raise ValueError(f"Could not load matching image/mask pair: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop = masked_grain_square_crop(image_rgb, mask, target_size=224)
        if crop is None:
            raise ValueError(f"Grain mask is empty: {mask_path}")
        return self.transform(crop), torch.tensor(label, dtype=torch.long)


def _group_split(labels, groups, test_size, seed):
    for random_state in range(seed, seed + 100):
        train_indices, test_indices = next(
            GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            ).split(np.zeros(len(labels)), labels, groups)
        )
        if set(labels[train_indices]) == {0, 1} and set(labels[test_indices]) == {0, 1}:
            return train_indices, test_indices
    raise ValueError(
        "Cannot create group-disjoint train/validation/test splits containing both classes. "
        "Add more independent clean and positive source groups."
    )


def _select_threshold(labels, positive_probabilities):
    candidates = np.unique(np.concatenate(([0.0], positive_probabilities, [1.0])))
    scores = [
        (f1_score(labels, positive_probabilities >= threshold, zero_division=0), threshold)
        for threshold in candidates
    ]
    return float(max(scores, key=lambda item: (item[0], item[1]))[1])


def _evaluate(labels, positive_probabilities, threshold):
    predictions = positive_probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    clean_count = int(matrix[0].sum())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, positive_probabilities)),
        "clean_false_positive_rate": float(matrix[0, 1] / clean_count) if clean_count else None,
        "confusion_matrix": matrix.tolist(),
    }


def build_resnet18_classifier(num_classes: int = 2):
    """Build ResNet-18 with fine-tuning head."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        logger.info("Loaded pretrained ResNet-18 ImageNet weights.")
        for param in model.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.warning(f"Could not download ImageNet weights ({e}); building standard ResNet-18.")
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_sprouted_model(epochs: int = 2, data_dir: Path = None):
    """Train/evaluate from real labeled grains with source-group disjoint splits."""
    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "processed" / "sprouted_weevilled"
    dataset = ManifestSproutedDataset(data_dir)
    train_val_indices, test_indices = _group_split(dataset.labels, dataset.groups, 0.15, 42)
    train_relative, val_relative = _group_split(
        dataset.labels[train_val_indices], dataset.groups[train_val_indices], 0.1765, 142
    )
    train_indices = train_val_indices[train_relative]
    val_indices = train_val_indices[val_relative]

    out_dir = PROJECT_ROOT / "models" / "sprouted_weevilled"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ResNet-18 Sprouted/Weevilled Model on device: {device}")

    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = build_resnet18_classifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tensors.size(0)
        epoch_loss = running_loss / len(train_dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    model.eval()
    val_probabilities = []
    val_targets = []
    with torch.no_grad():
        for tensors, labels in val_loader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            val_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            val_targets.extend(labels.numpy())
    decision_threshold = _select_threshold(
        np.asarray(val_targets), np.asarray(val_probabilities)
    )

    test_probabilities = []
    test_targets = []
    with torch.no_grad():
        for tensors, labels in test_loader:
            outputs = model(tensors.to(device))
            test_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            test_targets.extend(labels.numpy())
    test_metrics = _evaluate(
        np.asarray(test_targets), np.asarray(test_probabilities), decision_threshold
    )
    validation_metrics = _evaluate(
        np.asarray(val_targets), np.asarray(val_probabilities), decision_threshold
    )
    logger.info("ResNet-18 test metrics: %s", test_metrics)

    model_path = out_dir / "model.pth"
    torch.save(model, model_path)

    class_mapping = {
        "0": "normal",
        "1": "sprouted_weevilled"
    }
    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    metadata = {
        "model_name": "sprouted_resnet18",
        "version": "1.0.0",
        "backbone": "resnet18",
        "dataset": str(data_dir),
        "source_group_split": True,
        "classes": class_mapping,
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "decision_threshold": decision_threshold,
        "threshold_source": "validation F1; independent source-group split",
        "validation_metrics": validation_metrics,
        "metrics": test_metrics,
        "probability_calibration": "Softmax scores are not calibrated.",
        "status": "trained",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved ResNet-18 sprouted/weevilled model artifacts to {out_dir}")
    return metadata


if __name__ == "__main__":
    train_sprouted_model(epochs=2)
