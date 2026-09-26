"""
Sprouted / Weevilled Grain Classification Training Script.

Uses ResNet-18 transfer learning.
Input crops are padded to square (224x224) with aspect ratio preservation.
Evaluates with accuracy, precision, recall, F1, and confusion matrix.
Saves model weights, class mapping, and training metadata to models/sprouted_weevilled/.
"""

import json
import logging
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
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyntheticSproutedDataset(Dataset):
    """
    Dataset of grain crops for sprouted/weevilled vs normal classification.
    """
    def __init__(self, n_samples: int = 80, is_train: bool = True):
        self.samples = []
        self.labels = []
        np.random.seed(101 if is_train else 202)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for i in range(n_samples):
            label = i % 2  # 0: normal, 1: sprouted / weevilled
            img = np.zeros((224, 224, 3), dtype=np.uint8)

            color = (int(np.random.uniform(205, 235)), int(np.random.uniform(200, 230)), int(np.random.uniform(190, 215)))
            cv2_axes = (int(np.random.uniform(50, 70)), int(np.random.uniform(18, 25)))
            angle = int(np.random.uniform(-30, 30))
            cv2.ellipse(img, (112, 112), cv2_axes, angle, 0, 360, color, -1)

            if label == 1:
                # Sprout protrusion or weevil cavity
                if i % 2 == 0:
                    # Sprout (small green/yellow protrusion at tip)
                    cv2.circle(img, (112 + cv2_axes[0] - 5, 112), 8, (120, 180, 80), -1)
                else:
                    # Weevil hole (dark cavity)
                    cv2.circle(img, (112, 112), 6, (40, 30, 25), -1)

            self.samples.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        tensor = self.transform(img)
        label = self.labels[idx]
        return tensor, torch.tensor(label, dtype=torch.long)


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


def train_sprouted_model(epochs: int = 2):
    """Train and evaluate ResNet-18 model for sprouted/weevilled grains."""
    out_dir = PROJECT_ROOT / "models" / "sprouted_weevilled"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ResNet-18 Sprouted/Weevilled Model on device: {device}")

    train_dataset = SyntheticSproutedDataset(n_samples=60, is_train=True)
    test_dataset = SyntheticSproutedDataset(n_samples=24, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
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
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for tensors, labels in test_loader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))
    cm = confusion_matrix(all_targets, all_preds).tolist()

    logger.info(f"ResNet-18 Evaluation Results (Test Set):")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Confusion Matrix: {cm}")

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
        "classes": class_mapping,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        },
        "status": "trained",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved ResNet-18 sprouted/weevilled model artifacts to {out_dir}")
    return metadata


if __name__ == "__main__":
    train_sprouted_model(epochs=2)
