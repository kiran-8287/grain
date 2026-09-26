"""Legacy synthetic VGG-19 demo; real-data training is not implemented."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class SyntheticGrainCropDataset(Dataset):
    """
    Synthetic illustrations retained only to document the legacy demo source.

    These crops are not real rice data and must not be used to evaluate or
    publish model performance.
    """
    def __init__(self, n_samples: int = 120, is_train: bool = True):
        self.samples = []
        self.labels = []
        np.random.seed(42 if is_train else 99)

        # ImageNet normalization
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for i in range(n_samples):
            label = i % 2  # 0: normal, 1: damaged
            # Create a 224x224 canvas
            img = np.zeros((224, 224, 3), dtype=np.uint8)

            # Draw grain ellipse in the center
            if label == 0:
                # Normal clean translucent white/cream grain
                color = (int(np.random.uniform(210, 240)), int(np.random.uniform(205, 235)), int(np.random.uniform(195, 220)))
                cv2_axes = (int(np.random.uniform(50, 75)), int(np.random.uniform(18, 26)))
                angle = int(np.random.uniform(-45, 45))
                import cv2
                cv2.ellipse(img, (112, 112), cv2_axes, angle, 0, 360, color, -1)
            else:
                # Damaged grain: dark spots, cracks, brown discolouration
                color = (int(np.random.uniform(160, 200)), int(np.random.uniform(130, 170)), int(np.random.uniform(90, 130)))
                cv2_axes = (int(np.random.uniform(45, 70)), int(np.random.uniform(16, 25)))
                angle = int(np.random.uniform(-45, 45))
                import cv2
                cv2.ellipse(img, (112, 112), cv2_axes, angle, 0, 360, color, -1)
                # Add damage spots
                for _ in range(np.random.randint(2, 6)):
                    sx = int(np.random.uniform(90, 134))
                    sy = int(np.random.uniform(90, 134))
                    sr = int(np.random.uniform(3, 8))
                    cv2.circle(img, (sx, sy), sr, (int(np.random.uniform(30, 80)), int(np.random.uniform(20, 50)), int(np.random.uniform(10, 30))), -1)

            self.samples.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        tensor = self.transform(img)
        label = self.labels[idx]
        return tensor, torch.tensor(label, dtype=torch.long)


def build_vgg19_classifier(num_classes: int = 2):
    """Build VGG-19 with fine-tuning head."""
    # Use pretrained VGG-19 weights if accessible, otherwise initialize architecture
    try:
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        logger.info("Loaded pretrained VGG-19 ImageNet weights.")
        # Freeze feature backbone
        for param in model.features.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.warning(f"Could not download ImageNet weights ({e}); building standard VGG-19.")
        model = models.vgg19(weights=None)

    # Replace top classifier head
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_damaged_model(epochs: int = 3):
    """Refuse to replace the experimental checkpoint with synthetic-demo results."""
    manifest_path = PROJECT_ROOT / "data" / "processed" / "damaged" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Synthetic crop training is disabled; "
            "provide real labeled rice-grain data before retraining."
        )
    raise NotImplementedError(
        "The audited VGG-19 demo trainer does not yet implement real-manifest "
        "training. Do not publish or overwrite artifacts until that trainer is reviewed."
    )


if __name__ == "__main__":
    train_damaged_model(epochs=2)
