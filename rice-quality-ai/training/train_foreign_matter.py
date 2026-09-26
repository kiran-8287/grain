"""
Foreign Matter Detection Model Setup and Training Pipeline.

Integrates YOLO11n fine-tuning for non-rice foreign matter objects:
Classes:
0: stone
1: inorganic
2: organic
3: other_foreign_matter

Records dataset manifests, class mapping, mAP50, and training metadata.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: "stone",
    1: "inorganic",
    2: "organic",
    3: "other_foreign_matter"
}


def setup_foreign_matter_pipeline():
    """Configure model directory and manifests for foreign matter YOLO training."""
    out_dir = PROJECT_ROOT / "models" / "foreign_matter"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Save class mapping
    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(CLASS_MAPPING, f, indent=2)

    # Dataset YAML config for YOLO training
    dataset_yaml_content = f"""
path: {PROJECT_ROOT / 'data' / 'processed' / 'foreign_matter'}
train: images/train
val: images/val
names:
  0: stone
  1: inorganic
  2: organic
  3: other_foreign_matter
"""
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml_content.strip())

    manifest = {
        "dataset_name": "rice_foreign_matter_combined",
        "sources": [
            "Rice-Quality 3 Foreign Matter",
            "Agricultural Non-Grain Contaminants"
        ],
        "taxonomy": CLASS_MAPPING,
        "splits": {
            "train": 0.8,
            "val": 0.2
        },
        "target_model": "yolo11n.pt"
    }
    with open(manifest_dir / "foreign_matter_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    metadata = {
        "model_name": "foreign_matter_yolo11n",
        "version": "1.0.0",
        "architecture": "YOLO11n",
        "classes": CLASS_MAPPING,
        "dataset_yaml": str(yaml_path),
        "status": "ready_for_training",
        "note": "Ultralytics YOLO11n fine-tuning pipeline initialized. When external dataset downloaded, run: yolo detect train data=dataset.yaml model=yolo11n.pt",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Initialized foreign matter YOLO pipeline in {out_dir}")
    return metadata


if __name__ == "__main__":
    setup_foreign_matter_pipeline()
