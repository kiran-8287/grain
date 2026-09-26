"""Compile saved model metadata without claiming to evaluate checkpoints."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_report():
    models_dir = PROJECT_ROOT / "models"
    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_note": "This file compiles saved artifact metadata; it does not run evaluation. Metrics are populated only by each model's training/evaluation pipeline.",
        "models": {},
        "summary": {}
    }

    subdirs = ["rice_gate", "chalky", "damaged", "sprouted_weevilled", "foreign_matter", "segmentation"]
    for sub in subdirs:
        meta_file = models_dir / sub / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                data = json.load(f)
            report["models"][sub] = data
        else:
            report["models"][sub] = {
                "status": "not_yet_trained",
                "fallback_active": True
            }

    out_file = models_dir / "evaluation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report compiled and written to {out_file}")
    return report


if __name__ == "__main__":
    generate_evaluation_report()
