"""Checks that provenance registries do not overstate local evidence."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_PARAMETERS = {
    "Broken",
    "Damaged",
    "Discoloured",
    "Chalky",
    "Red Grain",
    "Dehusked",
    "Immature / Shrunken / Shriveled",
    "Sprouted / Weevilled",
    "Foreign Matter",
    "Admixture of Lower Class Grains",
    "Length",
    "Breadth",
    "L/B Ratio",
    "Total Count",
}


def _load_json(relative_path):
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_parameter_registry_has_exactly_fourteen_complete_entries():
    parameters = _load_json("configs/parameters.json")["parameters"]
    required_fields = {
        "name",
        "type",
        "status",
        "dataset",
        "official_standard_supported",
        "official_measurement_equivalent",
        "units",
        "confidence_available",
        "weight_based",
    }

    assert len(parameters) == 14
    assert {parameter["name"] for parameter in parameters} == EXPECTED_PARAMETERS
    assert all(required_fields <= parameter.keys() for parameter in parameters)


def test_public_dataset_manifest_distinguishes_candidates_from_training_data():
    manifest = _load_json("data/manifests/datasets.json")
    assert "No images or annotations" in manifest["local_dataset_status"]
    datasets = {dataset["dataset_id"]: dataset for dataset in manifest["datasets"]}

    assert datasets["graindet_rice_v2"]["license"] == "CC BY 4.0"
    assert datasets["graindet_rice_v2"]["status"] == "external_candidate_download_blocked"
    assert datasets["roboflow_rice_quality_parameters_v1"]["reported_images"] == 224
    assert datasets["uci_rice_cammeo_osmancik"]["reported_images"] == 0
    assert datasets["rice_gate_synthetic_features_v1"]["real_world"] is False


def test_synthetic_model_scores_are_not_recorded_as_valid_evaluation_metrics():
    models = _load_json("configs/models.json")
    for name in ("rice_gate", "chalky", "damaged", "sprouted_weevilled"):
        assert models[name]["validation_metrics"] is None

    report = _load_json("models/evaluation_report.json")
    for name in ("rice_gate", "chalky", "damaged", "sprouted_weevilled"):
        assert report["models"][name]["metrics"] is None


def test_legacy_gate_training_requires_real_manifest():
    from training.train_rice_gate import train_rice_gate_model

    try:
        train_rice_gate_model()
    except FileNotFoundError as error:
        assert "Real labeled" in str(error)
    else:
        raise AssertionError("Gate training must not emit synthetic validation metrics")


def test_damage_and_foreign_matter_artifacts_are_not_marked_validated():
    models = _load_json("configs/models.json")
    assert models["damaged"]["status"] == "experimental"
    assert models["damaged"]["validation_metrics"] is None
    assert models["foreign_matter"]["status"] == "unavailable"
    assert models["foreign_matter"]["validation_metrics"] is None
