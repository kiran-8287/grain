"""
Export utilities for Rice Quality Analysis results (JSON and CSV).
"""

import csv
import io
import json
from typing import Any, Dict


def export_result_json(result: Dict[str, Any]) -> str:
    """Export complete result as formatted JSON string."""
    # Omit giant base64 annotated image in the raw JSON export for cleaner downloads
    clean_result = {k: v for k, v in result.items() if k != "annotated_image_base64"}
    return json.dumps(clean_result, indent=2)


def export_result_csv(result: Dict[str, Any]) -> str:
    """
    Export grain-level and sample-level data as CSV format.
    
    Columns as specified:
    grain_id, length, breadth, L_over_B, broken, damaged, discoloured,
    chalky, red, dehusked, immature_shrunken, sprouted_weevilled,
    segmentation_confidence, quality
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # 1. Sample Summary Header
    summary = result.get("summary", {})
    writer.writerow(["# Rice Quality Analysis Report"])
    writer.writerow(["# Sample Summary"])
    writer.writerow(["Parameter", "Value", "Unit / Notes"])
    writer.writerow(["Total Rice Grains", summary.get("total_rice_grains", 0), "count"])
    writer.writerow(["Uncertain Grains", summary.get("uncertain_grains", 0), "count"])
    writer.writerow(["Foreign Matter Count", summary.get("foreign_matter_count", 0), "objects"])
    writer.writerow(["Broken Grain", summary.get("broken_percent") if summary.get("broken_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Damaged / Slightly Damaged", summary.get("damaged_percent") if summary.get("damaged_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Discoloured", summary.get("discoloured_percent") if summary.get("discoloured_percent") is not None else "Not determinable", "observed image fraction (%)"])
    chalky_fraction = summary.get("chalky_percent")
    writer.writerow([
        "Chalky",
        chalky_fraction if chalky_fraction is not None else "Not determinable",
        "observed image fraction (%)",
    ])
    writer.writerow(["Red Grain", summary.get("red_percent") if summary.get("red_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Dehusked (Visual Proxy)", summary.get("dehusked_percent") if summary.get("dehusked_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Immature / Shrunken", summary.get("immature_percent") if summary.get("immature_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Sprouted / Weevilled", summary.get("sprouted_percent") if summary.get("sprouted_percent") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Admixture of Lower Class", summary.get("admixture_percentage") if summary.get("admixture_percentage") is not None else "Not determinable", "observed image fraction (%)"])
    writer.writerow(["Average Length", summary.get("average_length", 0.0), summary.get("measurement_unit", "pixels")])
    writer.writerow(["Average Breadth", summary.get("average_breadth", 0.0), summary.get("measurement_unit", "pixels")])
    writer.writerow(["Average L/B Ratio", summary.get("average_lb_ratio", 0.0), "ratio"])
    writer.writerow(["Calibration Mode", summary.get("calibration_mode", "none"), ""])
    writer.writerow([])

    # 2. Per-Grain Data
    writer.writerow(["# Per-Grain Data"])
    grain_headers = [
        "grain_id",
        "length",
        "breadth",
        "L_over_B",
        "broken",
        "damaged",
        "discoloured",
        "chalky",
        "red",
        "dehusked",
        "immature_shrunken",
        "sprouted_weevilled",
        "segmentation_confidence",
        "quality",
    ]
    writer.writerow(grain_headers)

    grains = result.get("grains", [])
    for g in grains:
        geom = g.get("geometry", {})
        defects = g.get("defects", {})

        length_val = geom.get("length_mm") if geom.get("length_mm") is not None else geom.get("length_pixels", 0.0)
        breadth_val = geom.get("breadth_mm") if geom.get("breadth_mm") is not None else geom.get("breadth_pixels", 0.0)
        lb_val = geom.get("lb_ratio", 0.0)

        broken_val = defects.get("broken", {}).get("broken_label", "undetermined")
        damaged_val = defects.get("damaged", {}).get("damaged_label", "unknown")
        discoloured_val = defects.get("discoloured", {}).get("discoloured_label", "unknown")
        chalky_val = defects.get("chalky", {}).get("chalky_label", "unknown")
        red_val = defects.get("red", {}).get("red_label", "unknown")
        dehusked_val = defects.get("dehusked", {}).get("dehusked_label", "unknown")
        immature_val = defects.get("immature_shrunken", {}).get("immature_shrunken_status", "unknown")
        sprouted_val = defects.get("sprouted_weevilled", {}).get("sprouted_weevilled_label", "unknown")

        writer.writerow([
            g.get("id"),
            length_val,
            breadth_val,
            lb_val,
            broken_val,
            damaged_val,
            discoloured_val,
            chalky_val,
            red_val,
            dehusked_val,
            immature_val,
            sprouted_val,
            g.get("confidence", 0.0),
            g.get("segmentation_quality", "good"),
        ])

    return output.getvalue()
