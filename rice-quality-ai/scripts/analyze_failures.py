"""
scripts/analyze_failures.py
===========================
Analyse evaluation failures and generate structured reports.

Inputs:
  - results/evaluation/eval_report.json
  - results/evaluation/per_image_results.json

Outputs:
  a. docs/FAILURE_ANALYSIS.md
        - Executive summary table
        - Per-category (A-I) metrics table
        - Failure type breakdown (counts + %)
        - Per-category x failure-type heatmap/table
        - Top 10 worst-performing images (composite failure score)
        - Per-training-run appendable section
        - Auto-generated recommendations

  b. results/evaluation/failure_summary.json   (structured aggregates)
  c. results/evaluation/failures_by_category.csv

Usage:
    python scripts/analyze_failures.py
    python scripts/analyze_failures.py --eval-dir results/evaluation
    python scripts/analyze_failures.py --run-info '{"run_id":"run_042","date":"2026-09-28","notes":"Tuned NMS threshold"}' --append-docs
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FAILURE_TYPES = [
    "MISSED_GRAIN",
    "MERGED_GRAINS",
    "SPLIT_GRAIN",
    "FALSE_RICE",
    "FALSE_FOREIGN_MATTER",
    "DUPLICATE",
    "BAD_MASK",
    "LOW_CONFIDENCE",
]

FAILURE_WEIGHTS: Dict[str, float] = {
    "MISSED_GRAIN": 3.0,
    "MERGED_GRAINS": 2.5,
    "SPLIT_GRAIN": 2.0,
    "BAD_MASK": 2.0,
    "DUPLICATE": 1.0,
    "LOW_CONFIDENCE": 0.5,
    "FALSE_RICE": 4.0,
    "FALSE_FOREIGN_MATTER": 3.5,
}

CATEGORIES_9 = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
CATEGORY_NAMES = {
    "A": "single_isolated",
    "B": "few_separated",
    "C": "touching",
    "D": "overlapping",
    "E": "dense",
    "F": "rice_plus_foreign",
    "G": "no_rice",
    "H": "foreign_only",
    "I": "mixed_difficult",
}


def load_eval_results(eval_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load eval_report.json and per_image_results.json from the evaluation dir.

    Returns
    -------
    (eval_report, per_image_results)
    """
    eval_dir = Path(eval_dir)
    eval_report_path = eval_dir / "eval_report.json"
    per_image_path = eval_dir / "per_image_results.json"

    if not eval_report_path.exists():
        raise FileNotFoundError(f"eval_report.json not found at {eval_report_path}")
    if not per_image_path.exists():
        raise FileNotFoundError(f"per_image_results.json not found at {per_image_path}")

    with open(eval_report_path, "r", encoding="utf-8") as f:
        eval_report: Dict[str, Any] = json.load(f)

    with open(per_image_path, "r", encoding="utf-8") as f:
        per_image_results: List[Dict[str, Any]] = json.load(f)

    logger.info(
        f"Loaded eval_report (overall images: {eval_report.get('overall', {}).get('num_images', '?')}) "
        f"and {len(per_image_results)} per-image records."
    )
    return eval_report, per_image_results


def aggregate_failures_by_type(
    per_image_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate failures across all images grouped by failure type.

    Returns
    -------
    dict with keys:
        counts: {failure_type: total_count}
        percentages: {failure_type: pct_of_all_failures}
        per_image_mean: {failure_type: avg per image}
        total_failures: int
    """
    counts: Dict[str, int] = {ft: 0 for ft in FAILURE_TYPES}
    per_image_counts: Dict[str, List[int]] = {ft: [] for ft in FAILURE_TYPES}
    n_images = len(per_image_results)

    for rec in per_image_results:
        failures = rec.get("failures", {})
        for ft in FAILURE_TYPES:
            c = int(failures.get(ft, 0))
            counts[ft] += c
            per_image_counts[ft].append(c)

    total = sum(counts.values())

    percentages: Dict[str, float] = {
        ft: (counts[ft] / total * 100.0 if total > 0 else 0.0) for ft in FAILURE_TYPES
    }

    per_image_mean: Dict[str, float] = {
        ft: (float(np.mean(per_image_counts[ft])) if n_images > 0 else 0.0)
        for ft in FAILURE_TYPES
    }

    return {
        "counts": counts,
        "percentages": percentages,
        "per_image_mean": per_image_mean,
        "total_failures": total,
        "n_images": n_images,
    }


def aggregate_failures_by_category(
    per_image_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics and failures per category (A–I).

    Returns
    -------
    dict mapping category code -> {
        num_images, total_gt_rice,
        mean_ap50_box, mean_ap50_95_box, mean_ap50_mask, mean_ap50_95_mask,
        mean_count_error_pct,
        failure_counts: {ft: count},
        failure_rates: {ft: rate_per_gt_rice},
        composite_failure_score_mean: float
    }
    """
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in per_image_results:
        cat = str(rec.get("category", "I"))
        by_cat[cat].append(rec)

    result: Dict[str, Any] = {}

    for cat in CATEGORIES_9:
        recs = by_cat.get(cat, [])
        n = len(recs)
        if n == 0:
            result[cat] = {
                "category_name": CATEGORY_NAMES.get(cat, cat),
                "num_images": 0,
                "total_gt_rice": 0,
                "mean_ap50_box": 0.0,
                "mean_ap50_95_box": 0.0,
                "mean_ap50_mask": 0.0,
                "mean_ap50_95_mask": 0.0,
                "mean_count_error_pct": 0.0,
                "failure_counts": {ft: 0 for ft in FAILURE_TYPES},
                "failure_rates": {ft: 0.0 for ft in FAILURE_TYPES},
                "composite_failure_score_mean": 0.0,
            }
            continue

        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in recs if key in r]
            return float(np.mean(vals)) if vals else 0.0

        total_gt = sum(int(r.get("gt_count", 0)) for r in recs)

        failure_counts: Dict[str, int] = {ft: 0 for ft in FAILURE_TYPES}
        scores: List[float] = []
        for r in recs:
            f = r.get("failures", {})
            for ft in FAILURE_TYPES:
                failure_counts[ft] += int(f.get(ft, 0))
            scores.append(compute_failure_score(r))

        failure_rates: Dict[str, float] = {
            ft: (failure_counts[ft] / max(total_gt, 1)) for ft in FAILURE_TYPES
        }

        result[cat] = {
            "category_name": CATEGORY_NAMES.get(cat, cat),
            "num_images": n,
            "total_gt_rice": total_gt,
            "mean_ap50_box": _mean("ap50_box"),
            "mean_ap50_95_box": _mean("ap50_95_box"),
            "mean_ap50_mask": _mean("ap50_mask"),
            "mean_ap50_95_mask": _mean("ap50_95_mask"),
            "mean_count_error_pct": _mean("count_error_pct"),
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "composite_failure_score_mean": float(np.mean(scores)) if scores else 0.0,
        }

    return result


def compute_failure_score(image_result: Dict[str, Any]) -> float:
    """
    Weighted composite failure score for a single image.

    Higher = worse. Weights defined in FAILURE_WEIGHTS.
    """
    failures = image_result.get("failures", {})
    score = 0.0
    for ft, weight in FAILURE_WEIGHTS.items():
        score += float(failures.get(ft, 0)) * weight
    return round(score, 4)


def generate_recommendations(
    overall: Dict[str, Any],
    by_category: Dict[str, Any],
    by_type: Dict[str, Any],
) -> List[str]:
    """
    Generate human-readable recommendations from aggregate failure data.

    Returns a list of string recommendations (bullet-ready).
    """
    recs: List[str] = []

    pct = by_type.get("percentages", {})
    counts = by_type.get("counts", {})
    total = by_type.get("total_failures", 0)

    if total == 0:
        recs.append("No failures detected — consider expanding the test set or lowering confidence thresholds to exercise the pipeline.")
        return recs

    ranked = sorted(FAILURE_TYPES, key=lambda ft: pct.get(ft, 0.0), reverse=True)
    top_failure = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    if top_failure == "MISSED_GRAIN" and pct[top_failure] >= 25:
        recs.append(
            f"Missed grains dominate failures ({pct[top_failure]:.1f}%). "
            "Lower segmentation confidence threshold and/or add more low-contrast / partial-grain training examples."
        )
    elif top_failure == "MERGED_GRAINS" and pct[top_failure] >= 20:
        recs.append(
            f"Merged grains are the top failure ({pct[top_failure]:.1f}%). "
            "Add more touching / overlapping grain training data and investigate watershed / distance-transform post-processing."
        )
    elif top_failure == "FALSE_RICE" and pct[top_failure] >= 15:
        recs.append(
            f"FALSE_RICE is the most common failure ({pct[top_failure]:.1f}%). "
            "Raise NMS IoU threshold, tighten min-grain-area filter, and add hard-negative (background texture) training patches."
        )
    elif top_failure == "FALSE_FOREIGN_MATTER" and pct[top_failure] >= 15:
        recs.append(
            f"FALSE_FOREIGN_MATTER is the top failure ({pct[top_failure]:.1f}%). "
            "Review FM class training distribution and raise FM-specific confidence threshold."
        )
    elif top_failure == "SPLIT_GRAIN" and pct[top_failure] >= 15:
        recs.append(
            f"SPLIT_GRAIN is the top failure ({pct[top_failure]:.1f}%). "
            "Reduce oversegmentation: weaken watershed markers, increase mask NMS IoU threshold, or use larger morphological closing kernel."
        )
    elif top_failure == "BAD_MASK" and pct[top_failure] >= 15:
        recs.append(
            f"BAD_MASK is the top failure ({pct[top_failure]:.1f}%). "
            "Review mask quality in training set and consider fine-tuning with Dice / boundary loss."
        )

    for cat in CATEGORIES_9:
        cat_data = by_category.get(cat, {})
        if cat_data.get("num_images", 0) == 0:
            continue
        ap_mask = cat_data.get("mean_ap50_mask", 1.0)
        cnt_err = cat_data.get("mean_count_error_pct", 0.0)
        if ap_mask < 0.60:
            recs.append(
                f"Category {cat} ({cat_data.get('category_name', cat)}) has low mask AP@50 = {ap_mask:.3f}. "
                "Augment training data with scenes matching this category's characteristics."
            )
        if cnt_err > 25.0:
            recs.append(
                f"Category {cat} ({cat_data.get('category_name', cat)}) shows high count error ({cnt_err:.1f}%). "
                "Tune confidence / NMS thresholds for density and consider category-specific post-processing."
            )

    if counts.get("LOW_CONFIDENCE", 0) > 0:
        mean_conf_low = by_type.get("per_image_mean", {}).get("LOW_CONFIDENCE", 0.0)
        if mean_conf_low >= 1.0:
            recs.append(
                f"LOW_CONFIDENCE occurrences average {mean_conf_low:.2f} per image. "
                "Consider a calibration pass on confidence scores or threshold recalibration."
            )

    if counts.get("DUPLICATE", 0) > 0:
        recs.append(
            f"DUPLICATE detections present ({counts['DUPLICATE']} total). "
            "Increase mask-IoU NMS threshold or run additional box-level NMS pass with tighter IoU."
        )

    if second is not None and pct.get(second, 0) >= 15:
        if second == "MERGED_GRAINS":
            recs.append(
                "Secondary failure: merged grains. Review tile-overlap merging strategy for dense scenes."
            )
        elif second == "MISSED_GRAIN":
            recs.append(
                "Secondary failure: missed grains. Consider tiling strategy for high-resolution or dense inputs."
            )

    overall_fc = overall.get("failure_counts", {})
    if overall_fc.get("MERGED_GRAINS", 0) > 0 or overall_fc.get("SPLIT_GRAIN", 0) > 0:
        touching_cat = "C"
        overlap_cat = "D"
        if by_category.get(touching_cat, {}).get("num_images", 0) > 0:
            score = by_category[touching_cat].get("composite_failure_score_mean", 0.0)
            if score > 3.0:
                recs.append(
                    f"Category C (touching) composite score {score:.2f} — add more touching-grain examples to training split."
                )
        if by_category.get(overlap_cat, {}).get("num_images", 0) > 0:
            score = by_category[overlap_cat].get("composite_failure_score_mean", 0.0)
            if score > 3.0:
                recs.append(
                    f"Category D (overlapping) composite score {score:.2f} — add more overlapping-grain examples and tune mask NMS."
                )

    if not recs:
        recs.append(
            "All failure categories are within acceptable ranges. Continue monitoring and expand evaluation coverage."
        )

    return recs


def write_failure_csv(csv_path: Path, by_category: Dict[str, Any]) -> Path:
    """
    Write per-category metrics and failure counts to CSV.

    Returns the output path.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "category", "category_name", "num_images", "total_gt_rice",
        "mean_ap50_box", "mean_ap50_95_box", "mean_ap50_mask", "mean_ap50_95_mask",
        "mean_count_error_pct", "composite_failure_score_mean",
    ] + FAILURE_TYPES + [f"{ft}_rate" for ft in FAILURE_TYPES]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cat in CATEGORIES_9:
            d = by_category.get(cat, {})
            if d.get("num_images", 0) == 0:
                continue
            row = [
                cat,
                d.get("category_name", cat),
                d.get("num_images", 0),
                d.get("total_gt_rice", 0),
                f"{d.get('mean_ap50_box', 0.0):.4f}",
                f"{d.get('mean_ap50_95_box', 0.0):.4f}",
                f"{d.get('mean_ap50_mask', 0.0):.4f}",
                f"{d.get('mean_ap50_95_mask', 0.0):.4f}",
                f"{d.get('mean_count_error_pct', 0.0):.2f}",
                f"{d.get('composite_failure_score_mean', 0.0):.4f}",
            ]
            fc = d.get("failure_counts", {})
            fr = d.get("failure_rates", {})
            for ft in FAILURE_TYPES:
                row.append(fc.get(ft, 0))
            for ft in FAILURE_TYPES:
                row.append(f"{fr.get(ft, 0.0):.4f}")
            writer.writerow(row)

    logger.info(f"Wrote per-category CSV to {csv_path}")
    return csv_path


def write_failure_analysis_md(
    output_path: Path,
    overall_metrics: Dict[str, Any],
    by_category: Dict[str, Any],
    by_type: Dict[str, Any],
    worst_images: List[Dict[str, Any]],
    recommendations: List[str],
    run_info: Optional[Dict[str, Any]] = None,
    append_docs: bool = False,
) -> Path:
    """
    Generate / append FAILURE_ANALYSIS.md report.

    Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []

    existing_content = ""
    if append_docs and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
        except Exception as exc:
            logger.warning(f"Could not read existing {output_path}: {exc}")
            append_docs = False

    if not append_docs:
        lines.append("# Failure Analysis Report")
        lines.append("")
        lines.append(f"_Generated: {timestamp}_")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")

        o = overall_metrics
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Images evaluated | {o.get('num_images', 0)} |")
        lines.append(f"| Categories represented | {sum(1 for c in CATEGORIES_9 if by_category.get(c, {}).get('num_images', 0) > 0)} |")
        lines.append(f"| Total GT rice instances | {o.get('total_gt_rice', 0)} |")
        lines.append(f"| Total pred instances (rice+fm) | {o.get('total_pred_instances', 0)} |")
        lines.append(f"| Mean box AP@0.50 | {o.get('mean_ap50_box', 0.0):.4f} |")
        lines.append(f"| Mean box AP@0.50:0.95 | {o.get('mean_ap50_95_box', 0.0):.4f} |")
        lines.append(f"| Mean mask AP@0.50 | {o.get('mean_ap50_mask', 0.0):.4f} |")
        lines.append(f"| Mean mask AP@0.50:0.95 | {o.get('mean_ap50_95_mask', 0.0):.4f} |")
        lines.append(f"| Mean count error % | {o.get('mean_count_error_pct', 0.0):.2f} |")
        lines.append(f"| Mean inference ms | {o.get('mean_inference_ms', 0.0):.1f} |")
        lines.append(f"| **Total failures** | **{by_type.get('total_failures', 0)}** |")
        lines.append("")

        lines.append("## Per-Category Metrics (A–I)")
        lines.append("")
        lines.append(
            "| Cat | Name | N | GT Rice | AP50_box | mAP_box | AP50_mask | mAP_mask | CntE% | CompScore |"
        )
        lines.append(
            "|-----|------|---|---------|----------|---------|-----------|----------|-------|-----------|"
        )
        for cat in CATEGORIES_9:
            d = by_category.get(cat, {})
            n = d.get("num_images", 0)
            if n == 0:
                continue
            lines.append(
                f"| {cat} | {d.get('category_name', cat)} | {n} | "
                f"{d.get('total_gt_rice', 0)} | "
                f"{d.get('mean_ap50_box', 0.0):.4f} | {d.get('mean_ap50_95_box', 0.0):.4f} | "
                f"{d.get('mean_ap50_mask', 0.0):.4f} | {d.get('mean_ap50_95_mask', 0.0):.4f} | "
                f"{d.get('mean_count_error_pct', 0.0):.2f} | "
                f"{d.get('composite_failure_score_mean', 0.0):.2f} |"
            )
        lines.append("")

        lines.append("## Failure Type Breakdown")
        lines.append("")
        counts = by_type.get("counts", {})
        pcts = by_type.get("percentages", {})
        per_img_avg = by_type.get("per_image_mean", {})
        lines.append("| Failure Type | Count | % of All Failures | Avg per Image | Weight |")
        lines.append("|-------------|------:|------------------:|--------------:|-------:|")
        ranked_ft = sorted(FAILURE_TYPES, key=lambda ft: counts.get(ft, 0), reverse=True)
        for ft in ranked_ft:
            c = counts.get(ft, 0)
            if c == 0 and pcts.get(ft, 0.0) == 0:
                continue
            bar_len = int(round(pcts.get(ft, 0.0) / 2.0))
            bar = "\u2588" * bar_len if bar_len > 0 else ""
            lines.append(
                f"| {ft} | {c} | {pcts.get(ft, 0.0):.1f}% {bar} | "
                f"{per_img_avg.get(ft, 0.0):.2f} | {FAILURE_WEIGHTS.get(ft, 0.0):.1f} |"
            )
        lines.append("")

        lines.append("## Per-Category \u00d7 Failure-Type Matrix (counts)")
        lines.append("")
        header_cells = ["Cat"] + list(FAILURE_TYPES)
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["-----"] * len(header_cells)) + "|")
        for cat in CATEGORIES_9:
            d = by_category.get(cat, {})
            if d.get("num_images", 0) == 0:
                continue
            fc = d.get("failure_counts", {})
            row = [cat] + [str(fc.get(ft, 0)) for ft in FAILURE_TYPES]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## Top 10 Worst-Performing Images")
        lines.append("")
        lines.append(
            "| Rank | Image | Category | Score | Miss | Merg | Split | F-Rice | F-FM | Dup | BadMask | LowConf |"
        )
        lines.append(
            "|-----:|-------|----------|------:|-----:|-----:|------:|-------:|-----:|----:|--------:|--------:|"
        )
        for rank, wi in enumerate(worst_images[:10], start=1):
            f = wi.get("failures", {})
            lines.append(
                f"| {rank} | {wi.get('file_name','?')} | {wi.get('category','?')} | "
                f"{wi.get('failure_score', 0.0):.2f} | "
                f"{f.get('MISSED_GRAIN', 0)} | {f.get('MERGED_GRAINS', 0)} | "
                f"{f.get('SPLIT_GRAIN', 0)} | {f.get('FALSE_RICE', 0)} | "
                f"{f.get('FALSE_FOREIGN_MATTER', 0)} | {f.get('DUPLICATE', 0)} | "
                f"{f.get('BAD_MASK', 0)} | {f.get('LOW_CONFIDENCE', 0)} |"
            )
        lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        for i, r in enumerate(recommendations, start=1):
            lines.append(f"{i}. {r}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Training Run History")
        lines.append("")

    if run_info:
        lines.append(f"### Run: {run_info.get('run_id', 'unnamed')}")
        lines.append("")
        lines.append(f"- **Date**: {run_info.get('date', timestamp)}")
        lines.append(f"- **Notes**: {run_info.get('notes', '')}")
        lines.append(f"- **Model type**: {run_info.get('model_type', overall_metrics.get('model_type', 'n/a'))}")
        lines.append(f"- **Weights**: {run_info.get('weights_path', overall_metrics.get('weights_path', 'n/a'))}")
        lines.append(f"- **Images evaluated**: {overall_metrics.get('num_images', 0)}")
        lines.append(f"- **Mean mask AP@0.50**: {overall_metrics.get('mean_ap50_mask', 0.0):.4f}")
        lines.append(f"- **Mean box AP@0.50**: {overall_metrics.get('mean_ap50_box', 0.0):.4f}")
        lines.append(f"- **Total failures**: {by_type.get('total_failures', 0)}")
        fc = overall_metrics.get("failure_counts", {})
        lines.append(f"- Failure counts: " + ", ".join(f"{ft}={fc.get(ft, 0)}" for ft in FAILURE_TYPES if fc.get(ft, 0) > 0))
        if run_info.get("changes"):
            lines.append(f"- Changes: {run_info['changes']}")
        lines.append("")

    content = "\n".join(lines)
    if append_docs and existing_content:
        combined = existing_content.rstrip() + "\n\n" + content
    else:
        combined = content

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    logger.info(f"Wrote failure analysis markdown to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse evaluation failures and generate reports."
    )
    parser.add_argument("--eval-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "evaluation"),
                        help=f"Evaluation directory. Default: {PROJECT_ROOT / 'results' / 'evaluation'}")
    parser.add_argument("--run-info", type=str, default=None,
                        help='Optional JSON string with run metadata, e.g. \'{"run_id":"r1","date":"2026-09-28","notes":"Tuned NMS"}\'')
    parser.add_argument("--append-docs", action="store_true",
                        help="Append to existing docs/FAILURE_ANALYSIS.md instead of overwriting.")
    parser.add_argument("--docs-path", type=str,
                        default=str(PROJECT_ROOT / "docs" / "FAILURE_ANALYSIS.md"),
                        help="Path to FAILURE_ANALYSIS.md output.")
    parser.add_argument("--summary-json", type=str, default=None,
                        help="Override path to failure_summary.json output.")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Override path to failures_by_category.csv output.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    try:
        eval_report, per_image_results = load_eval_results(eval_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    overall: Dict[str, Any] = eval_report.get("overall", {})
    if "model_type" in eval_report:
        overall["model_type"] = eval_report["model_type"]
    if "weights_path" in eval_report:
        overall["weights_path"] = eval_report["weights_path"]

    by_type = aggregate_failures_by_type(per_image_results)
    by_category = aggregate_failures_by_category(per_image_results)

    scored: List[Dict[str, Any]] = []
    for rec in per_image_results:
        r = dict(rec)
        r["failure_score"] = compute_failure_score(rec)
        scored.append(r)
    scored.sort(key=lambda r: r["failure_score"], reverse=True)
    worst_images = scored[:10]

    recommendations = generate_recommendations(overall, by_category, by_type)

    run_info: Optional[Dict[str, Any]] = None
    if args.run_info:
        try:
            run_info = json.loads(args.run_info)
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid --run-info JSON: {exc}")
            sys.exit(1)

    summary_json_path = Path(args.summary_json) if args.summary_json else (eval_dir / "failure_summary.json")
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    failure_summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "evaluation_dir": str(eval_dir.resolve()),
        "overall": overall,
        "by_type": by_type,
        "by_category": by_category,
        "worst_images": [
            {
                "file_name": wi.get("file_name"),
                "category": wi.get("category"),
                "failure_score": wi.get("failure_score"),
                "failures": wi.get("failures", {}),
                "img_path": wi.get("img_path"),
            }
            for wi in worst_images
        ],
        "recommendations": recommendations,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(failure_summary, f, indent=2, default=float)
    logger.info(f"Wrote failure summary JSON to {summary_json_path}")

    csv_path = Path(args.csv_path) if args.csv_path else (eval_dir / "failures_by_category.csv")
    write_failure_csv(csv_path, by_category)

    write_failure_analysis_md(
        output_path=Path(args.docs_path),
        overall_metrics=overall,
        by_category=by_category,
        by_type=by_type,
        worst_images=worst_images,
        recommendations=recommendations,
        run_info=run_info,
        append_docs=args.append_docs,
    )

    total = by_type.get("total_failures", 0)
    print(
        f"Analysis complete: {len(per_image_results)} images, "
        f"{total} failures, {len(recommendations)} recommendations."
    )
    print(f"  -> {summary_json_path}")
    print(f"  -> {csv_path}")
    print(f"  -> {args.docs_path}")


if __name__ == "__main__":
    main()
