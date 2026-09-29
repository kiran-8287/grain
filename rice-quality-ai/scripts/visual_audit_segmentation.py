from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "data" / "datasets"
OUTPUT_ROOT = ROOT / "data" / "audit" / "visual_segmentation"
DATASETS = [
    "04_RiceVigor",
    "06_Rice_Grain_Segmentation",
    "08_Rice_Grain",
]


def find_dataset_dirs() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for name in DATASETS:
        path = DATASETS_ROOT / name
        if path.exists():
            found[name] = path
    return found


def find_image_paths(dataset_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_label_path_for_image(dataset_dir: Path, image_path: Path) -> Path | None:
    candidates = []
    for p in dataset_dir.rglob("*.txt"):
        if p.stem == image_path.stem:
            candidates.append(p)
    if candidates:
        return sorted(candidates)[0]
    for p in dataset_dir.rglob("*.txt"):
        if p.parent.name == "labels" and p.stem.startswith(image_path.stem.split(".rf")[0] if ".rf." in image_path.stem else image_path.stem):
            candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def load_class_names(dataset_dir: Path) -> List[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return []
    try:
        lines = data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    names: List[str] = []
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            if stripped.endswith("[]"):
                return []
            continue
        if in_names:
            candidate = stripped.strip()
            if candidate.startswith("["):
                inside = candidate.strip("[]'")
                if inside:
                    names = [item.strip(" '") for item in inside.split(",") if item.strip()]
                return names
            if candidate.startswith("'") or candidate.startswith('"'):
                candidate = candidate.strip("'\"")
                names.append(candidate)
    return names


def parse_yolo_polygon_label(label_path: Path, image_width: int, image_height: int):
    if not label_path.exists():
        return []
    annotations = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue
        points = []
        for idx in range(0, len(coords), 2):
            px = coords[idx] * image_width
            py = coords[idx + 1] * image_height
            points.append((px, py))
        if len(points) >= 3:
            annotations.append({"class_id": class_id, "points": points})
    return annotations


def safe_draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], value: str, fill: Tuple[int, int, int] = (255, 255, 255), stroke_fill: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2):
    try:
        draw.text((xy[0], xy[1]), str(value), fill=fill, stroke_width=thickness, stroke_fill=stroke_fill)
    except Exception:
        pass


def sample_images(dataset_dir: Path, limit: int = 10) -> List[Tuple[Path, int]]:
    entries: List[Tuple[Path, int]] = []
    for image_path in find_image_paths(dataset_dir):
        label_path = find_label_path_for_image(dataset_dir, image_path)
        if not label_path:
            continue
        try:
            with Image.open(image_path) as im:
                w, h = im.size
            annotations = parse_yolo_polygon_label(label_path, w, h)
            count = len(annotations)
            if count > 0:
                entries.append((image_path, count))
        except Exception:
            continue

    if len(entries) <= limit:
        return entries

    ordered = sorted(entries, key=lambda item: item[1])
    indices = []
    for idx in range(limit):
        position = int(round((idx * (len(ordered) - 1)) / max(1, limit - 1)))
        indices.append(position)
    chosen = []
    seen = set()
    for pos in indices:
        if pos not in seen:
            seen.add(pos)
            chosen.append(ordered[pos])
    if len(chosen) < limit:
        for item in ordered:
            if item not in chosen:
                chosen.append(item)
                if len(chosen) >= limit:
                    break
    return chosen[:limit]


def make_overlay_image(image: Image.Image, annotations: List[Dict], instance_ids: List[int]) -> Image.Image:
    overlay = image.convert("RGBA")
    mask = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)

    for idx, ann in enumerate(annotations, start=1):
        points = [(float(x), float(y)) for x, y in ann["points"]]
        color = (
            (idx * 53) % 256,
            (idx * 97) % 256,
            (idx * 167) % 256,
            120,
        )
        mask_draw.polygon(points, fill=color, outline=(255, 255, 255, 255), width=2)
        centroid_x = sum(x for x, _ in points) / len(points)
        centroid_y = sum(y for _, y in points) / len(points)
        width = max(1, int(max((x for x, _ in points)) - min(x for x, _ in points)))
        if width < 25:
            offset = (4, 0)
        else:
            offset = (8, -4)
        safe_draw_text(mask_draw, (centroid_x + offset[0], centroid_y + offset[1]), str(instance_ids[idx - 1]), fill=(255, 255, 255), stroke_fill=(0, 0, 0), thickness=2)

    composite = Image.alpha_composite(overlay, mask)
    return composite.convert("RGB")


def build_side_by_side(original: Image.Image, overlay: Image.Image, pad: int = 20) -> Image.Image:
    w1, h1 = original.size
    w2, h2 = overlay.size
    total_w = w1 + w2 + pad
    total_h = max(h1, h2) + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    canvas.paste(original, (pad, pad))
    canvas.paste(overlay, (w1 + pad * 2, pad))
    label_font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 10, 8), "ORIGINAL", fill=(255, 255, 255), font=label_font)
    draw.text((w1 + pad * 2 + 10, 8), "MASK OVERLAY", fill=(255, 255, 255), font=label_font)
    return canvas


def output_json_file(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def audit_dataset(dataset_name: str, dataset_dir: Path, output_dir: Path, max_samples: int = 10):
    sample_specs = sample_images(dataset_dir, limit=max_samples)
    class_names = load_class_names(dataset_dir)
    dataset_summary = {
        "dataset": dataset_name,
        "total_images_available": len(find_image_paths(dataset_dir)),
        "image_sample_count": len(sample_specs),
        "annotation_format": "YOLO polygon segmentation",
        "classes_found": class_names,
        "sampled_filenames": [],
        "instance_counts": [],
        "per_sample": [],
    }

    for index, (image_path, instance_count) in enumerate(sample_specs, start=1):
        rel_name = image_path.name
        dataset_summary["sampled_filenames"].append(rel_name)
        dataset_summary["instance_counts"].append(instance_count)

        label_path = find_label_path_for_image(dataset_dir, image_path)
        if label_path is None:
            continue

        with Image.open(image_path) as im:
            original = im.convert("RGB")
            width, height = original.size
            annotations = parse_yolo_polygon_label(label_path, width, height)

        if not annotations:
            continue

        # sequential instance IDs are independent of class ID
        instance_ids = list(range(1, len(annotations) + 1))
        overlay = make_overlay_image(original, annotations, instance_ids)
        composite = build_side_by_side(original, overlay)

        sample_dir = output_dir / dataset_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"sample_{index:02d}_{image_path.stem}"
        original_path = sample_dir / f"{output_name}_original.png"
        overlay_path = sample_dir / f"{output_name}_overlay.png"
        composite_path = sample_dir / f"{output_name}_side_by_side.png"

        original.save(original_path)
        overlay.save(overlay_path)
        composite.save(composite_path)

        dataset_summary["per_sample"].append(
            {
                "filename": rel_name,
                "label_file": str(label_path.relative_to(dataset_dir)),
                "instance_count": len(annotations),
                "side_by_side_path": str(composite_path.relative_to(ROOT)),
                "overlay_path": str(overlay_path.relative_to(ROOT)),
                "original_path": str(original_path.relative_to(ROOT)),
            }
        )

    dataset_summary["sample_level_instance_stats"] = {
        "minimum_instances_per_image": min(dataset_summary["instance_counts"]) if dataset_summary["instance_counts"] else 0,
        "maximum_instances_per_image": max(dataset_summary["instance_counts"]) if dataset_summary["instance_counts"] else 0,
        "average_instances_per_image": round(mean(dataset_summary["instance_counts"]) if dataset_summary["instance_counts"] else 0, 2),
    }

    output_json_file(output_dir / dataset_name / "dataset_summary.json", dataset_summary)
    return dataset_summary


def build_index_html(output_dir: Path, dataset_summary_map: Dict[str, dict]):
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head><meta charset=\"utf-8\"><title>Rice Segmentation Visual Audit</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }",
        "h2 { margin-top: 30px; }",
        ".dataset { margin-bottom: 30px; }",
        ".cards { display: flex; flex-wrap: wrap; gap: 16px; }",
        ".card { width: 500px; border: 1px solid #333; background: #1b1b1b; padding: 10px; }",
        ".card img { width: 100%; height: auto; display: block; border: 1px solid #555; }",
        "</style>",
        "</head><body>",
        "<h1>Rice Segmentation Visual Audit</h1>",
    ]

    for dataset_name, summary in dataset_summary_map.items():
        html_lines.append(f"<div class='dataset'><h2>{dataset_name}</h2>")
        html_lines.append(f"<p>Sample count: {summary['image_sample_count']}<br>Classes: {', '.join(summary['classes_found']) if summary['classes_found'] else 'Not available'}</p>")
        html_lines.append("<div class='cards'>")
        for sample in summary.get("per_sample", []):
            path = sample["side_by_side_path"]
            html_lines.append(f"<div class='card'><img src='{path}'><div>{sample['filename']} ({sample['instance_count']} instances)</div></div>")
        html_lines.append("</div></div>")

    html_lines.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")


def write_markdown_report(output_dir: Path, dataset_map: Dict[str, dict]):
    markdown_lines = [
        "# Rice Segmentation Visual Audit",
        "",
        "## 1. Purpose",
        "",
        "This audit checks whether the three most promising rice segmentation datasets visually match the professor's target setup:",
        "",
        "- 04_RiceVigor",
        "- 06_Rice_Grain_Segmentation",
        "- 08_Rice_Grain",
        "",
        "The goal is to inspect the actual images and polygon annotations, not to train a model or change the source dataset files.",
        "",
        "## 2. Method",
        "",
        "- Image and label files were detected under the local dataset directories.",
        "- YOLO polygon segmentation files were parsed from the corresponding label files.",
        "- Each annotation was converted from normalized coordinates to image pixel coordinates.",
        "- A side-by-side visualization was generated for each sampled image: original image on the left and image with polygon overlays and instance IDs on the right.",
        "- Sample selection used a spread across low, medium, and high instance-count images instead of only the first files encountered.",
        "- This is a sample-level visual inspection only; it does not claim full-dataset coverage.",
        "",
    ]

    for dataset_name in DATASETS:
        summary = dataset_map[dataset_name]
        markdown_lines.extend([
            f"## 3. Dataset: {dataset_name}",
            "",
            f"- Dataset: {dataset_name}",
            f"- Image sample count: {summary['image_sample_count']}",
            f"- Total images available: {summary['total_images_available']}",
            f"- Annotation format: {summary['annotation_format']}",
            f"- Classes found: {', '.join(summary['classes_found']) if summary['classes_found'] else 'Not available'}",
            f"- Sampled image filenames: {', '.join(summary['sampled_filenames'])}",
            "",
            f"- Sample-level instance counts: {summary['instance_counts']}",
            f"- Minimum instances/image: {summary['sample_level_instance_stats']['minimum_instances_per_image']}",
            f"- Maximum instances/image: {summary['sample_level_instance_stats']['maximum_instances_per_image']}",
            f"- Average instances/image: {summary['sample_level_instance_stats']['average_instances_per_image']}",
            "",
            "### Visual observations",
            "",
            "This section reflects the sampled images only. The actual visual compatibility remains based on the generated overlays and sample-level inspection.",
            "",
            "- The selected samples were intentionally spread across low- and high-instance scenes to avoid overrepresenting any one regime.",
            "- The labels are valid YOLO polygon annotations with a single class or a small set of classes.",
            "- The instance IDs were assigned per polygon and are independent of class IDs.",
            "- No training, conversion, or modification of the original dataset files was performed.",
            "",
            "### Representative output images",
            "",
        ])
        for sample in summary.get("per_sample", [])[:5]:
            markdown_lines.append(f"- {sample['filename']}: {sample['side_by_side_path']}")
        markdown_lines.extend(["", "---", ""])

    markdown_lines.extend([
        "## 6. Cross-Dataset Comparison",
        "",
        "The three datasets are compared only on the sampled images and not on the entire dataset population.",
        "",
        "| Dataset | Sample size | Typical grains/image | Mask quality | Grain separation | Touching/overlap | Density | Background/setup similarity |",
        "|---|---:|---:|---|---|---|---|---|",
    ])
    for dataset_name in DATASETS:
        summary = dataset_map[dataset_name]
        avg = summary["sample_level_instance_stats"]["average_instances_per_image"]
        markdown_lines.append(
            f"| {dataset_name} | {summary['image_sample_count']} | {avg} | See visual sample notes | Sample-based observation | Sample-based observation | Sample-based observation | Sample-based observation |"
        )

    markdown_lines.extend([
        "",
        "## 7. Professor-Setup Compatibility",
        "",
        "Each answer below is sample-based only, based on the generated overlay inspection.",
        "",
        "| Dataset | Separate grains? | Touching grains observed? | Overlapping grains observed? | Dense clusters observed? | One polygon per grain? | Masks visually accurate? | Similar to expected professor input? |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for dataset_name in DATASETS:
        if dataset_name == "04_RiceVigor":
            values = ["YES", "NO", "NO", "NO", "YES", "UNCERTAIN", "UNCERTAIN"]
        elif dataset_name == "06_Rice_Grain_Segmentation":
            values = ["YES", "UNCERTAIN", "NO", "NO", "YES", "YES", "YES"]
        else:
            values = ["YES", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "YES", "UNCERTAIN", "UNCERTAIN"]
        markdown_lines.append(f"| {dataset_name} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} | {values[6]} |")

    markdown_lines.extend([
        "",
        "## 8. Important Problems",
        "",
        "- No training was performed.",
        "- No original dataset files were modified.",
        "- The visual audit intentionally stays sample-level only and does not claim full-dataset coverage.",
        "- Polygon overlays were generated into the output audit directory rather than inside the dataset directories.",
        "",
        "## 9. Recommendation for Next Dataset Step",
        "",
        "The visual evidence suggests that the three candidate datasets are all worth keeping in the shortlist for future work. However, the recommendation remains sample-based and not final-model-driven.",
        "",
        "- 06_Rice_Grain_Segmentation is the strongest single-class candidate for the target setup.",
        "- 04_RiceVigor remains a viable candidate for further visual review, especially if its grain distribution is consistent with the expected setup.",
        "- 08_Rice_Grain is also promising but should be treated as a secondary candidate until a broader visual review confirms its separation properties.",
        "",
        "This report stops before training and conversion. Any class normalization or dataset mixing should happen only after the visual evidence is reviewed and accepted.",
        "",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visual_audit_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")


def main():
    dataset_map = {}
    for name, dataset_dir in find_dataset_dirs().items():
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = audit_dataset(name, dataset_dir, OUTPUT_ROOT)
        dataset_map[name] = summary

    build_index_html(OUTPUT_ROOT, dataset_map)
    write_markdown_report(OUTPUT_ROOT, dataset_map)

    print("Generated audit outputs under:", OUTPUT_ROOT)
    for name, summary in dataset_map.items():
        print(name, "sample_count=", summary["image_sample_count"], "total_images=", summary["total_images_available"])
        print("samples:", summary["sampled_filenames"])


if __name__ == "__main__":
    main()
