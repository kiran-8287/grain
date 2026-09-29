from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path('A:/grain')
DATASET_ROOT = ROOT / 'data' / 'datasets'
OUTPUT_ROOT = ROOT / 'data' / 'audit' / 'full_visual_segmentation'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

TARGET_DATASETS = [
    '01_Mendeley_Rice_Variety',
    '02_Grainalyze',
    '03_Rice_Variety',
    '04_RiceVigor',
    '05_Ricee',
]


def polygon_area(points: List[Tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bounds(points: List[Tuple[float, float]], width: int, height: int, padding: int = 0) -> Tuple[int, int, int, int]:
    return (
        max(0, int(math.floor(min(x for x, _ in points))) - padding),
        max(0, int(math.floor(min(y for _, y in points))) - padding),
        min(width - 1, int(math.ceil(max(x for x, _ in points))) + padding),
        min(height - 1, int(math.ceil(max(y for _, y in points))) + padding),
    )


def polygon_mask_in_bounds(points: List[Tuple[float, float]], bounds: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bounds
    mask = Image.new('1', (x_max - x_min + 1, y_max - y_min + 1), 0)
    ImageDraw.Draw(mask).polygon([(x - x_min, y - y_min) for x, y in points], fill=1)
    return np.array(mask, dtype=np.uint8)


def dilate(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.array(image.filter(ImageFilter.MaxFilter(3)), dtype=np.uint8) > 0


def pairwise_overlap(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], width: int, height: int) -> bool:
    bounds_a = polygon_bounds(points_a, width, height)
    bounds_b = polygon_bounds(points_b, width, height)
    shared = (
        max(bounds_a[0], bounds_b[0]), max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]), min(bounds_a[3], bounds_b[3]),
    )
    if shared[0] > shared[2] or shared[1] > shared[3]:
        return False
    mask_a = polygon_mask_in_bounds(points_a, shared)
    mask_b = polygon_mask_in_bounds(points_b, shared)
    return bool(np.logical_and(mask_a > 0, mask_b > 0).any())


def pairwise_touching(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], width: int, height: int) -> bool:
    bounds_a = polygon_bounds(points_a, width, height, padding=1)
    bounds_b = polygon_bounds(points_b, width, height, padding=1)
    shared = (
        max(bounds_a[0], bounds_b[0]), max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]), min(bounds_a[3], bounds_b[3]),
    )
    if shared[0] > shared[2] or shared[1] > shared[3]:
        return False
    mask_a = polygon_mask_in_bounds(points_a, shared)
    mask_b = polygon_mask_in_bounds(points_b, shared)
    return bool(np.logical_and(dilate(mask_a), dilate(mask_b)).any())


def nearest_centroid_distances(points: List[Tuple[float, float]]) -> List[float]:
    if len(points) < 2:
        return []
    coordinates = np.asarray(points, dtype=np.float32)
    distances = []
    for index in range(len(coordinates)):
        delta = coordinates - coordinates[index]
        norms = np.linalg.norm(delta, axis=1)
        norms[index] = np.inf
        distances.append(float(np.min(norms)))
    return distances


def build_side_by_side(original: Image.Image, overlay: Image.Image, title: str) -> Image.Image:
    padding = 20
    width, height = original.size
    canvas = Image.new('RGB', (width * 2 + padding * 2, height + padding * 2), (22, 22, 22))
    canvas.paste(original, (padding, padding))
    canvas.paste(overlay, (width + padding * 2, padding))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding + 10, 8), 'ORIGINAL', fill=(255, 255, 255))
    draw.text((width + padding * 2 + 10, 8), title, fill=(255, 255, 255))
    return canvas


def package_root(dataset_name: str) -> Path:
    return DATASET_ROOT / dataset_name / dataset_name


def segmentation_root(dataset_name: str) -> Path:
    root = package_root(dataset_name)
    if dataset_name == '01_Mendeley_Rice_Variety':
        return root / 'Rice Grain Dataset' / 'yolo_dataset'
    return root


def class_names_for(dataset_root: Path) -> List[str] | None:
    yaml_path = next(iter(sorted(dataset_root.rglob('data.yaml'))), None)
    if yaml_path is None:
        return None
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8', errors='ignore')) or {}
    names = config.get('names')
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return None


def class_name(class_id: int, names: List[str] | None) -> str:
    if names is not None and 0 <= class_id < len(names):
        return names[class_id]
    return f'class_{class_id}'


def image_files(dataset_root: Path) -> List[Path]:
    return sorted(
        path for path in dataset_root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.parent.name.lower() == 'images'
    )


def label_path_for(image_path: Path) -> Path:
    if image_path.parent.name.lower() == 'images':
        return image_path.parent.parent / 'labels' / f'{image_path.stem}.txt'
    return image_path.with_suffix('.txt')


def parse_label_file(
    label_path: Path,
    width: int,
    height: int,
    names: List[str] | None,
) -> Tuple[List[Dict[str, Any]], List[str], int, bool]:
    text = label_path.read_text(encoding='utf-8-sig', errors='replace')
    if not text.strip():
        return [], [], 0, True

    polygons: List[Dict[str, Any]] = []
    issues: List[str] = []
    invalid_lines = 0
    nonempty_lines = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().lstrip('\ufeff').split()
        if not parts:
            continue
        nonempty_lines += 1
        issue: str | None = None
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            issue = f'line_{line_number}: malformed_polygon'
        else:
            try:
                raw_class_id = float(parts[0])
                if not math.isfinite(raw_class_id) or not raw_class_id.is_integer() or raw_class_id < 0:
                    raise ValueError
                class_id = int(raw_class_id)
                values = [float(value) for value in parts[1:]]
                if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
                    issue = f'line_{line_number}: normalized_coords_out_of_range_or_nonfinite'
                elif names is not None and class_id >= len(names):
                    issue = f'line_{line_number}: class_id_out_of_range'
                else:
                    points = [(x * width, y * height) for x, y in zip(values[::2], values[1::2])]
                    area = polygon_area(points)
                    if area <= 0:
                        issue = f'line_{line_number}: polygon_area_is_zero_or_degenerate'
                    else:
                        bounds = polygon_bounds(points, width, height)
                        mask = polygon_mask_in_bounds(points, bounds)
                        if not mask.any():
                            issue = f'line_{line_number}: rasterized_mask_is_empty'
                        else:
                            polygons.append({'class_id': class_id, 'points': points, 'area': area})
            except (ValueError, OverflowError):
                issue = f'line_{line_number}: invalid_class_or_coordinate_value'
        if issue:
            invalid_lines += 1
            issues.append(issue)
    if nonempty_lines == 0:
        return [], [], 0, True
    return polygons, issues, invalid_lines, False


def geometry_flags(polygons: List[Dict[str, Any]], width: int, height: int) -> Tuple[bool, bool, bool, bool]:
    if not polygons:
        return False, False, False, False

    bounds = [polygon_bounds(poly['points'], width, height) for poly in polygons]
    ordered = sorted(range(len(polygons)), key=lambda index: bounds[index][0])
    touching = False
    overlap = False
    for order_index, first_index in enumerate(ordered):
        first_bounds = bounds[first_index]
        for second_index in ordered[order_index + 1:]:
            second_bounds = bounds[second_index]
            if second_bounds[0] > first_bounds[2] + 2:
                break
            if second_bounds[1] > first_bounds[3] + 2 or second_bounds[3] + 2 < first_bounds[1]:
                continue
            bbox_overlap = (
                max(first_bounds[0], second_bounds[0]) <= min(first_bounds[2], second_bounds[2])
                and max(first_bounds[1], second_bounds[1]) <= min(first_bounds[3], second_bounds[3])
            )
            points_a = polygons[first_index]['points']
            points_b = polygons[second_index]['points']
            if bbox_overlap and not overlap:
                overlap = pairwise_overlap(points_a, points_b, width, height)
            if not touching:
                touching = pairwise_touching(points_a, points_b, width, height)
            if touching and overlap:
                break
        if touching and overlap:
            break

    dense = False
    if len(polygons) >= 10:
        centroids = [
            (
                sum(x for x, _ in poly['points']) / len(poly['points']),
                sum(y for _, y in poly['points']) / len(poly['points']),
            )
            for poly in polygons
        ]
        nearest = nearest_centroid_distances(centroids)
        dense = bool(nearest) and float(np.mean(nearest)) <= 0.12 * math.hypot(width, height)

    separated = not (touching or overlap or dense)
    return touching, overlap, dense, separated


def audit_image(
    dataset_name: str,
    image_path: Path,
    names: List[str] | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with Image.open(image_path) as image:
        width, height = image.size
    label_path = label_path_for(image_path)
    polygons: List[Dict[str, Any]] = []
    issues: List[str] = []
    invalid_lines = 0
    empty_label = False
    missing_label = not label_path.is_file()
    if missing_label:
        issues.append('missing_label_file')
    else:
        polygons, issues, invalid_lines, empty_label = parse_label_file(label_path, width, height, names)

    touching, overlap, dense, separated = geometry_flags(polygons, width, height)
    if missing_label or empty_label or issues:
        separated = False
    per_image_classes = Counter(poly['class_id'] for poly in polygons)
    record = {
        'dataset': dataset_name,
        'image_file': image_path.relative_to(ROOT).as_posix(),
        'split': image_path.parent.parent.name,
        'image_dimensions': {'width': width, 'height': height},
        'label_file': label_path.relative_to(ROOT).as_posix() if label_path.is_file() else None,
        'valid_polygons': len(polygons),
        'class_instance_counts': {
            class_name(class_id, names): count
            for class_id, count in sorted(per_image_classes.items())
        },
        'missing_label_file': missing_label,
        'empty_label_file': empty_label,
        'invalid_annotation_lines': invalid_lines,
        'invalid_annotation': invalid_lines > 0,
        'issues': issues,
        'touching_grains': touching,
        'overlapping_grains': overlap,
        'dense_layout': dense,
        'separated_under_heuristic': separated,
        'representative_examples': [],
    }
    return record, polygons


def histogram_and_bins(values: List[int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    histogram = {str(value): count for value, count in sorted(Counter(values).items())}
    bins = {'0': 0, '1': 0, '2_to_5': 0, '6_to_10': 0, '11_to_25': 0, '26_to_50': 0, '51_plus': 0}
    for value in values:
        if value == 0:
            bins['0'] += 1
        elif value == 1:
            bins['1'] += 1
        elif 2 <= value <= 5:
            bins['2_to_5'] += 1
        elif 6 <= value <= 10:
            bins['6_to_10'] += 1
        elif 11 <= value <= 25:
            bins['11_to_25'] += 1
        elif 26 <= value <= 50:
            bins['26_to_50'] += 1
        else:
            bins['51_plus'] += 1
    return histogram, bins


def draw_overlay(
    dataset_name: str,
    category: str,
    image_path: Path,
    polygons: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    overlay = Image.new('RGBA', original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for instance_id, polygon in enumerate(polygons, start=1):
        points = polygon['points']
        color = ((instance_id * 53) % 256, (instance_id * 101) % 256, (instance_id * 191) % 256, 170)
        draw.polygon(points, fill=color, outline=(255, 255, 255, 220), width=2)
        center_x = sum(x for x, _ in points) / len(points)
        center_y = sum(y for _, y in points) / len(points)
        label = f'{instance_id}:c{polygon["class_id"]}'
        draw.text((center_x + 4, center_y + 4), label, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    composite = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
    side_by_side = build_side_by_side(original, composite, 'MASK OVERLAY + INSTANCE IDs')
    example_dir = output_dir / 'examples'
    example_dir.mkdir(parents=True, exist_ok=True)
    target = example_dir / f'{category}_{image_path.stem}_side_by_side.png'
    side_by_side.save(target)
    return target.relative_to(OUTPUT_ROOT).as_posix()


def select_examples(
    dataset_name: str,
    records: List[Dict[str, Any]],
    polygons_by_image: Dict[str, List[Dict[str, Any]]],
    image_by_record: Dict[str, Path],
    output_dir: Path,
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    def add(category: str, record: Dict[str, Any]) -> None:
        key = record['image_file']
        path = draw_overlay(dataset_name, category, image_by_record[key], polygons_by_image[key], output_dir)
        item = {'category': category, 'image_file': key, 'overlay_file': path}
        record['representative_examples'].append(item)
        examples.append(item)

    annotated = [record for record in records if record['valid_polygons'] > 0]
    if annotated:
        low_record = min(annotated, key=lambda record: (record['valid_polygons'], record['image_file']))
        median_count = median(record['valid_polygons'] for record in annotated)
        typical_record = min(annotated, key=lambda record: (abs(record['valid_polygons'] - median_count), record['image_file']))
        high_record = max(annotated, key=lambda record: (record['valid_polygons'], record['image_file']))
        add('low_instances', low_record)
        add('typical_instances', typical_record)
        add('high_instances', high_record)

    for flag, category in [
        ('touching_grains', 'touching_example'),
        ('overlapping_grains', 'overlap_example'),
        ('dense_layout', 'dense_example'),
    ]:
        flagged = next((record for record in records if record[flag]), None)
        if flagged is not None:
            add(category, flagged)

    problematic = next(
        (
            record for record in records
            if record['empty_label_file'] or record['missing_label_file'] or record['invalid_annotation']
        ),
        None,
    )
    if problematic is not None:
        add('empty_or_problematic', problematic)

    return examples


def inspect_mendeley_branches() -> Dict[str, Any]:
    root = package_root('01_Mendeley_Rice_Variety') / 'Rice Grain Dataset'
    image_classes: Dict[str, Dict[str, int]] = {}
    cnn_root = root / 'cnn_dataset'
    for split in ['train', 'val', 'test']:
        split_root = cnn_root / split
        counts: Dict[str, int] = {}
        if split_root.is_dir():
            for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                counts[class_dir.name] = sum(
                    1 for path in class_dir.rglob('*')
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                )
        image_classes[split] = counts

    feature_tables: Dict[str, Any] = {}
    xgb_root = root / 'xgb_dataset'
    for csv_path in sorted(xgb_root.glob('*.csv')):
        with csv_path.open('r', encoding='utf-8-sig', errors='replace', newline='') as source:
            reader = csv.DictReader(source)
            label_counts = Counter(row.get('label', '') for row in reader)
            columns = reader.fieldnames or []
        feature_tables[csv_path.name] = {
            'rows': sum(label_counts.values()),
            'columns': columns,
            'label_distribution': dict(sorted(label_counts.items())),
        }
    return {
        'segmentation_branch': 'Rice Grain Dataset/yolo_dataset',
        'cnn_classification_images_by_split': image_classes,
        'cnn_classification_image_count': sum(sum(counts.values()) for counts in image_classes.values()),
        'xgboost_feature_tables': feature_tables,
        'xgboost_feature_row_count': sum(table['rows'] for table in feature_tables.values()),
        'segmentation_class_name_mapping': 'No data.yaml or class-name mapping found in the YOLO branch; numeric class IDs are reported as stored.',
    }


def audit_dataset(dataset_name: str) -> Dict[str, Any]:
    dataset_root = segmentation_root(dataset_name)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Missing segmentation source directory: {dataset_root}')
    images = image_files(dataset_root)
    if not images:
        raise RuntimeError(f'No source images found under: {dataset_root}')
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names_for(dataset_root)
    label_files = {
        path for labels_dir in dataset_root.rglob('labels') if labels_dir.is_dir()
        for path in labels_dir.rglob('*.txt') if path.is_file()
    }

    records: List[Dict[str, Any]] = []
    polygons_by_image: Dict[str, List[Dict[str, Any]]] = {}
    image_by_record: Dict[str, Path] = {}
    class_counts: Counter[int] = Counter()
    dimension_counts: Counter[str] = Counter()
    matched_labels: set[Path] = set()
    for image_path in images:
        try:
            record, polygons = audit_image(dataset_name, image_path, names)
        except Exception as exc:
            raise RuntimeError(f'Failed to audit {image_path}: {exc}') from exc
        records.append(record)
        polygons_by_image[record['image_file']] = polygons
        image_by_record[record['image_file']] = image_path
        class_counts.update(polygon['class_id'] for polygon in polygons)
        dimensions = record['image_dimensions']
        dimension_counts[f'{dimensions["width"]}x{dimensions["height"]}'] += 1
        if record['label_file']:
            matched_labels.add(ROOT / Path(record['label_file']))

    values = [record['valid_polygons'] for record in records]
    histogram, bins = histogram_and_bins(values)
    configured_class_counts = {
        class_name(class_id, names): class_counts[class_id]
        for class_id in range(len(names))
    } if names is not None else {}
    for class_id, count in sorted(class_counts.items()):
        configured_class_counts.setdefault(class_name(class_id, names), count)

    missing = sum(record['missing_label_file'] for record in records)
    empty = sum(record['empty_label_file'] for record in records)
    invalid_images = sum(record['invalid_annotation'] for record in records)
    invalid_lines = sum(record['invalid_annotation_lines'] for record in records)
    touching = sum(record['touching_grains'] for record in records)
    overlap = sum(record['overlapping_grains'] for record in records)
    dense = sum(record['dense_layout'] for record in records)
    separated = sum(record['separated_under_heuristic'] for record in records)
    examples = select_examples(dataset_name, records, polygons_by_image, image_by_record, output_dir)

    summary: Dict[str, Any] = {
        'dataset': dataset_name,
        'segmentation_branch': dataset_root.relative_to(ROOT).as_posix(),
        'total_images': len(records),
        'total_valid_polygons': sum(values),
        'min_polygons_per_image': min(values) if values else 0,
        'max_polygons_per_image': max(values) if values else 0,
        'mean_polygons_per_image': round(mean(values), 2) if values else 0.0,
        'median_polygons_per_image': round(float(median(values)), 2) if values else 0.0,
        'polygons_per_image_histogram': histogram,
        'image_count_bins': bins,
        'image_dimensions_distribution': dict(sorted(dimension_counts.items())),
        'class_names_from_metadata': names,
        'class_instance_counts': configured_class_counts,
        'missing_label_files': missing,
        'invalid_annotation_images': invalid_images,
        'invalid_annotation_lines': invalid_lines,
        'empty_label_files': empty,
        'unmatched_label_files': len(label_files - matched_labels),
        'images_with_touching_masks': touching,
        'images_with_overlapping_masks': overlap,
        'images_with_dense_layout': dense,
        'images_separated_under_heuristic': separated,
        'representative_examples': examples,
        'records': records,
    }
    if dataset_name == '01_Mendeley_Rice_Variety':
        summary['other_package_branches'] = inspect_mendeley_branches()
    with (output_dir / 'full_summary.json').open('w', encoding='utf-8') as destination:
        json.dump(summary, destination, indent=2)
    return summary


def compatibility(summary: Dict[str, Any]) -> Tuple[str, str, str]:
    total = summary['total_images']
    separated = summary['images_separated_under_heuristic']
    ratio = separated / total if total else 0.0
    text = f'{separated}/{total} ({ratio:.1%})' if total else 'No segmentation images'
    if total == 0 or summary['total_valid_polygons'] == 0:
        similarity = 'Undetermined'
        role = 'Not suitable for current segmentation phase'
    elif ratio >= 0.9:
        similarity = f'High by geometry heuristic ({ratio:.1%})'
        role = 'Strong candidate'
    elif ratio >= 0.5:
        similarity = f'Mixed ({ratio:.1%} separated heuristic)' 
        role = 'Possible supplementary candidate'
    elif ratio > 0 or summary['empty_label_files'] or summary['invalid_annotation_images']:
        similarity = f'Low ({ratio:.1%} separated heuristic)'
        role = 'Needs filtering'
    else:
        similarity = 'Low by geometry heuristic'
        role = 'Poor match'
    return text, similarity, role


def class_structure(dataset_name: str, summary: Dict[str, Any]) -> str:
    if dataset_name == '01_Mendeley_Rice_Variety':
        return 'YOLO segmentation IDs 0/1/2 (names unmapped); separate CNN variety folders and XGBoost feature tables'
    names = summary['class_names_from_metadata'] or []
    return ', '.join(names) if names else 'No class names in metadata'


def potential_role(dataset_name: str, summary: Dict[str, Any]) -> str:
    _, _, role = compatibility(summary)
    if dataset_name == '03_Rice_Variety' and role == 'Strong candidate':
        return 'Possible supplementary candidate'
    return role


def single_grain_assessment(summary: Dict[str, Any]) -> str:
    total = summary['total_images']
    if not total:
        return 'Undetermined; no segmentation images'
    ratio = summary['images_separated_under_heuristic'] / total
    if ratio >= 0.9:
        return 'Geometry supports separated polygon instances; visible completeness unverified'
    if ratio >= 0.5:
        return 'Mixed layout; inspect touching, overlap, and dense examples'
    return 'Not consistently separated under the heuristic'


def write_comparison(summaries: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        '# Full Read-Only Audit: Datasets 01–05',
        '',
        'Audit date: 2026-09-29.',
        '',
        'All counts below are from every image in each local YOLO segmentation branch. Dataset 01 also has separate classification and feature-table branches, reported separately so formats are not conflated. No source dataset, ZIP, model, frontend, or deployment file was modified. No training, merging, or annotation conversion was performed.',
        '',
        '## Shared methodology',
        '',
        '- Parse every label file paired by exact image stem within its split; retain source class IDs and metadata names without remapping.',
        '- Validate YOLO polygon row structure, class IDs, finite normalized coordinates in [0, 1], nonzero polygon area, and nonempty rasterized masks.',
        '- Convert normalized coordinates to image pixels and rasterize masks in a shared image coordinate frame.',
        '- Use the same pairwise mask-overlap test and two-sided 3x3 dilation (two-pixel proximity) touching heuristic as the 06/08 audit.',
        '- Use mean nearest-centroid distance <= 12% of image diagonal for dense layout, only when an image has at least 10 polygons, matching the 06/08 implementation.',
        '- A separated image has valid polygon annotations and no touching, overlap, or dense-layout flag. This is a geometry heuristic, not proof of one polygon per visible grain.',
        '- Image dimensions and all per-image flags/counts are included in each dataset JSON. Representative overlays show original on the left and color masks with instance IDs/class IDs on the right.',
        '',
        '## Comparison: geometry and annotation health',
        '',
        '| Dataset | Images | Instances | Mean grains/image | Touching | Overlap | Dense | Invalid | Empty | Separated compatibility |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for name in TARGET_DATASETS:
        summary = summaries[name]
        separated, _, _ = compatibility(summary)
        lines.append(
            f'| {name} | {summary["total_images"]} | {summary["total_valid_polygons"]} | '
            f'{summary["mean_polygons_per_image"]:.2f} | {summary["images_with_touching_masks"]} | '
            f'{summary["images_with_overlapping_masks"]} | {summary["images_with_dense_layout"]} | '
            f'{summary["invalid_annotation_images"]} | {summary["empty_label_files"]} | {separated} |'
        )
    lines.extend([
        '',
        'Invalid is the number of images with one or more malformed annotation lines; empty is the number of empty label files. Missing image-label pairs are reported in each dataset detail section. Geometry-flag counts can overlap and should not be added together.',
        '',
        '## Comparison: class meaning and intended input fit',
        '',
        '| Dataset | Class structure | Single-grain polygons? | Professor setup similarity | Potential role |',
        '|---|---|---|---|---|',
    ])
    for name in TARGET_DATASETS:
        summary = summaries[name]
        separated, similarity, _ = compatibility(summary)
        role = potential_role(name, summary)
        classes = class_structure(name, summary).replace('|', '\\|')
        lines.append(
            f'| {name} | {classes} | {single_grain_assessment(summary)} | {similarity}; {separated} | {role} |'
        )

    for name in TARGET_DATASETS:
        summary = summaries[name]
        output_folder = name
        lines.extend([
            '',
            f'## {name}',
            '',
            f'- Segmentation branch: `{summary["segmentation_branch"]}`',
            f'- Images: {summary["total_images"]}; valid polygons: {summary["total_valid_polygons"]}',
            f'- Polygons/image: min {summary["min_polygons_per_image"]}, max {summary["max_polygons_per_image"]}, mean {summary["mean_polygons_per_image"]:.2f}, median {summary["median_polygons_per_image"]:.2f}',
            f'- Polygon-count histogram: `{summary["polygons_per_image_histogram"]}`',
            f'- Image-count bins: `{summary["image_count_bins"]}`',
            f'- Image dimensions (width x height: count): `{summary["image_dimensions_distribution"]}`',
            f'- Missing label files: {summary["missing_label_files"]}; empty label files: {summary["empty_label_files"]}; invalid annotation images: {summary["invalid_annotation_images"]}; invalid lines: {summary["invalid_annotation_lines"]}; unmatched label files: {summary["unmatched_label_files"]}',
            f'- Class instance counts: `{summary["class_instance_counts"]}`',
            f'- Touching: {summary["images_with_touching_masks"]}; overlap: {summary["images_with_overlapping_masks"]}; dense: {summary["images_with_dense_layout"]}; separated: {summary["images_separated_under_heuristic"]}/{summary["total_images"]}',
            f'- Per-image summary: [{name}/full_summary.json]({output_folder}/full_summary.json)',
            '',
            '### Representative overlays',
            '',
        ])
        if summary['representative_examples']:
            for example in summary['representative_examples']:
                relative_overlay = Path(example['overlay_file']).as_posix()
                lines.append(
                    f'- {example["category"]}: [{Path(example["overlay_file"]).name}]({relative_overlay}) '
                    f'for `{example["image_file"]}`'
                )
        else:
            lines.append('No valid polygons were available to render representative overlays.')

        if name == '01_Mendeley_Rice_Variety':
            branches = summary['other_package_branches']
            lines.extend([
                '',
                '### Other package branches (not segmentation instances)',
                '',
                f'- CNN classification branch: {branches["cnn_classification_image_count"]} images. Per-split class folders: `{branches["cnn_classification_images_by_split"]}`.',
                f'- XGBoost feature branch: {branches["xgboost_feature_row_count"]} rows across CSVs; class counts by table: `{ {name: table["label_distribution"] for name, table in branches["xgboost_feature_tables"].items()} }`. These are feature rows, not images or polygon instances.',
                f'- YOLO branch mapping: {branches["segmentation_class_name_mapping"]}',
            ])

        lines.extend(['', '### Class and setup interpretation', ''])
        if name == '02_Grainalyze':
            lines.append('The configured classes are grain-condition/quality labels (`broken_grain`, `chalky_grain`, `discolored_grain`, `whole_grain`). They are reported as-is. Whether each polygon corresponds to exactly one physical grain must be confirmed from overlays, especially where touching or overlap flags occur.')
        elif name == '03_Rice_Variety':
            lines.append('The configured labels are rice-variety names, not a generic rice-grain class. Preserve these class IDs and names; they must not be treated as interchangeable with a generic grain label. The separated-layout rate describes mask geometry only.')
        elif name == '04_RiceVigor':
            lines.append('The configured class is `rice-seed`. This is the only single-class YOLO branch among these five; empty labels, if any, are counted separately from malformed annotations.')
        elif name == '05_Ricee':
            lines.append('The configured classes are `broken`, `discolored`, `foreignObject`, `long`, and `medium`. These names are preserved exactly. Empty labels are reported separately and may require filtering or manual review for the current individual-grain objective.')
        else:
            lines.append('The YOLO branch contains numeric classes 0, 1, and 2 but no class-name map was found there. The separate CNN class folders and XGBoost labels are evidence of additional variety-classification formats, but no mapping from those names to YOLO IDs is assumed.')
        lines.extend([
            '',
            'The polygon separation result cannot establish visual annotation completeness or prove one polygon per visible grain. Review the low/typical/high and flagged overlays before using these labels for instance counting.',
        ])

    lines.extend([
        '',
        '## Decision boundary',
        '',
        'The potential-role labels above are descriptive evidence summaries only. This audit does not select a final training combination, normalize class IDs, merge datasets, convert labels, or train a model. The next comparison can incorporate datasets 06–08 before any training decision.',
        '',
    ])
    (OUTPUT_ROOT / 'datasets_01_to_05_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def write_blocked_artifacts(missing_sources: List[Tuple[str, Path, str]]) -> None:
    reason = 'Audit not run: the extracted source directories are absent or contain no images. No statistics or class conclusions are available.'
    for dataset_name, source_root, detail in missing_sources:
        output_dir = OUTPUT_ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked_summary = {
            'dataset': dataset_name,
            'status': 'BLOCKED',
            'audit_completed': False,
            'expected_source_directory': source_root.as_posix(),
            'blocker': detail,
            'note': reason,
        }
        with (output_dir / 'full_summary.json').open('w', encoding='utf-8') as destination:
            json.dump(blocked_summary, destination, indent=2)

    lines = [
        '# Full Read-Only Audit: Datasets 01–05',
        '',
        '**Status: BLOCKED.** No audit statistics were computed because the extracted source directories are unavailable in the current workspace.',
        '',
        'The audit runner found only `data/datasets/07_Raw_Rice_Seed`; the five requested source folders are absent. Zero-image output is not a valid audit result. No source files, ZIP files, model code, frontend, or deployment files were modified. No training, merging, or annotation conversion was performed.',
        '',
        '## Missing sources',
        '',
        '| Dataset | Expected source directory | Status |',
        '|---|---|---|',
    ]
    for dataset_name, source_root, detail in missing_sources:
        lines.append(f'| {dataset_name} | `{source_root.as_posix()}` | {detail} |')
    lines.extend([
        '',
        '## Comparison tables',
        '',
        '| Dataset | Images | Instances | Mean grains/image | Touching | Overlap | Dense | Invalid | Empty | Separated compatibility |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ])
    for dataset_name in TARGET_DATASETS:
        lines.append(f'| {dataset_name} | Not audited | Not audited | Not audited | Not audited | Not audited | Not audited | Not audited | Not audited | Undetermined |')
    lines.extend([
        '',
        '| Dataset | Class structure | Single-grain polygons? | Professor setup similarity | Potential role |',
        '|---|---|---|---|---|',
    ])
    for dataset_name in TARGET_DATASETS:
        lines.append(f'| {dataset_name} | Not audited | Undetermined | Undetermined | Undetermined |')
    lines.extend([
        '',
        'No dataset suitability or training-combination decision is made. Restore or point the audit runner at the extracted source directories, then rerun it to populate these tables, JSON summaries, and representative overlays.',
        '',
    ])
    (OUTPUT_ROOT / 'datasets_01_to_05_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    missing_sources: List[Tuple[str, Path, str]] = []
    for dataset_name in TARGET_DATASETS:
        source_root = segmentation_root(dataset_name)
        if not source_root.is_dir():
            missing_sources.append((dataset_name, source_root, 'source directory missing'))
        elif not image_files(source_root):
            missing_sources.append((dataset_name, source_root, 'no images found'))
    if missing_sources:
        write_blocked_artifacts(missing_sources)
        print('Audit blocked: requested extracted source directories are unavailable.')
        for name, source_root, detail in missing_sources:
            print(name, detail, source_root)
        print('Wrote blocked status under:', OUTPUT_ROOT)
        return

    summaries = {name: audit_dataset(name) for name in TARGET_DATASETS}
    write_comparison(summaries)
    print('Generated comparison:', OUTPUT_ROOT / 'datasets_01_to_05_comparison.md')
    for name in TARGET_DATASETS:
        summary = summaries[name]
        print(
            name,
            'images=', summary['total_images'],
            'polygons=', summary['total_valid_polygons'],
            'mean=', summary['mean_polygons_per_image'],
            'touching=', summary['images_with_touching_masks'],
            'overlap=', summary['images_with_overlapping_masks'],
            'dense=', summary['images_with_dense_layout'],
            'empty=', summary['empty_label_files'],
            'separated=', summary['images_separated_under_heuristic'],
        )


if __name__ == '__main__':
    main()