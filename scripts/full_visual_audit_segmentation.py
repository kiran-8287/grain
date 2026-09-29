from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path('A:/grain/rice-quality-ai')
DATASET_ROOT = ROOT / 'data' / 'datasets'
OUTPUT_ROOT = ROOT / 'data' / 'audit' / 'full_visual_segmentation'
TARGET_DATASETS = ['06_Rice_Grain_Segmentation', '08_Rice_Grain']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def resolve_dataset_path(dataset_name: str) -> Path:
    direct = DATASET_ROOT / dataset_name
    if direct.exists():
        return direct
    matches = [p for p in DATASET_ROOT.glob(f'{dataset_name}*') if p.is_dir()]
    for candidate in matches:
        if (candidate / 'data.yaml').exists() or any((candidate / s).exists() for s in ['train', 'valid', 'test']):
            return candidate
        nested = candidate / dataset_name
        if nested.exists() and ((nested / 'data.yaml').exists() or any((nested / s).exists() for s in ['train', 'valid', 'test'])):
            return nested
    raise FileNotFoundError(dataset_name)


def iter_images(dataset_root: Path) -> List[Path]:
    out: List[Path] = []
    for path in dataset_root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
    return sorted(out)


def find_label_path(dataset_root: Path, image_path: Path) -> Path | None:
    direct = image_path.with_suffix('.txt')
    if direct.exists():
        return direct
    stem = image_path.stem
    candidates: List[Path] = []
    for label_path in dataset_root.rglob('*.txt'):
        if label_path.stem == stem:
            candidates.append(label_path)
        else:
            base = stem.split('.rf')[0] if '.rf.' in stem else stem
            if label_path.stem.startswith(base):
                candidates.append(label_path)
    return sorted(candidates)[0] if candidates else None


def polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_annotations(label_path: Path | None, width: int, height: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    if label_path is None or not label_path.exists():
        return [], ['missing_label_file']
    try:
        lines = label_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return [], ['unreadable_label_file']
    polygons: List[Dict[str, Any]] = []
    issues: List[str] = []
    for idx, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            issues.append(f'line_{idx}: too_few_fields')
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            issues.append(f'line_{idx}: invalid_class_id')
            continue
        values = [float(v) for v in parts[1:]]
        if len(values) % 2 != 0 or len(values) < 6:
            issues.append(f'line_{idx}: malformed_polygon')
            continue
        coords = list(zip(values[::2], values[1::2]))
        if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in coords):
            issues.append(f'line_{idx}: normalized_coords_out_of_range')
            continue
        points = [(x * width, y * height) for x, y in coords]
        if len(points) < 3:
            issues.append(f'line_{idx}: polygon_has_fewer_than_3_points')
            continue
        if polygon_area(points) <= 0:
            issues.append(f'line_{idx}: polygon_area_is_zero_or_degenerate')
            continue
        polygons.append({'class_id': class_id, 'points': points})
    if not polygons:
        issues.append('no_valid_polygons')
    return polygons, issues


def polygon_bounds(points: List[Tuple[float, float]], width: int, height: int, padding: int = 0) -> Tuple[int, int, int, int]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return (
        max(0, int(math.floor(min(xs))) - padding),
        max(0, int(math.floor(min(ys))) - padding),
        min(width - 1, int(math.ceil(max(xs))) + padding),
        min(height - 1, int(math.ceil(max(ys))) + padding),
    )


def polygon_mask_in_bounds(points: List[Tuple[float, float]], bounds: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bounds
    img = Image.new('1', (x_max - x_min + 1, y_max - y_min + 1), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(x - x_min, y - y_min) for x, y in points], fill=1)
    return np.array(img, dtype=np.uint8)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if mask.size == 0:
        return mask
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    dil = img.filter(ImageFilter.MaxFilter(3))
    return np.array(dil, dtype=np.uint8) > 0


def pairwise_overlap(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], width: int, height: int) -> bool:
    bounds_a = polygon_bounds(points_a, width, height)
    bounds_b = polygon_bounds(points_b, width, height)
    shared = (
        max(bounds_a[0], bounds_b[0]),
        max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]),
        min(bounds_a[3], bounds_b[3]),
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
        max(bounds_a[0], bounds_b[0]),
        max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]),
        min(bounds_a[3], bounds_b[3]),
    )
    if shared[0] > shared[2] or shared[1] > shared[3]:
        return False
    mask_a = polygon_mask_in_bounds(points_a, shared)
    mask_b = polygon_mask_in_bounds(points_b, shared)
    return bool(np.logical_and(dilate(mask_a), dilate(mask_b)).any())


def nearest_centroid_distances(points: List[Tuple[float, float]]) -> List[float]:
    if len(points) < 2:
        return []
    arr = np.array(points, dtype=np.float32)
    dists: List[float] = []
    for i in range(len(arr)):
        diff = arr - arr[i]
        distances = np.linalg.norm(diff, axis=1)
        distances[i] = np.inf
        dists.append(float(np.min(distances)))
    return dists


def build_side_by_side(original: Image.Image, overlay: Image.Image, title: str = 'OVERLAY') -> Image.Image:
    pad = 20
    w1, h1 = original.size
    w2, h2 = overlay.size
    canvas = Image.new('RGB', (w1 + w2 + pad * 2, max(h1, h2) + pad * 2), (22, 22, 22))
    canvas.paste(original, (pad, pad))
    canvas.paste(overlay, (w1 + pad * 2, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 10, 8), 'ORIGINAL', fill=(255, 255, 255))
    draw.text((w1 + pad * 2 + 10, 8), title, fill=(255, 255, 255))
    return canvas


def save_example(dataset_name: str, image_path: Path, image: Image.Image, polygons: List[Dict[str, Any]], reason: str, output_dir: Path) -> str:
    overlay = image.convert('RGBA')
    alpha = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(alpha)
    for i, poly in enumerate(polygons, start=1):
        points = [(x, y) for x, y in poly['points']]
        draw.polygon(points, fill=((i * 53) % 256, (i * 101) % 256, (i * 191) % 256, 170), outline=(255, 255, 255, 220), width=2)
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        draw.text((cx + 4, cy + 4), str(i), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    composite = Image.alpha_composite(overlay, alpha).convert('RGB')
    side = build_side_by_side(image, composite, 'OVERLAY')
    example_dir = output_dir / 'verified' / dataset_name / 'examples'
    example_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{reason}_{image_path.stem}_side_by_side.png'
    target = example_dir / filename
    side.save(target)
    return str(target.relative_to(ROOT))


def compact_histogram(values: List[int]) -> Dict[str, Any]:
    hist = {str(k): v for k, v in sorted(Counter(values).items())}
    bins = {'0': 0, '1': 0, '2_to_5': 0, '6_to_10': 0, '11_to_25': 0, '26_to_50': 0, '51_plus': 0}
    for v in values:
        if v == 0:
            bins['0'] += 1
        elif v == 1:
            bins['1'] += 1
        elif 2 <= v <= 5:
            bins['2_to_5'] += 1
        elif 6 <= v <= 10:
            bins['6_to_10'] += 1
        elif 11 <= v <= 25:
            bins['11_to_25'] += 1
        elif 26 <= v <= 50:
            bins['26_to_50'] += 1
        elif v >= 51:
            bins['51_plus'] += 1
    return {'histogram': hist, 'bins': bins}


def audit_image(dataset_name: str, image_path: Path, dataset_root: Path, output_dir: Path, example_counter: Dict[str, int]) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
        original = image.convert('RGB')
    label_path = find_label_path(dataset_root, image_path)
    polygons, issues = parse_annotations(label_path, width, height)
    touching = False
    overlap = False
    dense_cluster = False
    alignment_ok = True
    annotations_form_separated_layout = True
    if polygons:
        for poly in polygons:
            if not all(0 <= x <= width and 0 <= y <= height for x, y in poly['points']):
                alignment_ok = False
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                points_a = polygons[i]['points']
                points_b = polygons[j]['points']
                if pairwise_overlap(points_a, points_b, width, height):
                    overlap = True
                if pairwise_touching(points_a, points_b, width, height):
                    touching = True
        if len(polygons) >= 10:
            centroids = [
                (sum(x for x, _ in poly['points']) / len(poly['points']), sum(y for _, y in poly['points']) / len(poly['points']))
                for poly in polygons
            ]
            nearest = nearest_centroid_distances(centroids)
            if nearest:
                avg_nn = float(np.mean(nearest))
                dense_cluster = avg_nn <= 0.12 * math.hypot(width, height)
        annotations_form_separated_layout = not (touching or overlap or dense_cluster) and not issues and alignment_ok
    else:
        alignment_ok = False
        annotations_form_separated_layout = False
    annotation_valid = not issues and alignment_ok
    separated = annotations_form_separated_layout and annotation_valid

    example_paths: List[Tuple[str, str]] = []
    if touching and example_counter['touching'] < 3:
        example_paths.append(('touching_grains', save_example(dataset_name, image_path, original, polygons, 'touching_grains', output_dir)))
        example_counter['touching'] += 1
    if overlap and example_counter['overlap'] < 3:
        example_paths.append(('overlapping_grains', save_example(dataset_name, image_path, original, polygons, 'overlapping_grains', output_dir)))
        example_counter['overlap'] += 1
    if dense_cluster and example_counter['dense'] < 3:
        example_paths.append(('dense_cluster', save_example(dataset_name, image_path, original, polygons, 'dense_cluster', output_dir)))
        example_counter['dense'] += 1
    if (issues or not annotations_form_separated_layout) and example_counter['suspicious'] < 3:
        example_paths.append(('suspicious_annotation', save_example(dataset_name, image_path, original, polygons, 'suspicious_annotation', output_dir)))
        example_counter['suspicious'] += 1
    if polygons and example_counter['representative'] < 2:
        example_paths.append(('representative_overlay', save_example(dataset_name, image_path, original, polygons, 'representative_overlay', output_dir)))
        example_counter['representative'] += 1

    return {
        'dataset': dataset_name,
        'image_file': str(image_path.relative_to(ROOT)),
        'image_dimensions': {'width': width, 'height': height},
        'annotated_instances': len(polygons),
        'annotation_valid': annotation_valid,
        'issues': issues,
        'touching_grains': touching,
        'overlapping_grains': overlap,
        'dense_clusters': dense_cluster,
        'separated_grains': separated,
        'annotations_form_separated_layout': annotations_form_separated_layout,
        'polygon_mask_alignment_ok': alignment_ok,
        'representative_example_paths': example_paths,
    }


def summarize_dataset(dataset_name: str, dataset_root: Path, output_dir: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    example_counter = {'touching': 0, 'overlap': 0, 'dense': 0, 'suspicious': 0, 'representative': 0}
    for image_path in iter_images(dataset_root):
        try:
            records.append(audit_image(dataset_name, image_path, dataset_root, output_dir, example_counter))
        except Exception as exc:
            raise RuntimeError(f'Failed to audit {image_path}: {exc}') from exc
    counts = [r['annotated_instances'] for r in records]
    hist = compact_histogram(counts)
    summary = {
        'dataset': dataset_name,
        'total_images': len(records),
        'total_annotated_instances': sum(counts),
        'min_grains_per_image': min(counts) if counts else 0,
        'max_grains_per_image': max(counts) if counts else 0,
        'mean_grains_per_image': round(mean(counts), 2) if counts else 0.0,
        'median_grains_per_image': round(float(median(counts)), 2) if counts else 0.0,
        'histogram': hist['histogram'],
        'grain_bins': hist['bins'],
        'images_with_touching_grains': sum(1 for r in records if r['touching_grains']),
        'images_with_overlapping_grains': sum(1 for r in records if r['overlapping_grains']),
        'images_with_dense_clusters': sum(1 for r in records if r['dense_clusters']),
        'images_with_missing_label_files': sum(1 for r in records if 'missing_label_file' in r['issues']),
        'images_with_invalid_annotations': sum(1 for r in records if r['issues'] and 'missing_label_file' not in r['issues']),
        'images_requiring_visual_review': sum(1 for r in records if not r['annotation_valid'] or r['touching_grains'] or r['overlapping_grains'] or r['dense_clusters']),
        'images_with_separated_annotation_layout': sum(1 for r in records if r['separated_grains']),
        'records': records,
    }
    with (output_dir / f'{dataset_name}_full_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    return summary


def write_report(report_path: Path, summaries: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        '# Full Visual + Statistical Audit for 06 and 08',
        '',
        '## Scope',
        '',
        'This audit was run read-only on the extracted local datasets only. No dataset files, ZIP files, or folder contents were modified.',
        '',
        '## Method',
        '',
        '- Matched each image to its YOLO polygon label file and counted valid polygon instances.',
        '- Counted annotations image-by-image.',
        '- Validated polygon geometry and normalized coordinate ranges; rasterized masks in a shared image coordinate frame.',
        '- Flagged raster-mask overlap, masks within two pixels (touching heuristic), and dense centroid layouts (mean nearest-centroid distance at most 12% of the image diagonal).',
        '- These structural checks cannot prove that every visible grain was annotated; representative overlays are provided for visual review.',
        '- Saved all output under the audit folder only.',
        '',
    ]
    for dataset_name in TARGET_DATASETS:
        s = summaries[dataset_name]
        lines.extend([
            f'## Dataset: {dataset_name}',
            '',
            f'- Total images: {s["total_images"]}',
            f'- Total annotated instances: {s["total_annotated_instances"]}',
            f'- Min grains/image: {s["min_grains_per_image"]}',
            f'- Max grains/image: {s["max_grains_per_image"]}',
            f'- Mean grains/image: {s["mean_grains_per_image"]}',
            f'- Median grains/image: {s["median_grains_per_image"]}',
            '',
            '### Grain-count distribution',
            '',
            f'- Histogram: {s["histogram"]}',
            f'- 0 annotated grains: {s["grain_bins"]["0"]} images',
            f'- 1 grain: {s["grain_bins"]["1"]} images',
            f'- 2–5 grains: {s["grain_bins"]["2_to_5"]} images',
            f'- 6–10 grains: {s["grain_bins"]["6_to_10"]} images',
            f'- 11–25 grains: {s["grain_bins"]["11_to_25"]} images',
            f'- 26–50 grains: {s["grain_bins"]["26_to_50"]} images',
            f'- 51+ grains: {s["grain_bins"]["51_plus"]} images',
            '',
            '### Defect counts',
            '',
            f'- Images with touching grains: {s["images_with_touching_grains"]}',
            f'- Images with overlapping grains: {s["images_with_overlapping_grains"]}',
            f'- Images with dense clusters: {s["images_with_dense_clusters"]}',
            f'- Images missing a label file: {s["images_with_missing_label_files"]}',
            f'- Images with invalid annotation data: {s["images_with_invalid_annotations"]}',
            f'- Images requiring visual review: {s["images_requiring_visual_review"]}',
            f'- Images with separated annotation layouts: {s["images_with_separated_annotation_layout"]}',
            '',
            '### Representative examples',
            '',
        ])
        combined: List[Tuple[str, str]] = []
        for record in s['records']:
            if record['representative_example_paths']:
                combined.extend(record['representative_example_paths'])
            if len(combined) >= 4:
                break
        if combined:
            for category, path in combined[:4]:
                normalized_path = path.replace('\\', '/')
                relative_path = normalized_path.split('data/audit/full_visual_segmentation/', 1)[-1]
                lines.append(f'- {category}: [{Path(relative_path).name}]({relative_path})')
        else:
            lines.append('No explicit problematic examples were found in this dataset.')
        lines.extend([
            '',
            '### Assessment',
            '',
            f'Automated checks found separated annotation layouts in {s["images_with_separated_annotation_layout"]} of {s["total_images"]} images. This is a polygon-geometry heuristic, not proof that every visible grain has exactly one annotation; inspect the overlays before drawing that conclusion.',
        ])
        lines.extend(['', '---', ''])
    lines.extend([
        '## Interpretation',
        '',
        '- Compare per-image counts and geometry flags above; the audit does not infer dataset suitability from names or pre-set assumptions.',
        '- Visual overlays can expose annotation alignment and grain separation, but confirming missing annotations requires checking visible grains in the source images.',
        '- No training, dataset merging, conversion, or original dataset modification was performed.',
        '',
    ])
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, Any]] = {}
    for dataset_name in TARGET_DATASETS:
        dataset_root = resolve_dataset_path(dataset_name)
        summaries[dataset_name] = summarize_dataset(dataset_name, dataset_root, OUTPUT_ROOT)
    write_report(OUTPUT_ROOT / 'dataset_full_audit_report.md', summaries)
    print('Generated audit under:', OUTPUT_ROOT)
    for dataset_name in TARGET_DATASETS:
        s = summaries[dataset_name]
        print(dataset_name, s['total_images'], s['total_annotated_instances'], s['mean_grains_per_image'])


if __name__ == '__main__':
    main()
