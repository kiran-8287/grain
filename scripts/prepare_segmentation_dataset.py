from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import imagehash
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / 'data' / 'audit' / 'full_visual_segmentation'
SOURCE_ROOT = ROOT / 'data' / 'datasets'
PREPARED_ROOT = ROOT / 'data' / 'prepared' / 'segmentation'
DATASETS = [
    '01_Mendeley_Rice_Variety',
    '02_Grainalyze',
    '03_Rice_Variety',
    '04_RiceVigor',
    '05_Ricee',
    '06_Rice_Grain_Segmentation',
    '07_Raw_Rice_Seed',
    '08_Rice_Grain',
]
SELECTED_DATASET = '06_Rice_Grain_Segmentation'
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = 'rice_grain'
SPLIT_SEED = 42
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
PHASH_DUPLICATE_DISTANCE = 4
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def normalized_path(value: str | None) -> str:
    return value.replace('\\', '/') if value else ''


def class_ids_for_record(record: Dict[str, Any], class_names: List[str] | None) -> List[int]:
    ids: set[int] = set()
    for class_name in record.get('class_instance_counts', {}):
        if class_names and class_name in class_names:
            ids.add(class_names.index(class_name))
        elif class_name.startswith('class_'):
            try:
                ids.add(int(class_name.removeprefix('class_')))
            except ValueError:
                continue
    return sorted(ids)


def exclusion_details(dataset: str, record: Dict[str, Any]) -> Tuple[str, str]:
    if dataset == SELECTED_DATASET:
        if record.get('missing_label_file'):
            return 'missing_label', 'Missing source label file.'
        if record.get('empty_label_file'):
            return 'empty_label', 'Empty label excluded from positive training data; it is not assumed to be a negative image.'
        if record.get('invalid_annotation'):
            return 'invalid_annotation', 'Annotation failed validation.'
        if not record.get('separated_under_heuristic'):
            return 'geometry_filter', 'Failed the separated-layout heuristic.'
        return '', ''

    if dataset == '04_RiceVigor':
        if record.get('empty_label_file'):
            return 'empty_label', 'Empty label excluded from positive training; not retained as a negative without visual review.'
        return 'mapping_pending', "Source class 'rice-seed' is not automatically mapped to target 'rice_grain'; explicit approval is pending."
    if dataset == '03_Rice_Variety':
        return 'variety_semantics', 'Source classes identify rice varieties; no mapping to generic rice_grain is approved.'
    if dataset == '02_Grainalyze':
        if record.get('touching_grains') or record.get('overlapping_grains') or record.get('dense_layout'):
            return 'geometry_filter', 'Touching, overlap, or dense-layout flag; excluded from the initial separated-grain set.'
        return 'quality_semantics_review', 'Geometry passes, but quality-condition classes and one-polygon-per-visible-grain semantics need review before inclusion.'
    if dataset == '05_Ricee':
        if record.get('empty_label_file'):
            return 'empty_label', 'Empty label excluded from positive training; not retained as a negative without visual review.'
        if record.get('touching_grains') or record.get('overlapping_grains') or record.get('dense_layout'):
            return 'geometry_filter', 'Touching, overlap, or dense-layout flag; excluded from the initial separated-grain set.'
        return 'class_mapping_review', 'Geometry passes, but multi-class labels need explicit mapping approval.'
    if dataset == '07_Raw_Rice_Seed':
        return 'geometry_filter', 'Dense layout in all images; unsuitable for the separated-grain initial set.'
    if dataset == '08_Rice_Grain':
        if record.get('touching_grains') or record.get('overlapping_grains') or record.get('dense_layout'):
            return 'geometry_filter', 'Touching, overlap, or dense-layout flag; excluded from the initial separated-grain set.'
        return 'class_mapping_review', "Only a small separated subset exists; classes 'Rice' and 'brokens' require explicit mapping review."
    if dataset == '01_Mendeley_Rice_Variety':
        return 'geometry_filter', 'High polygon counts, dense layout, touching, and overlap flags; poor match to separated grains.'
    return 'not_selected', 'Not selected for the first preparation experiment.'


def load_audit_records() -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        summary_path = AUDIT_ROOT / dataset / 'full_summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'Missing canonical audit summary: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('status') == 'BLOCKED' or not summary.get('records'):
            raise RuntimeError(f'Audit summary is not complete for {dataset}: {summary_path}')
        summaries[dataset] = summary
        class_names = summary.get('class_names_from_metadata')
        for record in summary['records']:
            category, reason = exclusion_details(dataset, record)
            include = dataset == SELECTED_DATASET and not category
            image_path = normalized_path(record['image_file'])
            label_path = normalized_path(record.get('label_file'))
            dimensions = record['image_dimensions']
            manifest_rows.append({
                'source_dataset': dataset,
                'original_image_path': image_path,
                'original_label_path': label_path,
                'image_width': dimensions['width'],
                'image_height': dimensions['height'],
                'polygon_count': record['valid_polygons'],
                'class_ids_present': class_ids_for_record(record, class_names),
                'class_names_present': sorted(record.get('class_instance_counts', {}).keys()),
                'touching_flag': bool(record.get('touching_grains')),
                'overlap_flag': bool(record.get('overlapping_grains')),
                'dense_flag': bool(record.get('dense_layout')),
                'empty_label_flag': bool(record.get('empty_label_file')),
                'missing_label_flag': bool(record.get('missing_label_file')),
                'invalid_annotation_flag': bool(record.get('invalid_annotation')),
                'separated_heuristic': bool(record.get('separated_under_heuristic')),
                'include_candidate': include,
                'exclusion_category': category,
                'exclusion_reason': reason,
                'source_split': record.get('split', ''),
                'prepared_split': '',
                'prepared_image_path': '',
                'prepared_label_path': '',
                'split_group': '',
            })
    return summaries, manifest_rows


def read_and_validate_yolo_polygon_file(
    label_path: Path,
    width: int,
    height: int,
    expected_class_id: int | None = None,
) -> Tuple[int, Counter[int]]:
    if not label_path.is_file():
        raise FileNotFoundError(f'Missing annotation: {label_path}')
    lines = label_path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
    if not any(line.strip() for line in lines):
        raise ValueError(f'Empty annotation is not allowed for positive data: {label_path}')

    instance_count = 0
    class_counts: Counter[int] = Counter()
    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().lstrip('\ufeff').split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(f'{label_path}:{line_number}: expected class plus >=3 coordinate pairs')
        try:
            raw_class = float(parts[0])
            if not raw_class.is_integer() or raw_class < 0:
                raise ValueError
            class_id = int(raw_class)
            coordinates = [float(value) for value in parts[1:]]
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'{label_path}:{line_number}: invalid class or coordinate') from exc
        if expected_class_id is not None and class_id != expected_class_id:
            raise ValueError(f'{label_path}:{line_number}: class {class_id}, expected {expected_class_id}')
        if any(not np.isfinite(value) or value < 0 or value > 1 for value in coordinates):
            raise ValueError(f'{label_path}:{line_number}: coordinates must be finite and normalized to [0,1]')

        pixel_points = [
            (coordinates[index] * width, coordinates[index + 1] * height)
            for index in range(0, len(coordinates), 2)
        ]
        area = abs(sum(
            x1 * pixel_points[(index + 1) % len(pixel_points)][1]
            - pixel_points[(index + 1) % len(pixel_points)][0] * y1
            for index, (x1, y1) in enumerate(pixel_points)
        )) / 2.0
        if area <= 0:
            raise ValueError(f'{label_path}:{line_number}: polygon area is zero')

        mask = Image.new('1', (width, height), 0)
        ImageDraw.Draw(mask).polygon(pixel_points, fill=1)
        if not np.asarray(mask, dtype=np.uint8).any():
            raise ValueError(f'{label_path}:{line_number}: polygon rasterizes to an empty mask')
        instance_count += 1
        class_counts[class_id] += 1
    if instance_count == 0:
        raise ValueError(f'No polygons found in positive annotation: {label_path}')
    return instance_count, class_counts


def source_family(image_path: Path) -> str:
    return image_path.stem.split('.rf', 1)[0]


def find_duplicate_groups(image_paths: List[Path]) -> Tuple[Dict[Path, str], Dict[str, Any]]:
    try:
        import imagehash
    except ImportError as exc:
        raise RuntimeError('imagehash is required to check near-duplicate leakage; do not split until available.') from exc

    parent = list(range(len(image_paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a = find(first)
        root_b = find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    family_indexes: Dict[str, List[int]] = defaultdict(list)
    hashes = []
    file_hashes: Dict[str, List[str]] = defaultdict(list)
    for index, path in enumerate(image_paths):
        family_indexes[source_family(path)].append(index)
        file_hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        with Image.open(path) as image:
            hashes.append(imagehash.phash(image.convert('RGB')))

    for indexes in family_indexes.values():
        for index in indexes[1:]:
            union(indexes[0], index)

    near_pairs: List[Dict[str, Any]] = []
    for first in range(len(image_paths)):
        for second in range(first + 1, len(image_paths)):
            distance = hashes[first] - hashes[second]
            if distance <= PHASH_DUPLICATE_DISTANCE:
                union(first, second)
                near_pairs.append({
                    'image_a': image_paths[first].name,
                    'image_b': image_paths[second].name,
                    'phash_distance': distance,
                })

    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(image_paths)):
        groups[find(index)].append(index)
    path_to_group: Dict[Path, str] = {}
    for group_number, indexes in enumerate(sorted(groups.values(), key=lambda values: min(values)), start=1):
        group_id = f'06_family_{group_number:04d}'
        for index in indexes:
            path_to_group[image_paths[index]] = group_id

    return path_to_group, {
        'method': 'group identical pre-.rf filename families; also group perceptual-hash pairs at Hamming distance <= 4',
        'phash_threshold': PHASH_DUPLICATE_DISTANCE,
        'group_count': len(groups),
        'exact_duplicate_groups': [names for names in file_hashes.values() if len(names) > 1],
        'near_duplicate_pairs': near_pairs,
    }


def make_splits(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    randomizer = random.Random(SPLIT_SEED)
    group_ids = sorted(groups)
    randomizer.shuffle(group_ids)
    group_sizes = [len(groups[group_id]) for group_id in group_ids]
    total = sum(group_sizes)
    target_train = round(total * SPLIT_RATIOS['train'])
    target_val = round(total * SPLIT_RATIOS['val'])

    if all(size == 1 for size in group_sizes):
        splits = ['train'] * target_train + ['val'] * target_val
        splits += ['test'] * (total - len(splits))
        return dict(zip(group_ids, splits))

    targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    current = {split: 0 for split in SPLIT_RATIOS}
    assignments: Dict[str, str] = {}
    ordered = sorted(group_ids, key=lambda group_id: (-len(groups[group_id]), group_ids.index(group_id)))
    for group_id in ordered:
        size = len(groups[group_id])
        selected = max(SPLIT_RATIOS, key=lambda split: (targets[split] - current[split], -current[split]))
        assignments[group_id] = selected
        current[selected] += size
    return assignments


def class_name_for_id(summary: Dict[str, Any], class_id: int) -> str:
    names = summary.get('class_names_from_metadata')
    if names and class_id < len(names):
        return names[class_id]
    return f'class_{class_id}'


def render_preview(
    image_path: Path,
    label_path: Path,
    target_path: Path,
    class_name: str,
) -> None:
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    width, height = original.size
    overlay = Image.new('RGBA', original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lines = label_path.read_text(encoding='utf-8-sig').splitlines()
    for instance_id, line in enumerate((line for line in lines if line.strip()), start=1):
        parts = line.split()
        coordinates = [float(value) for value in parts[1:]]
        points = [
            (coordinates[index] * width, coordinates[index + 1] * height)
            for index in range(0, len(coordinates), 2)
        ]
        color = ((instance_id * 61) % 256, (instance_id * 113) % 256, (instance_id * 173) % 256, 170)
        draw.polygon(points, fill=color, outline=(255, 255, 255, 230), width=2)
        center_x = sum(x for x, _ in points) / len(points)
        center_y = sum(y for _, y in points) / len(points)
        draw.text((center_x + 3, center_y + 3), f'{instance_id}', fill='white', stroke_width=1, stroke_fill='black')
    composite = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')

    pad = 18
    canvas = Image.new('RGB', (width * 2 + pad * 2, height + 42), (28, 31, 33))
    canvas.paste(original, (pad, 32))
    canvas.paste(composite, (width + pad * 2, 32))
    labels = ImageDraw.Draw(canvas)
    labels.text((pad, 8), f'Original | {width}x{height}', fill='white')
    labels.text((width + pad * 2, 8), f'{class_name} masks + instance IDs', fill='white')
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path)


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def build() -> Dict[str, Any]:
    if PREPARED_ROOT.exists():
        existing_files = [path for path in PREPARED_ROOT.rglob('*') if path.is_file()]
        if existing_files:
            raise FileExistsError(
                f'{PREPARED_ROOT} already contains files. This builder will not overwrite prepared data; review it and choose a new output path or explicitly archive it first.'
            )

    summaries, manifest_rows = load_audit_records()
    selected = [row for row in manifest_rows if row['include_candidate']]
    if not selected:
        raise RuntimeError('No images passed the explicit candidate policy.')

    selected_paths = [ROOT / row['original_image_path'] for row in selected]
    group_by_path, duplicate_audit = find_duplicate_groups(selected_paths)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_by_image_path = {ROOT / row['original_image_path']: row for row in selected}
    for image_path in selected_paths:
        groups[group_by_path[image_path]].append(rows_by_image_path[image_path])
    group_splits = make_splits(groups)

    source_class_names = summaries[SELECTED_DATASET]['class_names_from_metadata'] or []
    if source_class_names != [TARGET_CLASS_NAME]:
        raise RuntimeError(f'Expected one source class {TARGET_CLASS_NAME!r}, found {source_class_names!r}')

    for row, image_path in zip(selected, selected_paths):
        if row['class_ids_present'] not in ([], [0]):
            raise RuntimeError(f'Unexpected class IDs in selected source image: {row["original_image_path"]}')
        label_path = ROOT / row['original_label_path']
        image_width = row['image_width']
        image_height = row['image_height']
        polygon_count, classes = read_and_validate_yolo_polygon_file(
            label_path, image_width, image_height, expected_class_id=0
        )
        if polygon_count != row['polygon_count'] or any(class_id != 0 for class_id in classes):
            raise RuntimeError(f'Audit manifest and source label disagree: {row["original_image_path"]}')
        if row['touching_flag'] or row['overlap_flag'] or row['dense_flag'] or not row['separated_heuristic']:
            raise RuntimeError(f'Selected image fails the explicit geometry policy: {row["original_image_path"]}')
        row['split_group'] = group_by_path[image_path]
        row['prepared_split'] = group_splits[row['split_group']]
        filename = image_path.name
        row['prepared_image_path'] = f'images/{row["prepared_split"]}/{filename}'
        row['prepared_label_path'] = f'labels/{row["prepared_split"]}/{image_path.stem}.txt'

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_RATIOS:
        (PREPARED_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    preview_root = PREPARED_ROOT / 'preview'
    preview_root.mkdir(exist_ok=True)

    split_rows: List[Dict[str, Any]] = []
    split_instance_counts: Counter[str] = Counter()
    split_class_counts: Dict[str, Counter[str]] = {split: Counter() for split in SPLIT_RATIOS}
    split_source_counts: Dict[str, Counter[str]] = {split: Counter() for split in SPLIT_RATIOS}
    prepared_by_row: Dict[str, Tuple[Path, Path]] = {}

    for row in selected:
        source_image = ROOT / row['original_image_path']
        source_label = ROOT / row['original_label_path']
        prepared_image = PREPARED_ROOT / row['prepared_image_path']
        prepared_label = PREPARED_ROOT / row['prepared_label_path']
        if prepared_image.exists() or prepared_label.exists():
            raise FileExistsError(f'Prepared file collision for {source_image.name}')
        shutil.copy2(source_image, prepared_image)
        shutil.copy2(source_label, prepared_label)

        with Image.open(prepared_image) as image:
            width, height = image.size
        if (width, height) != (row['image_width'], row['image_height']):
            raise ValueError(f'Image dimensions changed during copy: {prepared_image}')
        polygon_count, class_counts = read_and_validate_yolo_polygon_file(
            prepared_label, width, height, expected_class_id=TARGET_CLASS_ID
        )
        if polygon_count != row['polygon_count']:
            raise ValueError(f'Copied label polygon count mismatch: {prepared_label}')

        prepared_by_row[row['original_image_path']] = (prepared_image, prepared_label)
        split = row['prepared_split']
        split_instance_counts[split] += polygon_count
        split_class_counts[split][TARGET_CLASS_NAME] += polygon_count
        split_source_counts[split][SELECTED_DATASET] += 1
        split_rows.append({
            'source_dataset': SELECTED_DATASET,
            'original_image_path': row['original_image_path'],
            'original_label_path': row['original_label_path'],
            'prepared_image_path': row['prepared_image_path'],
            'prepared_label_path': row['prepared_label_path'],
            'split': split,
            'split_group': row['split_group'],
            'image_width': width,
            'image_height': height,
            'polygon_count': polygon_count,
            'source_class_ids': [TARGET_CLASS_ID],
            'source_class_names': [TARGET_CLASS_NAME],
            'target_class_id': TARGET_CLASS_ID,
            'target_class_name': TARGET_CLASS_NAME,
        })

    selected_sorted = sorted(selected, key=lambda row: (row['polygon_count'], row['original_image_path']))
    low = selected_sorted[0]
    target_median = summaries[SELECTED_DATASET]['median_polygons_per_image']
    typical = next(
        (row for row in selected_sorted if row['polygon_count'] >= target_median and row['original_image_path'] != low['original_image_path']),
        selected_sorted[len(selected_sorted) // 2],
    )
    high = selected_sorted[-1]
    for category, row in [('low', low), ('typical', typical), ('high', high)]:
        image_path, label_path = prepared_by_row[row['original_image_path']]
        render_preview(
            image_path,
            label_path,
            preview_root / f'{category}_{image_path.stem}.png',
            TARGET_CLASS_NAME,
        )

    manifest_json = {
        'audit_source': 'data/audit/full_visual_segmentation/',
        'created_date': '2026-09-29',
        'target_ontology': [{'class_id': TARGET_CLASS_ID, 'class_name': TARGET_CLASS_NAME}],
        'included_image_count': len(selected),
        'excluded_image_count': len(manifest_rows) - len(selected),
        'candidate_policy': 'Include only 06_Rice_Grain_Segmentation images that are valid, nonempty, and separated under the audit heuristic. No other dataset or source class is mapped automatically.',
        'records': manifest_rows,
    }
    (PREPARED_ROOT / 'dataset_manifest.json').write_text(json.dumps(manifest_json, indent=2), encoding='utf-8')
    manifest_columns = [
        'source_dataset', 'original_image_path', 'original_label_path', 'image_width', 'image_height',
        'polygon_count', 'class_ids_present', 'class_names_present', 'touching_flag', 'overlap_flag',
        'dense_flag', 'empty_label_flag', 'missing_label_flag', 'invalid_annotation_flag',
        'separated_heuristic', 'include_candidate', 'exclusion_category', 'exclusion_reason',
        'source_split', 'prepared_split', 'prepared_image_path', 'prepared_label_path', 'split_group',
    ]
    write_csv(PREPARED_ROOT / 'dataset_manifest.csv', manifest_rows, manifest_columns)
    split_columns = [
        'source_dataset', 'original_image_path', 'original_label_path', 'prepared_image_path',
        'prepared_label_path', 'split', 'split_group', 'image_width', 'image_height', 'polygon_count',
        'source_class_ids', 'source_class_names', 'target_class_id', 'target_class_name',
    ]
    write_csv(PREPARED_ROOT / 'split_manifest.csv', split_rows, split_columns)

    (PREPARED_ROOT / 'data.yaml').write_text(
        "path: data/prepared/segmentation\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['rice_grain']\n",
        encoding='utf-8',
    )

    preparation_config = {
        'created_date': '2026-09-29',
        'target_class': {'id': TARGET_CLASS_ID, 'name': TARGET_CLASS_NAME},
        'included_source_datasets': [SELECTED_DATASET],
        'source_class_mapping': [{
            'source_dataset': SELECTED_DATASET,
            'source_class_id': 0,
            'source_class_name': 'rice_grain',
            'target_class_id': TARGET_CLASS_ID,
            'target_class_name': TARGET_CLASS_NAME,
            'mapping_type': 'identity; no annotation conversion',
        }],
        'excluded_mapping_decisions': {
            '04_RiceVigor': "Not included pending approval to map 'rice-seed' to 'rice_grain'.",
            '03_Rice_Variety': 'Not included; variety semantics are not mapped to a generic grain class.',
        },
        'split_seed': SPLIT_SEED,
        'split_ratios': SPLIT_RATIOS,
        'split_grouping': duplicate_audit,
        'empty_label_policy': 'Do not use empty labels as negative examples; exclude from positive data pending visual review.',
        'geometry_policy': 'Exclude any image with touching, overlap, or dense flag; require valid nonempty annotations.',
    }
    (PREPARED_ROOT / 'preparation_config.json').write_text(json.dumps(preparation_config, indent=2), encoding='utf-8')

    counts = Counter(row['prepared_split'] for row in selected)
    excluded_by_dataset = Counter(row['source_dataset'] for row in manifest_rows if not row['include_candidate'])
    exclusion_reasons = Counter(row['exclusion_category'] for row in manifest_rows if not row['include_candidate'])
    selected_summary = summaries[SELECTED_DATASET]
    all_validation_errors = 0
    yaml_content = (PREPARED_ROOT / 'data.yaml').read_text(encoding='utf-8')
    report_lines = [
        '# Prepared Segmentation Dataset Report',
        '',
        '**Status: prepared for a controlled pilot; training was not started.** Only the unambiguous 06 source was included. The audit heuristic does not establish that every visible physical grain is annotated.',
        '',
        '## Source and selection',
        '',
        '- Included source: `06_Rice_Grain_Segmentation` only; 104 images and 653 polygons.',
        '- Source class and target class are both `rice_grain` (ID 0); this is an identity mapping and no label conversion was performed.',
        '- 04 is held pending explicit approval to map `rice-seed` to `rice_grain`; its three empty-label images are not used as negatives.',
        '- 03 is excluded because its five labels encode rice varieties; no collapse to generic `rice_grain` is performed.',
        '- 01, 02, 05, 07, and 08 are excluded by the manifest rules for density/touching/overlap, empty labels, or unresolved class semantics.',
        f'- Manifest coverage: {len(manifest_rows)} image records across all eight audited datasets; {len(selected)} included and {len(manifest_rows)-len(selected)} excluded.',
        f'- Excluded image counts by source: `{dict(sorted(excluded_by_dataset.items()))}`.',
        f'- Exclusion categories: `{dict(sorted(exclusion_reasons.items()))}`.',
        '',
        '## Prepared dataset statistics',
        '',
        f'- Images: {len(selected)}; polygons: {selected_summary["total_valid_polygons"]}.',
        f'- Image dimensions: `{selected_summary["image_dimensions_distribution"]}`.',
        f'- Polygon count distribution: `{selected_summary["polygons_per_image_histogram"]}`.',
        f'- Bins: `{selected_summary["image_count_bins"]}`.',
        f'- Source distribution: `{{"{SELECTED_DATASET}": {len(selected)}}}`.',
        f'- Class distribution: `{{"{TARGET_CLASS_NAME}": {selected_summary["total_valid_polygons"]}}}`.',
        '',
        '## Train/validation/test split',
        '',
        '| Split | Images | Instances | Source distribution | Class distribution |',
        '|---|---:|---:|---|---|',
    ]
    for split in SPLIT_RATIOS:
        report_lines.append(
            f'| {split} | {counts[split]} | {split_instance_counts[split]} | '
            f'`{dict(split_source_counts[split])}` | `{dict(split_class_counts[split])}` |'
        )
    report_lines.extend([
        '',
        f'- Split seed: {SPLIT_SEED}; requested ratios: `{SPLIT_RATIOS}`.',
        f'- Leakage guard: grouped exact source-family stems and perceptual-hash pairs with distance <= {PHASH_DUPLICATE_DISTANCE}; {duplicate_audit["group_count"]} groups for {len(selected)} images, {len(duplicate_audit["exact_duplicate_groups"])} exact duplicate groups, {len(duplicate_audit["near_duplicate_pairs"])} near-duplicate pairs.',
        '- No source family or detected near-duplicate group is split across train/val/test.',
        '',
        '## Annotation validation',
        '',
        f'- Source audit: {selected_summary["total_valid_polygons"]} polygons; missing labels {selected_summary["missing_label_files"]}; empty labels {selected_summary["empty_label_files"]}; invalid images {selected_summary["invalid_annotation_images"]}; invalid lines {selected_summary["invalid_annotation_lines"]}.',
        f'- Copied positive annotations revalidated: {selected_summary["total_valid_polygons"]} valid polygon rows; all class IDs are 0; coordinates finite and normalized; polygons have nonzero area and nonempty raster masks.',
        f'- Touching / overlap / dense in selected data: {selected_summary["images_with_touching_masks"]} / {selected_summary["images_with_overlapping_masks"]} / {selected_summary["images_with_dense_layout"]}.',
        '- Empty-label policy: empty labels are not treated as negative examples; the three 04 empties and 1,387 05 empties are excluded pending review.',
        '- Dataset YAML: `data.yaml` defines one class, `rice_grain`, at ID 0. The selected source labels already use ID 0.',
        f'- Validation errors: {all_validation_errors}.',
        '',
        '## Preview and limitations',
        '',
        '- Preview images show the original at left and the polygon overlay with instance IDs, class name, and image dimensions at right.',
        '- Only 104 images are included. This is a small pilot set and is not established as representative of normal upload conditions.',
        '- Visual inspection of representative images cannot prove annotation completeness for all images. Manual audit remains appropriate before production claims.',
        '- Audit geometric flags are proxies; touching within two pixels and dense centroid distance are heuristic thresholds.',
        '- 04 mapping, 03 variety mapping, and any filtered subsets from 02/05/08 remain unresolved and excluded.',
        '',
        '## Proposed model and training configuration (not run)',
        '',
        '- Model: Ultralytics YOLOv8l-seg, because the current inference code tries that path first and expects `models/yolo_seg/run1_baseline/weights/best.pt`.',
        '- Annotation format: YOLO polygon segmentation; one class `rice_grain`, ID 0.',
        '- Input resolution: 640, matching the selected source images.',
        '- Initialization: pretrained `yolov8l-seg.pt`; checkpoint is not present in the repository and would need to be obtained when approved.',
        '- Suggested pilot: 100 epochs maximum, batch 8, AdamW, learning rate 0.001, patience 20, seed 42, deterministic mode; mild flips/scale/translation, no mixup or mosaic for the initial separated-sheet pilot.',
        '- Metrics to report: box and mask precision/recall, mask mAP50 and mAP50-95, plus count MAE and per-image count error on the held-out test split.',
        '- Dependency blocker: `ultralytics` is not installed or declared in `backend/requirements.txt`; no weights/checkpoint exist. Torch/torchvision are installed. No dependencies were installed for this preparation task.',
        '- Backend/API compatibility caveat: inference code prioritizes YOLOv8l-seg, but the registry currently marks segmentation as a Mask R-CNN placeholder and the existing class map uses background 0 / rice_grain 1. The YOLO path recognizes the model class name `rice_grain` at class ID 0, but registry/path consistency should be verified before training or deployment. No backend changes were made.',
        '',
        'Exact command after dependency/checkpoint approval (not executed):',
        '',
        '```powershell',
        'yolo segment train model=yolov8l-seg.pt data=data/prepared/segmentation/data.yaml imgsz=640 epochs=100 batch=8 optimizer=AdamW lr0=0.001 patience=20 seed=42 deterministic=True mosaic=0.0 mixup=0.0 fliplr=0.5 flipud=0.0 degrees=10 translate=0.05 scale=0.1 project=models/yolo_seg name=run1_baseline',
        '```',
        '',
        '## Prepared artifacts',
        '',
        '- Dataset: `data/prepared/segmentation/`.',
        '- Machine-readable manifest: `dataset_manifest.csv` and `dataset_manifest.json`.',
        '- Split manifest: `split_manifest.csv`.',
        '- Rules/mapping/split config: `preparation_config.json`.',
        '- Training data report: `TRAINING_DATA_REPORT.md`.',
        '- YOLO dataset config: `data.yaml`.',
        '- Previews: `preview/`.',
        '',
        'No training run was started. Approve the `rice-seed` mapping if 04 should be added, resolve the YOLO dependency/registry checkpoint path, and review the previews before using the command above.',
        '',
    ])
    (PREPARED_ROOT / 'TRAINING_DATA_REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')
    return {
        'included_images': len(selected),
        'excluded_images': len(manifest_rows) - len(selected),
        'split_counts': dict(counts),
        'split_instances': dict(split_instance_counts),
        'phash_audit': duplicate_audit,
    }


def main() -> None:
    result = build()
    print('Prepared dataset at:', PREPARED_ROOT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()