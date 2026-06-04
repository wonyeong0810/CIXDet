from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import clear_annotation_warnings, get_annotation_warnings, parse_annotation, validate_bbox
from src.composition import infer_composition_type, known_compositions
from src.utils import (
    ANNOTATION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    ensure_dir,
    read_image_size,
    scan_files,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local AIHub X-ray dataset without modifying it.")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for dataset_stats outputs.")
    return parser.parse_args()


def match_annotations_to_images(annotation_paths: list[Path], image_paths: list[Path]) -> dict[Path, Path]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        by_stem[image_path.stem].append(image_path)

    matches: dict[Path, Path] = {}
    for annotation_path in annotation_paths:
        candidates = by_stem.get(annotation_path.stem, [])
        if not candidates:
            continue
        same_parent = [path for path in candidates if path.parent == annotation_path.parent]
        matches[annotation_path] = sorted(same_parent or candidates)[0]
    return matches


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    from tqdm import tqdm
    import pandas as pd

    dataset_root = Path(args.dataset_root)
    output_dir = ensure_dir(args.output_dir)
    clear_annotation_warnings()

    print(f"Scanning images and annotations under: {dataset_root}")
    image_paths = scan_files(dataset_root, IMAGE_EXTENSIONS)
    annotation_paths = scan_files(dataset_root, ANNOTATION_EXTENSIONS)
    matches = match_annotations_to_images(annotation_paths, image_paths)

    image_size_counts: Counter[str] = Counter()
    image_read_errors: list[str] = []
    image_sizes: dict[Path, tuple[int, int]] = {}

    for image_path in tqdm(image_paths, desc="Reading image sizes"):
        try:
            size = read_image_size(image_path)
            image_sizes[image_path] = size
            image_size_counts[f"{size[0]}x{size[1]}"] += 1
        except Exception as exc:
            image_read_errors.append(f"{image_path}: {exc}")

    class_counts: Counter[str] = Counter()
    object_counts: list[int] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_area_ratios: list[float] = []
    composition_counts: Counter[str] = Counter()
    composition_examples: dict[str, list[str]] = {name: [] for name in known_compositions() + ["unknown"]}

    for annotation_path in tqdm(annotation_paths, desc="Parsing annotations"):
        image_path = matches.get(annotation_path)
        metadata = {"image_path": str(image_path)} if image_path else None
        composition = infer_composition_type(str(annotation_path), metadata)
        composition_counts[composition] += 1
        if len(composition_examples.setdefault(composition, [])) < 5:
            composition_examples[composition].append(str(annotation_path))

        objects = parse_annotation(annotation_path)
        object_counts.append(len(objects))
        for obj in objects:
            class_name = str(obj["class_name"])
            class_counts[class_name] += 1
            bbox = obj["bbox"]
            if not validate_bbox(bbox):
                continue
            xmin, ymin, xmax, ymax = [float(value) for value in bbox]
            bbox_widths.append(xmax - xmin)
            bbox_heights.append(ymax - ymin)
            if image_path in image_sizes:
                width, height = image_sizes[image_path]
                if width > 0 and height > 0:
                    bbox_area_ratios.append(((xmax - xmin) * (ymax - ymin)) / float(width * height))

    warnings = get_annotation_warnings()
    warning_counts = Counter(item["message"] for item in warnings)

    stats = {
        "num_images": len(image_paths),
        "num_annotation_files": len(annotation_paths),
        "matched_image_annotation_pairs": len(matches),
        "detected_class_names": sorted(class_counts),
        "class_frequency": dict(sorted(class_counts.items())),
        "image_size_distribution": dict(image_size_counts.most_common()),
        "bbox_size_distribution": {
            "width_min": min(bbox_widths) if bbox_widths else None,
            "width_p50": percentile(bbox_widths, 50),
            "width_p90": percentile(bbox_widths, 90),
            "height_min": min(bbox_heights) if bbox_heights else None,
            "height_p50": percentile(bbox_heights, 50),
            "height_p90": percentile(bbox_heights, 90),
            "area_ratio_p50": percentile(bbox_area_ratios, 50),
            "area_ratio_p90": percentile(bbox_area_ratios, 90),
        },
        "objects_per_image_or_annotation": {
            "min": min(object_counts) if object_counts else None,
            "p50": percentile([float(v) for v in object_counts], 50),
            "p90": percentile([float(v) for v in object_counts], 90),
            "max": max(object_counts) if object_counts else None,
        },
        "composition_type_counts": dict(composition_counts),
        "unknown_composition_count": composition_counts.get("unknown", 0),
        "composition_examples": composition_examples,
        "image_read_errors": image_read_errors[:50],
        "parsing_errors_summary": dict(warning_counts.most_common()),
        "num_parsing_warnings": len(warnings),
        "parsing_warning_examples": warnings[:20],
    }

    write_json(output_dir / "dataset_stats.json", stats)

    rows: list[dict[str, object]] = []
    rows.extend(
        [
            {"section": "summary", "key": "num_images", "value": len(image_paths)},
            {"section": "summary", "key": "num_annotation_files", "value": len(annotation_paths)},
            {"section": "summary", "key": "matched_image_annotation_pairs", "value": len(matches)},
            {"section": "summary", "key": "unknown_composition_count", "value": composition_counts.get("unknown", 0)},
            {"section": "summary", "key": "num_parsing_warnings", "value": len(warnings)},
        ]
    )
    rows.extend({"section": "class_frequency", "key": key, "value": value} for key, value in sorted(class_counts.items()))
    rows.extend({"section": "composition_type_counts", "key": key, "value": value} for key, value in sorted(composition_counts.items()))
    rows.extend({"section": "image_size_distribution", "key": key, "value": value} for key, value in image_size_counts.most_common())
    pd.DataFrame(rows).to_csv(output_dir / "dataset_stats.csv", index=False)

    print("\nDataset inspection complete")
    print(f"Images: {len(image_paths)}")
    print(f"Annotations: {len(annotation_paths)}")
    print(f"Matched pairs: {len(matches)}")
    print(f"Classes: {len(class_counts)}")
    print(f"Composition counts: {dict(composition_counts)}")
    if warnings:
        print(f"Warnings: {len(warnings)}. See {output_dir / 'dataset_stats.json'} for examples.")


if __name__ == "__main__":
    main()

