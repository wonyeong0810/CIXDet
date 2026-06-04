from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import convert_xyxy_to_yolo, get_annotation_warnings, parse_annotation
from src.composition import infer_composition_type
from src.utils import (
    ANNOTATION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    classes_to_field,
    ensure_dir,
    load_class_map,
    materialize_file,
    parse_extensions,
    read_image_size,
    save_class_map,
    scan_files,
    unique_stem,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local AIHub-style X-ray annotations to YOLO format.")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root.")
    parser.add_argument("--output-dir", required=True, help="YOLO dataset output directory.")
    parser.add_argument("--class-map", default="configs/class_map.yaml", help="YAML class map path.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--allow-unknown-classes", action="store_true", help="Skip unknown classes instead of failing.")
    parser.add_argument("--image-extensions", default=",".join(IMAGE_EXTENSIONS))
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


def main() -> None:
    args = parse_args()
    from tqdm import tqdm
    import pandas as pd

    dataset_root = Path(args.dataset_root)
    output_dir = ensure_dir(args.output_dir)
    images_all = ensure_dir(output_dir / "images" / "all")
    labels_all = ensure_dir(output_dir / "labels" / "all")
    image_extensions = parse_extensions(args.image_extensions, IMAGE_EXTENSIONS)

    image_paths = scan_files(dataset_root, image_extensions)
    annotation_paths = scan_files(dataset_root, ANNOTATION_EXTENSIONS)
    matches = match_annotations_to_images(annotation_paths, image_paths)
    if not matches:
        raise RuntimeError("No image/annotation pairs were matched by filename stem. Check the dataset root or naming schema.")

    parsed_by_annotation: dict[Path, list[dict]] = {}
    discovered_classes: set[str] = set()
    for annotation_path in tqdm(annotation_paths, desc="Discovering classes"):
        objects = parse_annotation(annotation_path)
        parsed_by_annotation[annotation_path] = objects
        discovered_classes.update(str(obj["class_name"]) for obj in objects)

    class_map_path = Path(args.class_map)
    if class_map_path.exists():
        class_to_id, class_names = load_class_map(class_map_path)
        if class_names:
            print(f"Using class map: {class_map_path}")
        else:
            print(f"Class map {class_map_path} is empty; auto-discovering classes.")
            class_names = sorted(discovered_classes)
            class_to_id = {name: idx for idx, name in enumerate(class_names)}
            save_class_map(PROJECT_ROOT / "configs" / "class_map.yaml", class_names)
    else:
        class_names = sorted(discovered_classes)
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        save_class_map(PROJECT_ROOT / "configs" / "class_map.yaml", class_names)
        print(f"Auto-generated class map: {PROJECT_ROOT / 'configs' / 'class_map.yaml'}")

    if not class_names:
        raise RuntimeError("No classes were discovered. Inspect annotation parser warnings in conversion_report.json.")

    rows: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    skipped_unknown_classes: Counter[str] = Counter()
    skipped_invalid_boxes = 0
    converted_pairs = 0
    actual_copy_modes: Counter[str] = Counter()

    for annotation_path, image_path in tqdm(sorted(matches.items()), desc="Converting"):
        objects = parsed_by_annotation.get(annotation_path) or parse_annotation(annotation_path)
        try:
            width, height = read_image_size(image_path)
        except Exception as exc:
            print(f"Warning: failed to read image size for {image_path}: {exc}")
            continue

        stem = unique_stem(image_path, dataset_root)
        image_dst = images_all / f"{stem}{image_path.suffix.lower()}"
        label_dst = labels_all / f"{stem}.txt"
        label_lines: list[str] = []
        kept_classes: list[str] = []

        for obj in objects:
            class_name = str(obj["class_name"])
            if class_name not in class_to_id:
                skipped_unknown_classes[class_name] += 1
                if args.allow_unknown_classes:
                    continue
                continue
            yolo_bbox = convert_xyxy_to_yolo(obj["bbox"], width, height)
            if yolo_bbox is None:
                skipped_invalid_boxes += 1
                continue
            class_id = class_to_id[class_name]
            label_lines.append(
                f"{class_id} " + " ".join(f"{value:.6f}" for value in yolo_bbox)
            )
            kept_classes.append(class_name)
            class_counts[class_name] += 1

        actual_mode = materialize_file(image_path, image_dst, args.copy_mode)
        actual_copy_modes[actual_mode] += 1
        label_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        rows.append(
            {
                "image_path": image_dst.as_posix(),
                "label_path": label_dst.as_posix(),
                "original_image_path": image_path.as_posix(),
                "original_annotation_path": annotation_path.as_posix(),
                "width": width,
                "height": height,
                "composition_type": infer_composition_type(str(image_path), {"annotation_path": str(annotation_path)}),
                "num_objects": len(label_lines),
                "classes": classes_to_field(kept_classes),
                "split_source": "all",
            }
        )
        converted_pairs += 1

    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    report = {
        "dataset_root": dataset_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_images_found": len(image_paths),
        "num_annotations_found": len(annotation_paths),
        "matched_pairs": len(matches),
        "converted_pairs": converted_pairs,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": dict(sorted(class_counts.items())),
        "skipped_unknown_classes": dict(skipped_unknown_classes),
        "skipped_invalid_boxes": skipped_invalid_boxes,
        "copy_modes_used": dict(actual_copy_modes),
        "parse_warnings": get_annotation_warnings()[:100],
        "num_parse_warnings": len(get_annotation_warnings()),
    }
    write_json(output_dir / "conversion_report.json", report)

    if skipped_unknown_classes and not args.allow_unknown_classes:
        print("Warning: unknown classes were skipped. Pass --allow-unknown-classes to keep conversion non-fatal by design.")
    print(f"Conversion complete: {converted_pairs} pairs")
    print(f"Metadata: {metadata_path}")
    print(f"Classes: {output_dir / 'classes.txt'}")


if __name__ == "__main__":
    main()

