from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import convert_xyxy_to_yolo
from src.composition import infer_composition_type
from src.utils import classes_to_field, ensure_dir, materialize_file, read_image_size, unique_stem, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a COCO-format X-ray dataset, such as PIDray, to YOLO.")
    parser.add_argument("--images-root", required=True, help="Root directory containing images referenced by COCO JSON.")
    parser.add_argument("--annotation-json", required=True, help="COCO annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="YOLO dataset output directory.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--split-source", default="all", help="Source split label to store in metadata, e.g. train/easy/hard/hidden.")
    parser.add_argument(
        "--composition-type",
        default="auto",
        help="Override composition type, or use auto to infer from path/split-source.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing metadata/classes when converting multiple COCO JSONs into one output_dir.",
    )
    return parser.parse_args()


def find_image(images_root: Path, file_name: str) -> Path | None:
    candidate = images_root / file_name
    if candidate.exists():
        return candidate
    candidate = images_root / Path(file_name).name
    if candidate.exists():
        return candidate
    matches = list(images_root.rglob(Path(file_name).name))
    return sorted(matches)[0] if matches else None


def load_existing_classes(output_dir: Path) -> list[str]:
    classes_txt = output_dir / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def main() -> None:
    args = parse_args()
    import pandas as pd
    from tqdm import tqdm

    images_root = Path(args.images_root)
    annotation_json = Path(args.annotation_json)
    output_dir = ensure_dir(args.output_dir)
    images_all = ensure_dir(output_dir / "images" / "all")
    labels_all = ensure_dir(output_dir / "labels" / "all")

    coco = json.loads(annotation_json.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in coco.get("images", [])}
    categories = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in coco.get("categories", [])}
    if not images or not categories:
        raise RuntimeError("COCO JSON must contain non-empty images and categories arrays.")

    existing_classes = load_existing_classes(output_dir) if args.append else []
    new_classes = sorted(set(categories.values()) - set(existing_classes))
    discovered_classes = existing_classes + new_classes
    class_to_id = {name: idx for idx, name in enumerate(discovered_classes)}

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id"))
        if ann.get("iscrowd", 0):
            continue
        annotations_by_image[image_id].append(ann)

    rows: list[dict[str, object]] = []
    if args.append and (output_dir / "metadata.csv").exists():
        rows.extend(pd.read_csv(output_dir / "metadata.csv").to_dict("records"))

    class_counts: Counter[str] = Counter()
    missing_images = 0
    skipped_boxes = 0

    for image_id, image_info in tqdm(sorted(images.items()), desc="Converting COCO"):
        file_name = str(image_info.get("file_name", ""))
        image_path = find_image(images_root, file_name)
        if image_path is None:
            missing_images += 1
            continue

        width = int(image_info.get("width") or 0)
        height = int(image_info.get("height") or 0)
        if width <= 0 or height <= 0:
            try:
                width, height = read_image_size(image_path)
            except Exception:
                skipped_boxes += len(annotations_by_image.get(image_id, []))
                continue

        stem = unique_stem(image_path, images_root)
        image_dst = images_all / f"{stem}{image_path.suffix.lower()}"
        label_dst = labels_all / f"{stem}.txt"
        materialize_file(image_path, image_dst, args.copy_mode)

        label_lines: list[str] = []
        kept_classes: list[str] = []
        for ann in annotations_by_image.get(image_id, []):
            category_name = categories.get(int(ann.get("category_id", -1)))
            bbox = ann.get("bbox")
            if category_name is None or not isinstance(bbox, list) or len(bbox) < 4:
                skipped_boxes += 1
                continue
            x, y, box_w, box_h = [float(value) for value in bbox[:4]]
            yolo_bbox = convert_xyxy_to_yolo([x, y, x + box_w, y + box_h], width, height)
            if yolo_bbox is None:
                skipped_boxes += 1
                continue
            label_lines.append(
                f"{class_to_id[category_name]} " + " ".join(f"{value:.6f}" for value in yolo_bbox)
            )
            kept_classes.append(category_name)
            class_counts[category_name] += 1
        label_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        composition_type = (
            args.composition_type
            if args.composition_type != "auto"
            else infer_composition_type(
                str(image_path),
                {"annotation_json": str(annotation_json), "split_source": args.split_source, "file_name": file_name},
            )
        )
        rows.append(
            {
                "image_path": image_dst.as_posix(),
                "label_path": label_dst.as_posix(),
                "original_image_path": image_path.as_posix(),
                "original_annotation_path": annotation_json.as_posix(),
                "width": width,
                "height": height,
                "composition_type": composition_type,
                "num_objects": len(label_lines),
                "classes": classes_to_field(kept_classes),
                "split_source": args.split_source,
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
    (output_dir / "classes.txt").write_text("\n".join(discovered_classes) + "\n", encoding="utf-8")
    write_json(
        output_dir / "coco_conversion_report.json",
        {
            "images_root": images_root.as_posix(),
            "annotation_json": annotation_json.as_posix(),
            "split_source": args.split_source,
            "num_coco_images": len(images),
            "num_rows_total": len(rows),
            "num_classes": len(discovered_classes),
            "classes": discovered_classes,
            "class_counts_this_run": dict(sorted(class_counts.items())),
            "missing_images": missing_images,
            "skipped_boxes": skipped_boxes,
            "composition_type_mode": args.composition_type,
        },
    )
    print(f"COCO conversion complete. Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
