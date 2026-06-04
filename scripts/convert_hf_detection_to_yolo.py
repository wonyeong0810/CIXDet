from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.composition import infer_composition_type
from src.utils import classes_to_field, ensure_dir, stable_hash, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face object-detection dataset to YOLO.")
    parser.add_argument("--repo-id", default="Voxel51/PIDray", help="Hugging Face dataset repo id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-dir", help="Hugging Face cache directory, preferably on persistent storage.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--composition-strategy",
        choices=["path", "object_count_proxy"],
        default="path",
        help="How to fill composition_type when the HF dataset has no AIHub/PIDray split labels.",
    )
    return parser.parse_args()


def find_image_value(sample: dict[str, Any]) -> Any:
    for key in ("image", "img", "filepath", "file_path", "image_path", "path"):
        if key in sample:
            return sample[key]
    return None


def detection_from_dict(value: dict[str, Any]) -> dict[str, Any] | None:
    bbox = value.get("bounding_box", value.get("bbox", value.get("box")))
    label = value.get("label", value.get("class_name", value.get("class", value.get("category"))))
    if bbox is None or label is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    return {"label": str(label), "bbox": [float(v) for v in bbox[:4]]}


def extract_detections(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        direct = detection_from_dict(value)
        if direct is not None:
            return [direct]
        if isinstance(value.get("detections"), list):
            detections: list[dict[str, Any]] = []
            for item in value["detections"]:
                detections.extend(extract_detections(item))
            return detections
        detections = []
        for item in value.values():
            detections.extend(extract_detections(item))
        return detections
    if isinstance(value, list):
        detections = []
        for item in value:
            detections.extend(extract_detections(item))
        return detections
    return []


def bbox_to_yolo(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = bbox[:4]
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        box_w = w
        box_h = h
    else:
        x_center = (x + w / 2.0) / float(width)
        y_center = (y + h / 2.0) / float(height)
        box_w = w / float(width)
        box_h = h / float(height)

    if box_w <= 0 or box_h <= 0:
        return None
    values = [
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(box_w, 0.0), 1.0),
        min(max(box_h, 0.0), 1.0),
    ]
    return values if values[2] > 0 and values[3] > 0 else None


def composition_from_object_count(num_objects: int) -> str:
    if num_objects <= 1:
        return "single_basic"
    if num_objects <= 3:
        return "complex_target"
    return "complex_with_non_target"


def image_identity(sample: dict[str, Any], idx: int) -> str:
    for key in ("filepath", "file_path", "image_path", "path"):
        if sample.get(key):
            return str(sample[key])
    return f"hf_sample_{idx:06d}"


def main() -> None:
    args = parse_args()
    import pandas as pd
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm

    if args.hf_cache_dir:
        ensure_dir(args.hf_cache_dir)

    dataset = load_dataset(
        args.repo_id,
        split=args.split,
        cache_dir=args.hf_cache_dir,
    )
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    samples = list(dataset)
    class_names = sorted({det["label"] for sample in samples for det in extract_detections(sample)})
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    if not class_names:
        raise RuntimeError(
            "No object-detection labels were found in the Hugging Face dataset. "
            "This converter expects FiftyOne-style detections with label and bounding_box fields."
        )

    output_dir = ensure_dir(args.output_dir)
    images_dir = ensure_dir(output_dir / "images" / "all")
    labels_dir = ensure_dir(output_dir / "labels" / "all")
    rows: list[dict[str, Any]] = []
    skipped_no_image = 0
    skipped_boxes = 0

    for idx, sample in enumerate(tqdm(samples, desc="Converting HF dataset")):
        image_value = find_image_value(sample)
        if image_value is None:
            skipped_no_image += 1
            continue

        if isinstance(image_value, Image.Image):
            image = image_value.convert("RGB")
            original_path = image_identity(sample, idx)
        elif isinstance(image_value, dict) and "path" in image_value:
            image = Image.open(image_value["path"]).convert("RGB")
            original_path = str(image_value["path"])
        elif isinstance(image_value, (str, Path)) and Path(image_value).exists():
            image = Image.open(image_value).convert("RGB")
            original_path = str(image_value)
        else:
            skipped_no_image += 1
            continue

        width, height = image.size
        stem = f"hf_{idx:06d}_{stable_hash(original_path)}"
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        image.save(image_path, quality=95)

        detections = extract_detections(sample)
        label_lines = []
        kept_classes = []
        for detection in detections:
            yolo_bbox = bbox_to_yolo(detection["bbox"], width, height)
            if yolo_bbox is None:
                skipped_boxes += 1
                continue
            class_name = detection["label"]
            label_lines.append(
                f"{class_to_id[class_name]} " + " ".join(f"{value:.6f}" for value in yolo_bbox)
            )
            kept_classes.append(class_name)
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        path_composition = infer_composition_type(original_path, sample)
        composition_type = path_composition
        if composition_type == "unknown" and args.composition_strategy == "object_count_proxy":
            composition_type = composition_from_object_count(len(label_lines))

        rows.append(
            {
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "original_image_path": original_path,
                "original_annotation_path": f"hf://datasets/{args.repo_id}",
                "width": width,
                "height": height,
                "composition_type": composition_type,
                "num_objects": len(label_lines),
                "classes": classes_to_field(kept_classes),
                "split_source": f"hf_{args.split}",
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    write_json(
        output_dir / "hf_conversion_report.json",
        {
            "repo_id": args.repo_id,
            "split": args.split,
            "num_samples_loaded": len(samples),
            "num_rows_written": len(rows),
            "num_classes": len(class_names),
            "classes": class_names,
            "skipped_no_image": skipped_no_image,
            "skipped_boxes": skipped_boxes,
            "composition_strategy": args.composition_strategy,
            "note": "object_count_proxy is a research convenience when no composition labels are provided by the HF dataset.",
        },
    )
    print(f"HF conversion complete. Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()

