from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, yolo_label_to_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract object crops from YOLO labels for feature analysis.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="object_crops")
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--max-crops-per-class", type=int)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def load_class_names(metadata_path: Path) -> list[str]:
    classes_txt = metadata_path.parent / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def main() -> None:
    args = parse_args()
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)
    output_dir = ensure_dir(args.output_dir)
    class_names = load_class_names(metadata_path)
    class_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting crops"):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        composition = str(row.get("composition_type", "unknown"))
        if not image_path.exists() or not label_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        for obj_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            if args.max_crops_per_class is not None and class_counts[class_name] >= args.max_crops_per_class:
                continue
            xmin, ymin, xmax, ymax = yolo_label_to_xyxy([float(v) for v in parts[1:]], width, height)
            xmin = max(0, int(xmin) - args.padding)
            ymin = max(0, int(ymin) - args.padding)
            xmax = min(width, int(xmax) + args.padding)
            ymax = min(height, int(ymax) + args.padding)
            if (xmax - xmin) < args.min_box_size or (ymax - ymin) < args.min_box_size:
                continue
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop_dir = ensure_dir(output_dir / safe_name(class_name) / safe_name(composition))
            crop_path = crop_dir / f"{image_path.stem}_{obj_idx:03d}.png"
            crop.save(crop_path)
            class_counts[class_name] += 1
            rows.append(
                {
                    "crop_path": crop_path.as_posix(),
                    "class_name": class_name,
                    "class_id": class_id,
                    "composition_type": composition,
                    "source_image": image_path.as_posix(),
                    "bbox_xyxy": json.dumps([xmin, ymin, xmax, ymax]),
                    "width": xmax - xmin,
                    "height": ymax - ymin,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "crops_metadata.csv", index=False)
    print(f"Saved {len(rows)} crops to {output_dir}")


if __name__ == "__main__":
    main()

