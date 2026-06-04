from __future__ import annotations

import argparse
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_names_from_data_yaml, materialize_file, write_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO and append CI-XDet metrics CSV rows.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--metadata", help="Optional metadata_test.csv for composition-specific evaluation.")
    parser.add_argument("--output-csv", default="outputs/eval_results.csv")
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    return parser.parse_args()


def experiment_name_from_weights(weights: Path) -> str:
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def metric_value(metrics, attr: str) -> float | None:
    box = getattr(metrics, "box", None)
    value = getattr(box, attr, None) if box is not None else getattr(metrics, attr, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_val(model, data_yaml: Path, args: argparse.Namespace):
    kwargs = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
    }
    if args.conf is not None:
        kwargs["conf"] = args.conf
    if args.iou is not None:
        kwargs["iou"] = args.iou
    return model.val(**kwargs)


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    ensure_dir(path.parent)
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def metrics_row(metrics, experiment_name: str, split_name: str, eval_subset: str, composition_type: str, num_images: int):
    return {
        "experiment_name": experiment_name,
        "split_name": split_name,
        "eval_subset": eval_subset,
        "composition_type": composition_type,
        "num_images": num_images,
        "mAP50": metric_value(metrics, "map50"),
        "mAP50_95": metric_value(metrics, "map"),
        "precision": metric_value(metrics, "mp"),
        "recall": metric_value(metrics, "mr"),
        "fitness": float(metrics.fitness) if hasattr(metrics, "fitness") else None,
    }


def per_class_rows(metrics, names: list[str], base_row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    box = getattr(metrics, "box", None)
    maps = getattr(box, "maps", None) if box is not None else None
    ap50 = getattr(box, "ap50", None) if box is not None else None
    if maps is None:
        return rows
    for class_id, map_value in enumerate(list(maps)):
        row = {
            key: base_row[key]
            for key in ["experiment_name", "split_name", "eval_subset", "composition_type", "num_images"]
        }
        row.update(
            {
                "class_id": class_id,
                "class_name": names[class_id] if class_id < len(names) else str(class_id),
                "mAP50_95": float(map_value),
                "mAP50": float(ap50[class_id]) if ap50 is not None and class_id < len(ap50) else None,
            }
        )
        rows.append(row)
    return rows


def create_filtered_eval_yaml(df, composition: str, names: list[str], base_dir: Path) -> Path:
    split_dir = ensure_dir(base_dir / composition)
    for subset in ("train", "val", "test"):
        ensure_dir(split_dir / "images" / subset)
        ensure_dir(split_dir / "labels" / subset)

    for _, row in df.reset_index(drop=True).iterrows():
        image_src = Path(row["image_path"])
        label_src = Path(row["label_path"])
        for subset in ("train", "val", "test"):
            materialize_file(image_src, split_dir / "images" / subset / image_src.name, "symlink")
            if label_src.exists():
                materialize_file(label_src, split_dir / "labels" / subset / label_src.name, "symlink")
            else:
                (split_dir / "labels" / subset / label_src.name).write_text("", encoding="utf-8")
    data_yaml = split_dir / "data.yaml"
    write_data_yaml(data_yaml, split_dir, names)
    return data_yaml


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO
    import pandas as pd

    weights = Path(args.weights)
    model = YOLO(str(weights))
    experiment_name = experiment_name_from_weights(weights)
    names = load_names_from_data_yaml(args.data_yaml)

    rows: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []

    metrics = run_val(model, Path(args.data_yaml), args)
    num_images = 0
    if args.metadata:
        num_images = len(pd.read_csv(args.metadata))
    base = metrics_row(metrics, experiment_name, args.split_name, "test", "all", num_images)
    rows.append(base)
    per_class.extend(per_class_rows(metrics, names, base))

    if args.metadata:
        metadata_df = pd.read_csv(args.metadata)
        if "composition_type" not in metadata_df.columns:
            print("Warning: metadata has no composition_type column; skipping composition-specific evaluation.")
        else:
            with tempfile.TemporaryDirectory(prefix="ci_xdet_eval_") as tmp:
                tmp_dir = Path(tmp)
                for composition, subset_df in metadata_df.groupby("composition_type"):
                    if len(subset_df) == 0:
                        continue
                    filtered_yaml = create_filtered_eval_yaml(subset_df, str(composition), names, tmp_dir)
                    subset_metrics = run_val(model, filtered_yaml, args)
                    subset_row = metrics_row(
                        subset_metrics,
                        experiment_name,
                        args.split_name,
                        f"composition_{composition}",
                        str(composition),
                        len(subset_df),
                    )
                    rows.append(subset_row)
                    per_class.extend(per_class_rows(subset_metrics, names, subset_row))

    output_csv = Path(args.output_csv)
    append_rows(output_csv, rows)
    if per_class:
        append_rows(output_csv.parent / "per_class_eval_results.csv", per_class)
    print(f"Evaluation rows appended to {output_csv}")


if __name__ == "__main__":
    main()

