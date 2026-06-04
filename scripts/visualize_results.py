from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, parse_classes_field, yolo_label_to_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CI-XDet dataset and result visualizations.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--eval-csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser.parse_args()


def save_bar(labels, values, title: str, ylabel: str, path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_class_names(metadata_path: Path) -> list[str]:
    classes_txt = metadata_path.parent / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def collect_bbox_stats(df):
    areas = []
    aspects = []
    objects_per_image = []
    for _, row in df.iterrows():
        label_path = Path(row["label_path"])
        width = int(row.get("width", 0) or 0)
        height = int(row.get("height", 0) or 0)
        count = 0
        if not label_path.exists() or width <= 0 or height <= 0:
            objects_per_image.append(0)
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = parts
            bw = float(bw)
            bh = float(bh)
            if bw <= 0 or bh <= 0:
                continue
            areas.append(bw * bh)
            aspects.append((bw * width) / max(1e-9, bh * height))
            count += 1
        objects_per_image.append(count)
    return areas, aspects, objects_per_image


def save_hist(values, title: str, xlabel: str, path: Path, plt, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#59a14f", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_samples(df, output_dir: Path, class_names: list[str], plt) -> None:
    from PIL import Image
    import matplotlib.patches as patches

    for composition, group in df.groupby("composition_type"):
        sample = group.head(4)
        if sample.empty:
            continue
        fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
        if len(sample) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sample.iterrows()):
            image_path = Path(row["image_path"])
            label_path = Path(row["label_path"])
            ax.axis("off")
            ax.set_title(str(composition))
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
            width, height = image.size
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id = int(float(parts[0]))
                    bbox = yolo_label_to_xyxy([float(v) for v in parts[1:]], width, height)
                    xmin, ymin, xmax, ymax = bbox
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5, edgecolor="#e15759", facecolor="none")
                    ax.add_patch(rect)
                    label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                    ax.text(xmin, max(0, ymin - 2), label, color="white", fontsize=7, bbox={"facecolor": "#e15759", "alpha": 0.8, "pad": 1})
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(composition))
        fig.savefig(output_dir / f"samples_{safe_name}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = ensure_dir(args.output_dir)
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)

    class_counter: Counter[str] = Counter()
    for value in df.get("classes", []).fillna(""):
        class_counter.update(parse_classes_field(value))
    if class_counter:
        labels, values = zip(*class_counter.most_common())
        save_bar(labels, values, "Class Distribution", "Images/objects containing class", output_dir / "class_distribution.png", plt)

    comp_counts = df["composition_type"].fillna("unknown").value_counts()
    save_bar(comp_counts.index.tolist(), comp_counts.values.tolist(), "Composition Distribution", "Images", output_dir / "composition_distribution.png", plt)

    areas, aspects, objects_per_image = collect_bbox_stats(df)
    if areas:
        save_hist(areas, "BBox Area Ratio", "Normalized bbox area", output_dir / "bbox_area_histogram.png", plt)
    if aspects:
        save_hist(aspects, "BBox Aspect Ratio", "Width / height", output_dir / "bbox_aspect_ratio_histogram.png", plt)
    if objects_per_image:
        save_hist(objects_per_image, "Objects Per Image", "Object count", output_dir / "objects_per_image_histogram.png", plt, bins=max(10, min(50, max(objects_per_image) + 1)))

    class_names = load_class_names(metadata_path)
    draw_samples(df, output_dir, class_names, plt)

    if args.eval_csv and Path(args.eval_csv).exists():
        eval_df = pd.read_csv(args.eval_csv)
        all_rows = eval_df[eval_df["composition_type"] == "all"]
        if not all_rows.empty and "mAP50" in all_rows.columns:
            grouped = all_rows.groupby("split_name")["mAP50"].last().sort_index()
            save_bar(grouped.index.tolist(), grouped.values.tolist(), "mAP50 by Split", "mAP50", output_dir / "map_by_split.png", plt)
        comp_rows = eval_df[eval_df["composition_type"] != "all"]
        if not comp_rows.empty:
            if "mAP50" in comp_rows.columns:
                grouped = comp_rows.groupby("composition_type")["mAP50"].last().sort_index()
                save_bar(grouped.index.tolist(), grouped.values.tolist(), "mAP50 by Composition", "mAP50", output_dir / "map_by_composition.png", plt)
            if "recall" in comp_rows.columns:
                grouped = comp_rows.groupby("composition_type")["recall"].last().sort_index()
                save_bar(grouped.index.tolist(), grouped.values.tolist(), "Recall by Composition", "Recall", output_dir / "recall_by_composition.png", plt)

    cigg_csv = output_dir.parent / "cigg_results.csv"
    if cigg_csv.exists():
        cigg_df = pd.read_csv(cigg_csv).dropna(subset=["gap"])
        if not cigg_df.empty:
            labels = cigg_df["gap_name"].astype(str).tolist()
            values = cigg_df["gap"].astype(float).tolist()
            save_bar(labels, values, "Composition Generalization Gaps", "Gap", output_dir / "cigg_comparison.png", plt)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

