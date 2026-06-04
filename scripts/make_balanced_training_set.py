from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_names_from_data_yaml, materialize_file, parse_classes_field, write_data_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a composition-balanced mixed YOLO split.")
    parser.add_argument("--split-dir", required=True, help="Existing mixed split directory.")
    parser.add_argument("--metadata", required=True, help="metadata_train.csv from the mixed split.")
    parser.add_argument("--output-dir", required=True, help="Output balanced_mixed split directory.")
    parser.add_argument("--target-composition-policy", choices=["equal", "max", "custom"], default="equal")
    parser.add_argument("--rare-class-oversample", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def composition_balance(df, seed: int, policy: str):
    import pandas as pd

    parts = []
    counts = df["composition_type"].fillna("unknown").value_counts()
    if counts.empty:
        return df.copy(), counts.to_dict(), {}
    target_count = int(counts.max())
    if policy == "custom":
        print("Warning: custom policy currently uses the max observed composition count. Edit this script for project-specific targets.")
    for composition, group in df.groupby(df["composition_type"].fillna("unknown")):
        replace = len(group) < target_count
        sampled = group.sample(n=target_count, replace=replace, random_state=seed)
        parts.append(sampled)
    balanced = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced, counts.to_dict(), balanced["composition_type"].fillna("unknown").value_counts().to_dict()


def rare_class_balance(df, seed: int):
    import pandas as pd

    rng = random.Random(seed)
    class_counts: Counter[str] = Counter()
    row_classes: list[list[str]] = []
    for _, row in df.iterrows():
        classes = parse_classes_field(row.get("classes", ""))
        row_classes.append(classes)
        class_counts.update(set(classes))
    if not class_counts:
        return df, {}

    target = int(sorted(class_counts.values())[len(class_counts) // 2])
    additions = []
    for class_name, count in class_counts.items():
        if count >= target:
            continue
        candidates = [idx for idx, classes in enumerate(row_classes) if class_name in classes]
        if not candidates:
            continue
        for _ in range(target - count):
            additions.append(df.iloc[rng.choice(candidates)])
    if additions:
        df = pd.concat([df, pd.DataFrame(additions)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, dict(class_counts)


def materialize_subset(df, output_dir: Path, subset: str, copy_mode: str, duplicate_safe: bool = False):
    rows = []
    ensure_dir(output_dir / "images" / subset)
    ensure_dir(output_dir / "labels" / subset)
    for idx, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        image_src = Path(row["image_path"])
        label_src = Path(row["label_path"])
        suffix = f"_dup{idx:06d}" if duplicate_safe else ""
        image_name = f"{image_src.stem}{suffix}{image_src.suffix}"
        label_name = f"{label_src.stem}{suffix}.txt"
        image_dst = output_dir / "images" / subset / image_name
        label_dst = output_dir / "labels" / subset / label_name
        materialize_file(image_src, image_dst, copy_mode)
        if label_src.exists():
            materialize_file(label_src, label_dst, copy_mode)
        else:
            label_dst.write_text("", encoding="utf-8")
        new_row = row.to_dict()
        new_row["image_path"] = image_dst.as_posix()
        new_row["label_path"] = label_dst.as_posix()
        new_row["split_source"] = "balanced_mixed"
        rows.append(new_row)
    return rows


def main() -> None:
    args = parse_args()
    import pandas as pd

    split_dir = Path(args.split_dir)
    output_dir = ensure_dir(args.output_dir)
    train_df = pd.read_csv(args.metadata)
    if "composition_type" not in train_df.columns:
        raise RuntimeError("Training metadata must include composition_type.")

    balanced_df, before_counts, after_counts = composition_balance(train_df, args.seed, args.target_composition_policy)
    rare_before = {}
    if args.rare_class_oversample:
        balanced_df, rare_before = rare_class_balance(balanced_df, args.seed)
        after_counts = balanced_df["composition_type"].fillna("unknown").value_counts().to_dict()

    metadata_rows = {
        "train": materialize_subset(balanced_df, output_dir, "train", args.copy_mode, duplicate_safe=True)
    }
    for subset in ("val", "test"):
        source_metadata = split_dir / f"metadata_{subset}.csv"
        if source_metadata.exists():
            subset_df = pd.read_csv(source_metadata)
        else:
            print(f"Warning: {source_metadata} not found; creating empty {subset} subset.")
            subset_df = train_df.iloc[0:0].copy()
        metadata_rows[subset] = materialize_subset(subset_df, output_dir, subset, args.copy_mode, duplicate_safe=False)

    for subset, rows in metadata_rows.items():
        pd.DataFrame(rows).to_csv(output_dir / f"metadata_{subset}.csv", index=False)

    names = load_names_from_data_yaml(split_dir / "data.yaml")
    write_data_yaml(output_dir / "data.yaml", output_dir, names)

    report_rows = []
    for composition, count in before_counts.items():
        report_rows.append({"composition_type": composition, "stage": "before", "count": count})
    for composition, count in after_counts.items():
        report_rows.append({"composition_type": composition, "stage": "after", "count": count})
    ensure_dir("outputs")
    pd.DataFrame(report_rows).to_csv("outputs/balancing_report.csv", index=False)
    write_json(
        "outputs/balancing_report.json",
        {
            "policy": args.target_composition_policy,
            "rare_class_oversample": args.rare_class_oversample,
            "composition_before": before_counts,
            "composition_after": after_counts,
            "rare_class_counts_before": rare_before,
            "num_train_rows_before": len(train_df),
            "num_train_rows_after": len(balanced_df),
        },
    )
    print(f"Balanced split written to {output_dir}")


if __name__ == "__main__":
    main()
