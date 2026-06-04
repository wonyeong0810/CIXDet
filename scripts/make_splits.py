from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.composition import known_compositions
from src.utils import ensure_dir, load_names_from_data_yaml, materialize_file, parse_classes_field, write_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CI-XDet research splits from converted YOLO metadata.")
    parser.add_argument("--metadata", required=True, help="dataset_yolo/metadata.csv")
    parser.add_argument("--output-dir", required=True, help="dataset_yolo/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--min-val-images", type=int, default=100)
    return parser.parse_args()


def infer_class_names(metadata_path: Path, df) -> list[str]:
    classes_txt = metadata_path.parent / "classes.txt"
    if classes_txt.exists():
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_yaml = metadata_path.parent / "data.yaml"
    if data_yaml.exists():
        return load_names_from_data_yaml(data_yaml)
    names = sorted({name for value in df["classes"].fillna("") for name in parse_classes_field(value)})
    if not names:
        raise RuntimeError("Could not infer class names. Expected classes.txt next to metadata.csv.")
    return names


def safe_train_test_split(df, test_size: float, seed: int, stratify_col: str | None = None):
    from sklearn.model_selection import train_test_split

    if len(df) < 2 or test_size <= 0:
        return df.copy(), df.iloc[0:0].copy()
    stratify = None
    if stratify_col and stratify_col in df.columns:
        counts = df[stratify_col].value_counts()
        if len(counts) > 1 and counts.min() >= 2 and int(round(len(df) * test_size)) >= len(counts):
            stratify = df[stratify_col]
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
        return train_df.copy(), test_df.copy()
    except ValueError:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        return train_df.copy(), test_df.copy()


def split_val_from_train(train_df, val_ratio: float, min_val_images: int, seed: int):
    if len(train_df) < 2:
        return train_df.copy(), train_df.iloc[0:0].copy()
    val_count = max(min_val_images, int(round(len(train_df) * val_ratio)))
    val_count = min(max(1, val_count), max(1, len(train_df) - 1))
    val_ratio_adjusted = val_count / float(len(train_df))
    return safe_train_test_split(train_df, val_ratio_adjusted, seed, "composition_type")


def create_split(split_name: str, split_parts: dict[str, object], output_dir: Path, class_names: list[str], copy_mode: str) -> None:
    split_dir = ensure_dir(output_dir / split_name)
    for subset in ("train", "val", "test"):
        ensure_dir(split_dir / "images" / subset)
        ensure_dir(split_dir / "labels" / subset)

    for subset, df in split_parts.items():
        rows = []
        for _, row in df.reset_index(drop=True).iterrows():
            image_src = Path(row["image_path"])
            label_src = Path(row["label_path"])
            image_dst = split_dir / "images" / subset / image_src.name
            label_dst = split_dir / "labels" / subset / label_src.name
            materialize_file(image_src, image_dst, copy_mode)
            if label_src.exists():
                materialize_file(label_src, label_dst, copy_mode)
            else:
                label_dst.write_text("", encoding="utf-8")
            new_row = row.to_dict()
            new_row["image_path"] = image_dst.as_posix()
            new_row["label_path"] = label_dst.as_posix()
            new_row["split_source"] = split_name
            rows.append(new_row)
        output_df = df.__class__(rows) if rows else df.iloc[0:0].copy()
        output_df.to_csv(split_dir / f"metadata_{subset}.csv", index=False)

    write_data_yaml(split_dir / "data.yaml", split_dir, class_names)
    print(
        f"Created {split_name}: "
        f"train={len(split_parts['train'])}, val={len(split_parts['val'])}, test={len(split_parts['test'])}"
    )


def make_random_mixed(df, args):
    train_val, test = safe_train_test_split(df, args.test_ratio, args.seed, "composition_type")
    train, val = split_val_from_train(train_val, args.val_ratio, args.min_val_images, args.seed)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    args = parse_args()
    import pandas as pd

    random.seed(args.seed)
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)
    required = {"image_path", "label_path", "composition_type"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Metadata is missing required columns: {sorted(missing)}")

    output_dir = ensure_dir(args.output_dir)
    class_names = infer_class_names(metadata_path, df)

    create_split("mixed", make_random_mixed(df, args), output_dir, class_names, args.copy_mode)

    simple = df[df["composition_type"].isin(["single_basic", "single_with_non_target"])]
    complex_df = df[df["composition_type"].isin(["complex_target", "complex_with_non_target"])]
    if len(simple) == 0 or len(complex_df) == 0:
        print("Warning: simple_to_complex lacks simple or complex samples; falling back to random mixed split.")
        simple_to_complex = make_random_mixed(df, args)
    else:
        train, val = split_val_from_train(simple, args.val_ratio, args.min_val_images, args.seed)
        simple_to_complex = {"train": train, "val": val, "test": complex_df.copy()}
    create_split("simple_to_complex", simple_to_complex, output_dir, class_names, args.copy_mode)

    holdout = df[df["composition_type"] == "complex_with_non_target"]
    train_pool = df[df["composition_type"] != "complex_with_non_target"]
    if len(holdout) == 0 or len(train_pool) == 0:
        print("Warning: complex_with_non_target_holdout is not possible; falling back to random mixed split.")
        holdout_split = make_random_mixed(df, args)
    else:
        train, val = split_val_from_train(train_pool, args.val_ratio, args.min_val_images, args.seed)
        holdout_split = {"train": train, "val": val, "test": holdout.copy()}
    create_split("complex_with_non_target_holdout", holdout_split, output_dir, class_names, args.copy_mode)

    for composition in known_compositions():
        held_out = df[df["composition_type"] == composition]
        train_pool = df[(df["composition_type"] != composition) & (df["composition_type"].isin(known_compositions()))]
        split_name = f"leave_out_{composition}"
        if len(held_out) == 0 or len(train_pool) == 0:
            print(f"Warning: skipping {split_name}; not enough known samples.")
            continue
        train, val = split_val_from_train(train_pool, args.val_ratio, args.min_val_images, args.seed)
        create_split(split_name, {"train": train, "val": val, "test": held_out.copy()}, output_dir, class_names, args.copy_mode)


if __name__ == "__main__":
    main()
