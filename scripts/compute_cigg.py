from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_cigg, compute_gap
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CI-XDet composition generalization gaps from eval CSVs.")
    parser.add_argument("--eval-csv", default="outputs/eval_results.csv")
    parser.add_argument("--output-json", default="outputs/cigg_results.json")
    parser.add_argument("--output-csv", default="outputs/cigg_results.csv")
    return parser.parse_args()


def pick_metric(df, split_name: str, composition_type: str = "all", metric: str = "mAP50") -> float | None:
    subset = df[(df["split_name"] == split_name) & (df["composition_type"] == composition_type)]
    if subset.empty:
        return None
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        return None
    return float(subset.iloc[-1][metric])


def add_gap(rows: list[dict[str, object]], name: str, reference: float | None, target: float | None, metric: str) -> None:
    if reference is None or target is None:
        rows.append({"gap_name": name, "metric": metric, "reference": reference, "target": target, "gap": None})
    else:
        rows.append({"gap_name": name, "metric": metric, "reference": reference, "target": target, "gap": compute_gap(reference, target)})


def main() -> None:
    args = parse_args()
    import pandas as pd

    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv}")

    df = pd.read_csv(eval_csv)
    required = {"split_name", "composition_type", "mAP50"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Eval CSV is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    missing_messages: list[str] = []

    mixed_map = pick_metric(df, "mixed", "all", "mAP50")
    simple_to_complex_map = pick_metric(df, "simple_to_complex", "all", "mAP50")
    complex_with_non_target_holdout_map = pick_metric(df, "complex_with_non_target_holdout", "all", "mAP50")
    complex_with_non_target_subset_map = pick_metric(df, "mixed", "complex_with_non_target", "mAP50")

    if mixed_map is None:
        missing_messages.append("Run evaluate_yolo.py for the mixed split.")
    if simple_to_complex_map is None:
        missing_messages.append("Run evaluate_yolo.py for the simple_to_complex split.")
    if complex_with_non_target_holdout_map is None:
        missing_messages.append("Run evaluate_yolo.py for the complex_with_non_target_holdout split.")
    if complex_with_non_target_subset_map is None:
        missing_messages.append("Run mixed evaluation with --metadata to get composition-specific complex_with_non_target rows.")

    if mixed_map is not None and simple_to_complex_map is not None:
        rows.append(
            {
                "gap_name": "CIGG_simple_to_complex",
                "metric": "mAP50",
                "reference": mixed_map,
                "target": simple_to_complex_map,
                "gap": compute_cigg(mixed_map, simple_to_complex_map),
            }
        )
    add_gap(rows, "CIGG_complex_with_non_target_holdout", mixed_map, complex_with_non_target_holdout_map, "mAP50")
    add_gap(rows, "mixed_complex_with_non_target_subset_gap", mixed_map, complex_with_non_target_subset_map, "mAP50")

    for metric in ("recall", "mAP50_95"):
        if metric in df.columns:
            add_gap(rows, f"{metric}_simple_to_complex_gap", pick_metric(df, "mixed", "all", metric), pick_metric(df, "simple_to_complex", "all", metric), metric)
            add_gap(rows, f"{metric}_complex_with_non_target_holdout_gap", pick_metric(df, "mixed", "all", metric), pick_metric(df, "complex_with_non_target_holdout", "all", metric), metric)

    per_class_csv = eval_csv.parent / "per_class_eval_results.csv"
    per_class_rows: list[dict[str, object]] = []
    if per_class_csv.exists():
        per_df = pd.read_csv(per_class_csv)
        if {"split_name", "composition_type", "class_name", "mAP50_95"}.issubset(per_df.columns):
            mixed = per_df[(per_df["split_name"] == "mixed") & (per_df["composition_type"] == "all")]
            target = per_df[(per_df["split_name"] == "simple_to_complex") & (per_df["composition_type"] == "all")]
            for class_name in sorted(set(mixed["class_name"]) & set(target["class_name"])):
                ref_value = float(mixed[mixed["class_name"] == class_name].iloc[-1]["mAP50_95"])
                target_value = float(target[target["class_name"] == class_name].iloc[-1]["mAP50_95"])
                per_class_rows.append(
                    {
                        "gap_name": "per_class_simple_to_complex_gap",
                        "class_name": class_name,
                        "metric": "mAP50_95",
                        "reference": ref_value,
                        "target": target_value,
                        "gap": compute_gap(ref_value, target_value),
                    }
                )

    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    pd.DataFrame(rows + per_class_rows).to_csv(output_csv, index=False)
    write_json(
        args.output_json,
        {
            "gaps": rows,
            "per_class_gaps": per_class_rows,
            "missing_evaluations": missing_messages,
            "definition": "CIGG = mixed_test_mAP50 - complex_holdout_mAP50",
        },
    )

    if missing_messages:
        print("CIGG computed where possible, but some evaluations are missing:")
        for message in missing_messages:
            print(f"- {message}")
    print(f"Saved CIGG results to {output_csv}")


if __name__ == "__main__":
    main()

