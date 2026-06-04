from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure object-class signal and composition leakage in crop embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="outputs/feature_bias")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    output_dir = ensure_dir(args.output_dir)
    embeddings = np.load(args.embeddings)
    metadata = pd.read_csv(args.metadata)
    if len(metadata) != len(embeddings):
        raise RuntimeError(f"Embedding count ({len(embeddings)}) does not match metadata rows ({len(metadata)}).")

    def stratify_or_none(labels):
        values, counts = np.unique(labels, return_counts=True)
        return labels if len(values) > 1 and counts.min() >= 2 else None

    def fit_probe(column: str, filename: str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(metadata[column].astype(str))
        if len(encoder.classes_) < 2:
            return {"accuracy": None, "classes": encoder.classes_.tolist(), "message": f"Only one {column} class present."}
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify_or_none(y),
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        acc = float(accuracy_score(y_test, pred))
        cm = confusion_matrix(y_test, pred, labels=list(range(len(encoder.classes_))))
        fig, ax = plt.subplots(figsize=(max(5, len(encoder.classes_) * 0.5), max(4, len(encoder.classes_) * 0.45)))
        ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title(f"{column} probe accuracy: {acc:.3f}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
        return {"accuracy": acc, "classes": encoder.classes_.tolist()}

    class_probe = fit_probe("class_name", "class_confusion_matrix.png")
    composition_probe = fit_probe("composition_type", "composition_confusion_matrix.png")

    per_composition_rows = []
    class_encoder = LabelEncoder()
    y_class = class_encoder.fit_transform(metadata["class_name"].astype(str))
    if len(class_encoder.classes_) >= 2:
        x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
            embeddings,
            y_class,
            metadata,
            test_size=0.25,
            random_state=42,
            stratify=stratify_or_none(y_class),
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        eval_frame = meta_test.copy()
        eval_frame["correct"] = pred == y_test
        for composition, group in eval_frame.groupby("composition_type"):
            per_composition_rows.append(
                {
                    "composition_type": composition,
                    "object_accuracy": float(group["correct"].mean()),
                    "num_samples": int(len(group)),
                }
            )

    report = {
        "object_class_accuracy": class_probe.get("accuracy"),
        "composition_leakage_accuracy": composition_probe.get("accuracy"),
        "class_probe": class_probe,
        "composition_probe": composition_probe,
        "per_composition_object_accuracy": per_composition_rows,
        "interpretation": "Good object features should have high object-class accuracy and relatively lower composition leakage.",
    }
    (output_dir / "feature_bias_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"metric": "object_class_accuracy", "value": class_probe.get("accuracy")},
            {"metric": "composition_leakage_accuracy", "value": composition_probe.get("accuracy")},
        ]
        + [{"metric": f"object_accuracy_{row['composition_type']}", "value": row["object_accuracy"]} for row in per_composition_rows]
    ).to_csv(output_dir / "feature_bias_report.csv", index=False)
    print(f"Feature-bias report saved to {output_dir}")


if __name__ == "__main__":
    main()

