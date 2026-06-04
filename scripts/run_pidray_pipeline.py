from __future__ import annotations

import argparse
import importlib.util
import shutil
import shlex
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIDRAY_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1zvMIc1bqteRN9Z36hHYpoTGoZArsh4mE"
RUNPOD_STORAGE_ROOT = "/workspace/CIXDet_data"

HARDWARE_PRESETS = {
    "generic": {"model": "auto", "epochs": 50, "imgsz": 640, "batch": 8, "workers": 4},
    "rtx5080": {"model": "yolo11s.pt", "epochs": 50, "imgsz": 640, "batch": 8, "workers": 4},
    "rtx5090": {"model": "yolo11m.pt", "epochs": 100, "imgsz": 640, "batch": 16, "workers": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PIDray pipeline: download, convert, split, train, evaluate, CIGG, and figures."
    )
    parser.add_argument("--download-url", default=PIDRAY_GDRIVE_FOLDER_URL, help="Official PIDray Google Drive folder URL.")
    parser.add_argument(
        "--download-source",
        choices=["gdrive", "hf", "local"],
        default="gdrive",
        help="Use gdrive download, Hugging Face subset conversion, or an already-downloaded local PIDray folder.",
    )
    parser.add_argument("--hf-repo-id", default="Voxel51/PIDray")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-cache-dir", default="hf_cache")
    parser.add_argument("--hf-max-samples", type=int)
    parser.add_argument("--raw-dir", default="datasets/pidray_raw", help="Where PIDray raw files are downloaded/extracted.")
    parser.add_argument("--yolo-dir", default="dataset_yolo", help="Converted YOLO dataset directory.")
    parser.add_argument("--splits-dir", default="dataset_yolo/splits", help="YOLO split output directory.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument(
        "--storage-root",
        help="Persistent storage root for generated data. Relative raw/yolo/splits/outputs/runs paths are placed under it.",
    )
    parser.add_argument(
        "--runpod",
        action="store_true",
        help=f"Shortcut for --storage-root {RUNPOD_STORAGE_ROOT}, RunPod Pod volume default.",
    )
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--skip-download", action="store_true", help="Use existing --raw-dir.")
    parser.add_argument("--gdown-continue", action="store_true", help="Ask gdown to continue partial downloads when supported.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip archive extraction.")
    parser.add_argument(
        "--delete-archives-after-extract",
        action="store_true",
        help="Delete each archive after it is successfully extracted to reduce persistent volume usage.",
    )
    parser.add_argument("--reuse-conversion", action="store_true", help="Skip COCO to YOLO conversion if metadata.csv exists.")
    parser.add_argument("--experiment-set", choices=["mixed", "core", "balanced", "full"], default="core")
    parser.add_argument(
        "--hardware-preset",
        choices=sorted(HARDWARE_PRESETS),
        default="generic",
        help="Apply training defaults for a GPU class. Explicit --model/--epochs/--imgsz/--batch/--workers override it.",
    )
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none", help="Optional training image cache mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Use yolo11n.pt, 3 epochs, batch 4.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--max-crops-per-class", type=int, default=1000, help="Used only with --experiment-set full.")
    return parser.parse_args()


def apply_hardware_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = HARDWARE_PRESETS[args.hardware_preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resolve_workspace_path(path_value: str, storage_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if storage_root is not None:
        return storage_root / path
    return PROJECT_ROOT / path


def apply_storage_root(args: argparse.Namespace) -> argparse.Namespace:
    if args.runpod and not args.storage_root:
        args.storage_root = RUNPOD_STORAGE_ROOT
    storage_root = Path(args.storage_root).expanduser() if args.storage_root else None
    if storage_root is not None and not args.dry_run:
        storage_root.mkdir(parents=True, exist_ok=True)

    args.raw_dir = resolve_workspace_path(args.raw_dir, storage_root).as_posix()
    args.yolo_dir = resolve_workspace_path(args.yolo_dir, storage_root).as_posix()
    args.splits_dir = resolve_workspace_path(args.splits_dir, storage_root).as_posix()
    args.outputs_dir = resolve_workspace_path(args.outputs_dir, storage_root).as_posix()
    args.project = resolve_workspace_path(args.project, storage_root).as_posix()
    args.hf_cache_dir = resolve_workspace_path(args.hf_cache_dir, storage_root).as_posix()
    args.object_crops_dir = resolve_workspace_path("object_crops", storage_root).as_posix()
    return args


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], dry_run: bool = False) -> None:
    print(f"\n$ {quote_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_download(command: list[str], dry_run: bool = False) -> None:
    try:
        run(command, dry_run)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Dataset download failed. If the log mentions Google Drive quota "
            "('Too many users have viewed or downloaded this file recently'), "
            "this cannot be fixed by the script. Wait and rerun, download the "
            "blocked files manually into the raw directory, or use an alternate mirror.\n"
            "After manual download, rerun with --skip-download."
        ) from exc


def require_python_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise RuntimeError(
        f"Required Python module is not installed in this environment: {module_name}\n"
        f"Install it with:\n  {install_hint}\n"
        "If you created a virtual environment, activate it before running the pipeline."
    )


def gdown_supports_remaining_ok() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "gdown", "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return "--remaining-ok" in result.stdout


def format_gib(num_bytes: int | float) -> str:
    return f"{float(num_bytes) / (1024**3):.2f} GiB"


def estimate_archive_uncompressed_size(archive: Path) -> int:
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zip_file:
                return int(sum(info.file_size for info in zip_file.infolist()))
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                return int(sum(member.size for member in tar.getmembers()))
    except Exception:
        return 0
    return 0


def ensure_enough_space_for_archive(archive: Path, target_parent: Path) -> None:
    estimated = estimate_archive_uncompressed_size(archive)
    if estimated <= 0:
        return
    free = shutil.disk_usage(target_parent).free
    # Leave a small margin for metadata, labels, and filesystem overhead.
    required = int(estimated * 1.05)
    if free < required:
        raise RuntimeError(
            "Not enough free disk space to extract archive.\n"
            f"Archive: {archive}\n"
            f"Estimated extracted size: {format_gib(estimated)}\n"
            f"Free space: {format_gib(free)}\n"
            "Increase the RunPod volume size, remove partial extracted folders, "
            "or use a smaller dataset/subset before rerunning."
        )


def extract_archives(raw_dir: Path, delete_archives_after_extract: bool = False) -> None:
    archive_paths = sorted(
        [
            path
            for path in raw_dir.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() == ".zip"
                or path.suffix.lower() in {".tar", ".gz", ".tgz"}
                or "".join(path.suffixes[-2:]).lower() in {".tar.gz", ".tar.bz2", ".tar.xz"}
            )
        ]
    )
    for archive in archive_paths:
        target = archive.parent / f"{archive.stem}_extracted"
        if target.exists() and any(target.iterdir()):
            print(f"Skipping already extracted archive: {archive}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        ensure_enough_space_for_archive(archive, target.parent)
        print(f"Extracting {archive} -> {target}")
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(target)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                tar.extractall(target)
        else:
            print(f"Warning: unsupported archive format skipped: {archive}")
            continue
        if delete_archives_after_extract:
            archive.unlink()
            print(f"Deleted archive after extraction: {archive}")


def score_json_for_split(path: Path, split: str) -> tuple[int, int]:
    text = path.as_posix().lower()
    stem = path.stem.lower()
    score = 0
    if split in stem:
        score += 10
    if split in text:
        score += 5
    if "annotation" in text or "coco" in text or "instance" in text:
        score += 2
    if split == "train" and "test" in text:
        score -= 5
    return score, path.stat().st_size


def find_pidray_jsons(raw_dir: Path) -> dict[str, Path]:
    jsons = [
        path
        for path in raw_dir.rglob("*.json")
        if not path.name.endswith("_report.json") and "conversion_report" not in path.name
    ]
    result: dict[str, Path] = {}
    for split in ("train", "easy", "hard", "hidden"):
        candidates = [path for path in jsons if score_json_for_split(path, split)[0] > 0]
        if candidates:
            result[split] = sorted(candidates, key=lambda path: score_json_for_split(path, split), reverse=True)[0]
    if not result and len(jsons) == 1:
        result["all"] = jsons[0]
    return result


def train_and_eval(args: argparse.Namespace, split_name: str, run_name: str, model_name: str, epochs: int, batch: int) -> None:
    data_yaml = Path(args.splits_dir) / split_name / "data.yaml"
    metadata_test = Path(args.splits_dir) / split_name / "metadata_test.csv"
    train_command = [
        sys.executable,
        "scripts/train_yolo.py",
        "--data-yaml",
        data_yaml.as_posix(),
        "--model",
        model_name,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--project",
        str(args.project),
        "--name",
        run_name,
        "--seed",
        str(args.seed),
    ]
    if args.cache != "none":
        train_command.extend(["--cache", args.cache])
    run(train_command, args.dry_run)
    weights = Path(args.project) / run_name / "weights" / "best.pt"
    run(
        [
            sys.executable,
            "scripts/evaluate_yolo.py",
            "--weights",
            weights.as_posix(),
            "--data-yaml",
            data_yaml.as_posix(),
            "--metadata",
            metadata_test.as_posix(),
            "--split-name",
            split_name,
            "--output-csv",
            (Path(args.outputs_dir) / "eval_results.csv").as_posix(),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(batch),
            "--workers",
            str(args.workers),
            "--device",
            str(args.device),
        ],
        args.dry_run,
    )


def main() -> None:
    args = apply_storage_root(apply_hardware_preset(parse_args()))
    raw_dir = Path(args.raw_dir)
    yolo_dir = Path(args.yolo_dir)
    splits_dir = Path(args.splits_dir)
    outputs_dir = Path(args.outputs_dir)

    model_name = "yolo11n.pt" if args.debug else args.model
    epochs = 3 if args.debug else args.epochs
    batch = 4 if args.debug else args.batch
    if args.debug:
        args.imgsz = min(int(args.imgsz), 640)
    prefix = "debug_pidray" if args.debug else "pidray"

    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.download_source == "hf":
        hf_command = [
            sys.executable,
            "scripts/convert_hf_detection_to_yolo.py",
            "--repo-id",
            args.hf_repo_id,
            "--split",
            args.hf_split,
            "--output-dir",
            yolo_dir.as_posix(),
            "--hf-cache-dir",
            args.hf_cache_dir,
            "--composition-strategy",
            "object_count_proxy",
        ]
        if args.hf_max_samples is not None:
            hf_command.extend(["--max-samples", str(args.hf_max_samples)])
        run(hf_command, args.dry_run)
    elif args.download_source == "local":
        print(f"Using local PIDray files under: {raw_dir}")
    elif not args.skip_download:
        if not args.dry_run:
            require_python_module("gdown", f"{sys.executable} -m pip install -r requirements.txt")
        download_command = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            args.download_url,
            "-O",
            raw_dir.as_posix(),
        ]
        if args.gdown_continue:
            download_command.append("--continue")
        if not args.dry_run and gdown_supports_remaining_ok():
            download_command.append("--remaining-ok")
        run_download(download_command, args.dry_run)
    else:
        print(f"Skipping download; using existing raw directory: {raw_dir}")

    if args.download_source in {"gdrive", "local"} and not args.skip_extract:
        if args.dry_run:
            print("Dry run: archive extraction would run here.")
        else:
            extract_archives(raw_dir, delete_archives_after_extract=args.delete_archives_after_extract)

    metadata_csv = yolo_dir / "metadata.csv"
    if args.download_source == "hf":
        pass
    elif args.reuse_conversion and metadata_csv.exists():
        print(f"Skipping conversion; using existing metadata: {metadata_csv}")
    else:
        split_jsons = find_pidray_jsons(raw_dir)
        if not split_jsons:
            if args.dry_run:
                print(f"Dry run: no COCO JSON files found under {raw_dir}; conversion commands omitted.")
            else:
                raise RuntimeError(
                    f"No COCO JSON files found under {raw_dir}. "
                    "Download PIDray manually, then rerun with --skip-download --raw-dir <path>."
                )
        else:
            first = True
            for split_name, annotation_json in split_jsons.items():
                command = [
                    sys.executable,
                    "scripts/convert_coco_to_yolo.py",
                    "--images-root",
                    raw_dir.as_posix(),
                    "--annotation-json",
                    annotation_json.as_posix(),
                    "--output-dir",
                    yolo_dir.as_posix(),
                    "--copy-mode",
                    args.copy_mode,
                    "--split-source",
                    split_name,
                ]
                if not first:
                    command.append("--append")
                run(command, args.dry_run)
                first = False

    run(
        [
            sys.executable,
            "scripts/make_splits.py",
            "--metadata",
            metadata_csv.as_posix(),
            "--output-dir",
            splits_dir.as_posix(),
            "--copy-mode",
            args.copy_mode,
            "--seed",
            str(args.seed),
        ],
        args.dry_run,
    )

    train_and_eval(args, "mixed", f"{prefix}_baseline_mixed", model_name, epochs, batch)

    if args.experiment_set in {"core", "balanced", "full"}:
        train_and_eval(args, "simple_to_complex", f"{prefix}_baseline_simple_to_complex", model_name, epochs, batch)
        train_and_eval(
            args,
            "complex_with_non_target_holdout",
            f"{prefix}_baseline_complex_with_non_target_holdout",
            model_name,
            epochs,
            batch,
        )

    if args.experiment_set in {"balanced", "full"}:
        balanced_dir = splits_dir / "balanced_mixed"
        run(
            [
                sys.executable,
                "scripts/make_balanced_training_set.py",
                "--split-dir",
                (splits_dir / "mixed").as_posix(),
                "--metadata",
                (splits_dir / "mixed" / "metadata_train.csv").as_posix(),
                "--output-dir",
                balanced_dir.as_posix(),
                "--copy-mode",
                args.copy_mode,
                "--seed",
                str(args.seed),
            ],
            args.dry_run,
        )
        train_and_eval(args, "balanced_mixed", f"{prefix}_balanced_mixed", model_name, epochs, batch)

    run(
        [
            sys.executable,
            "scripts/compute_cigg.py",
            "--eval-csv",
            (outputs_dir / "eval_results.csv").as_posix(),
            "--output-json",
            (outputs_dir / "cigg_results.json").as_posix(),
            "--output-csv",
            (outputs_dir / "cigg_results.csv").as_posix(),
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            "scripts/visualize_results.py",
            "--metadata",
            metadata_csv.as_posix(),
            "--eval-csv",
            (outputs_dir / "eval_results.csv").as_posix(),
            "--output-dir",
            (outputs_dir / "figures").as_posix(),
        ],
        args.dry_run,
    )

    if args.experiment_set == "full":
        crops_dir = Path(args.object_crops_dir)
        run(
            [
                sys.executable,
                "scripts/extract_object_crops.py",
                "--metadata",
                metadata_csv.as_posix(),
                "--output-dir",
                crops_dir.as_posix(),
                "--max-crops-per-class",
                str(args.max_crops_per_class),
            ],
            args.dry_run,
        )
        run(
            [
                sys.executable,
                "scripts/train_object_contrastive_encoder.py",
                "--crops-metadata",
                (crops_dir / "crops_metadata.csv").as_posix(),
                "--output-dir",
                (outputs_dir / "contrastive").as_posix(),
                "--epochs",
                "30",
                "--batch",
                "64",
                "--device",
                str(args.device),
                "--workers",
                str(args.workers),
            ],
            args.dry_run,
        )
        run(
            [
                sys.executable,
                "scripts/analyze_feature_bias.py",
                "--embeddings",
                (outputs_dir / "contrastive" / "embeddings.npy").as_posix(),
                "--metadata",
                (outputs_dir / "contrastive" / "embeddings_metadata.csv").as_posix(),
                "--output-dir",
                (outputs_dir / "feature_bias").as_posix(),
            ],
            args.dry_run,
        )

    print("\nPIDray pipeline complete.")
    print(f"Evaluation CSV: {outputs_dir / 'eval_results.csv'}")
    print(f"CIGG report: {outputs_dir / 'cigg_results.json'}")
    print(f"Figures: {outputs_dir / 'figures'}")


if __name__ == "__main__":
    main()
