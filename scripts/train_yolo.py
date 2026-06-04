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
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO baseline for CI-XDet.")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--model", default="auto", help="YOLO model path/name, or auto.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="ci_xdet_baseline")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_cuda_info(torch_module) -> None:
    if torch_module.cuda.is_available():
        device_index = torch_module.cuda.current_device()
        props = torch_module.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / (1024**3)
        print(f"CUDA available: {props.name} ({total_gb:.1f} GB VRAM)")
    else:
        print("CUDA is not available; training will run on CPU unless Ultralytics selects otherwise.")


def resolve_model(model_arg: str):
    from ultralytics import YOLO

    if model_arg != "auto":
        return model_arg, YOLO(model_arg)

    for candidate in ("yolo11s.pt", "yolov8s.pt"):
        try:
            model = YOLO(candidate)
            print(f"Using model: {candidate}")
            return candidate, model
        except Exception as exc:
            print(f"Could not initialize {candidate}: {exc}")
    raise RuntimeError("Unable to initialize yolo11s.pt or yolov8s.pt. Provide --model explicitly.")


def main() -> None:
    args = parse_args()
    import torch

    print_cuda_info(torch)
    model_name, model = resolve_model(args.model)

    train_config = vars(args).copy()
    train_config["resolved_model"] = model_name
    train_config["cuda_available"] = bool(torch.cuda.is_available())

    try:
        results = model.train(
            data=args.data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            amp=args.amp,
            patience=args.patience,
            seed=args.seed,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("\nCUDA out-of-memory detected.")
            print("Try: --batch 4, --imgsz 512, a smaller model, or fewer --workers.")
        raise

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    ensure_dir(save_dir)
    (save_dir / "training_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    print(f"Training complete. Run directory: {save_dir}")


if __name__ == "__main__":
    main()

