"""Shared path, file, YAML, and metadata helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
ANNOTATION_EXTENSIONS = (".xml", ".json")


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def parse_extensions(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    normalized = []
    for item in items:
        normalized.append(item.lower() if item.startswith(".") else f".{item.lower()}")
    return tuple(normalized)


def scan_files(root: Path | str, extensions: Sequence[str]) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    ext_set = {ext.lower() for ext in extensions}
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in ext_set)


def read_image_size(path: Path | str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def unique_stem(path: Path | str, root: Path | str | None = None) -> str:
    path_obj = Path(path)
    try:
        identity = path_obj.resolve().as_posix()
    except OSError:
        identity = path_obj.as_posix()
    if root is not None:
        try:
            identity = path_obj.resolve().relative_to(Path(root).resolve()).as_posix()
        except (OSError, ValueError):
            pass
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path_obj.stem)
    return f"{safe}_{stable_hash(identity)}"


def materialize_file(src: Path | str, dst: Path | str, mode: str = "symlink") -> str:
    """Copy, symlink, or hardlink src to dst.

    Returns the mode actually used. Symlink and hardlink fall back to copy if the
    platform denies the operation.
    """

    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()

    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return "copy"
    if mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return "hardlink"
        except OSError:
            shutil.copy2(src_path, dst_path)
            return "copy"
    if mode == "symlink":
        try:
            dst_path.symlink_to(src_path.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(src_path, dst_path)
            return "copy"
    raise ValueError(f"Unsupported copy mode: {mode}")


def write_json(path: Path | str, data: object) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path | str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_class_map(path: Path | str) -> tuple[dict[str, int], list[str]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"Class map is empty: {path}")

    if isinstance(data, list):
        names = [str(item) for item in data]
    elif isinstance(data, dict):
        source = data.get("names", data.get("classes", data))
        if isinstance(source, list):
            names = [str(item) for item in source]
        elif isinstance(source, dict):
            int_to_name = {int(idx): str(name) for idx, name in source.items()}
            names = [int_to_name[idx] for idx in sorted(int_to_name)]
        else:
            raise ValueError(f"Unsupported class map format in {path}")
    else:
        raise ValueError(f"Unsupported class map format in {path}")

    class_to_id = {name: idx for idx, name in enumerate(names)}
    return class_to_id, names


def save_class_map(path: Path | str, names: Sequence[str]) -> None:
    import yaml

    target = Path(path)
    ensure_dir(target.parent)
    payload = {"names": {idx: str(name) for idx, name in enumerate(names)}}
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_data_yaml(path: Path | str, split_dir: Path | str, names: Sequence[str]) -> None:
    import yaml

    target = Path(path)
    split_path = Path(split_dir)
    payload = {
        "path": split_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: str(name) for idx, name in enumerate(names)},
    }
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_names_from_data_yaml(path: Path | str) -> list[str]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        int_to_name = {int(idx): str(name) for idx, name in names.items()}
        return [int_to_name[idx] for idx in sorted(int_to_name)]
    return []


def parse_classes_field(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return [item for item in text.split("|") if item]


def classes_to_field(classes: Iterable[str]) -> str:
    return "|".join(sorted({str(item) for item in classes if str(item)}))


def yolo_label_to_xyxy(values: Sequence[float], width: int, height: int) -> list[float]:
    x_center, y_center, box_w, box_h = values
    x_center *= width
    y_center *= height
    box_w *= width
    box_h *= height
    xmin = x_center - box_w / 2.0
    ymin = y_center - box_h / 2.0
    xmax = x_center + box_w / 2.0
    ymax = y_center + box_h / 2.0
    return [xmin, ymin, xmax, ymax]
