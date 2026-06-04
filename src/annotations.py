"""Defensive annotation parsing and YOLO bbox conversion helpers."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence


PARSE_WARNINGS: list[dict[str, str]] = []

_CLASS_KEYS = ("class_name", "class", "label", "category", "category_name", "name", "item")
_BBOX_KEYS = ("bbox", "box", "boxes", "bndbox", "bounding_box", "bounds")


def clear_annotation_warnings() -> None:
    PARSE_WARNINGS.clear()


def get_annotation_warnings() -> list[dict[str, str]]:
    return list(PARSE_WARNINGS)


def _warn(path: Path, message: str) -> None:
    PARSE_WARNINGS.append({"path": str(path), "message": message})


def parse_annotation(path: Path) -> list[dict[str, Any]]:
    """Parse an XML or JSON annotation file into object dictionaries."""

    path = Path(path)
    try:
        if path.suffix.lower() == ".xml":
            return parse_xml_annotation(path)
        if path.suffix.lower() == ".json":
            return parse_json_annotation(path)
        _warn(path, f"Unsupported annotation extension: {path.suffix}")
        return []
    except Exception as exc:  # Defensive by design for heterogeneous AIHub files.
        _warn(path, f"Failed to parse annotation: {exc}")
        return []


def parse_xml_annotation(path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    objects: list[dict[str, Any]] = []

    candidates = list(root.findall(".//object"))
    if not candidates:
        candidates = [
            node
            for node in root.iter()
            if any(_child_text(node, key) for key in _CLASS_KEYS)
            and (_bbox_from_xml_node(node) is not None)
        ]

    for node in candidates:
        class_name = _class_from_xml_node(node)
        bbox = _bbox_from_xml_node(node)
        if class_name and bbox and validate_bbox(bbox):
            objects.append({"class_name": class_name, "bbox": bbox})

    if not objects:
        child_tags = sorted({child.tag for child in list(root)[:20]})
        _warn(path, f"Unrecognized XML schema or no valid boxes. Top-level keys: {child_tags}")
    return objects


def parse_json_annotation(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    objects: list[dict[str, Any]] = []

    for item in _iter_json_object_candidates(data):
        parsed = _object_from_json_dict(item)
        if parsed is not None:
            objects.append(parsed)

    if not objects:
        objects.extend(_parse_parallel_json_objects(data))

    if not objects:
        keys = sorted(data.keys())[:20] if isinstance(data, dict) else [type(data).__name__]
        _warn(path, f"Unrecognized JSON schema or no valid boxes. Keys: {keys}")
    return objects


def _child_text(node: ET.Element, key: str) -> str | None:
    for child in list(node):
        if child.tag.lower().split("}")[-1] == key:
            if child.text and child.text.strip():
                return child.text.strip()
    found = node.find(f".//{key}")
    if found is not None and found.text and found.text.strip():
        return found.text.strip()
    return None


def _class_from_xml_node(node: ET.Element) -> str | None:
    for key in _CLASS_KEYS:
        value = _child_text(node, key)
        if value:
            return value
    return None


def _float_text(node: ET.Element, key: str) -> float | None:
    value = _child_text(node, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bbox_from_xml_node(node: ET.Element) -> list[float] | None:
    box_node = node.find(".//bndbox")
    target = box_node if box_node is not None else node
    xmin = _float_text(target, "xmin")
    ymin = _float_text(target, "ymin")
    xmax = _float_text(target, "xmax")
    ymax = _float_text(target, "ymax")
    if None not in (xmin, ymin, xmax, ymax):
        return [float(xmin), float(ymin), float(xmax), float(ymax)]

    x = _float_text(target, "x")
    y = _float_text(target, "y")
    width = _float_text(target, "width") or _float_text(target, "w")
    height = _float_text(target, "height") or _float_text(target, "h")
    if None not in (x, y, width, height):
        return [float(x), float(y), float(x + width), float(y + height)]
    return None


def _iter_json_object_candidates(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_json_object_candidates(item)
        return

    if not isinstance(data, dict):
        return

    if _looks_like_json_object(data):
        yield data

    for key in ("annotations", "objects", "labels", "items", "instances", "shapes"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                yield from _iter_json_object_candidates(item)
        elif isinstance(value, dict):
            yield from _iter_json_object_candidates(value)


def _looks_like_json_object(item: dict[str, Any]) -> bool:
    has_class = any(key in item for key in _CLASS_KEYS)
    has_bbox = any(key in item for key in _BBOX_KEYS) or any(
        key in item for key in ("xmin", "ymin", "xmax", "ymax", "x", "y", "w", "h", "width", "height")
    )
    return has_class and has_bbox


def _object_from_json_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    class_name = _json_class_name(item)
    bbox = _json_bbox(item)
    if class_name and bbox and validate_bbox(bbox):
        return {"class_name": class_name, "bbox": bbox}
    return None


def _json_class_name(item: dict[str, Any]) -> str | None:
    for key in _CLASS_KEYS:
        value = item.get(key)
        if isinstance(value, dict):
            nested = _json_class_name(value)
            if nested:
                return nested
        elif value is not None and str(value).strip():
            return str(value).strip()
    return None


def _json_bbox(item: dict[str, Any]) -> list[float] | None:
    for key in _BBOX_KEYS:
        if key in item:
            parsed = _parse_bbox_value(item[key], item)
            if parsed is not None:
                return parsed

    if all(key in item for key in ("xmin", "ymin", "xmax", "ymax")):
        return [float(item["xmin"]), float(item["ymin"]), float(item["xmax"]), float(item["ymax"])]

    x_key = "x" if "x" in item else "left" if "left" in item else None
    y_key = "y" if "y" in item else "top" if "top" in item else None
    w_key = "w" if "w" in item else "width" if "width" in item else None
    h_key = "h" if "h" in item else "height" if "height" in item else None
    if x_key and y_key and w_key and h_key:
        x = float(item[x_key])
        y = float(item[y_key])
        return [x, y, x + float(item[w_key]), y + float(item[h_key])]
    return None


def _parse_bbox_value(value: Any, parent: dict[str, Any]) -> list[float] | None:
    if isinstance(value, dict):
        return _json_bbox(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            numbers = [float(value[idx]) for idx in range(4)]
        except (TypeError, ValueError):
            if value and isinstance(value[0], (list, tuple, dict)):
                return _parse_bbox_value(value[0], parent)
            return None
        format_hint = str(parent.get("bbox_format", parent.get("format", ""))).lower()
        if "xyxy" in format_hint or "xmax" in format_hint:
            return numbers
        x, y, w, h = numbers
        return [x, y, x + w, y + h]
    return None


def _parse_parallel_json_objects(data: Any) -> list[dict[str, Any]]:
    """Handle simple schemas with parallel boxes and labels arrays."""

    parsed: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            parsed.extend(_parse_parallel_json_objects(item))
        return parsed
    if not isinstance(data, dict):
        return parsed

    boxes = data.get("boxes", data.get("bboxes"))
    labels = data.get("labels", data.get("classes", data.get("categories")))
    if isinstance(boxes, list) and isinstance(labels, list):
        for box, label in zip(boxes, labels):
            bbox = _parse_bbox_value(box, data)
            if bbox and validate_bbox(bbox):
                parsed.append({"class_name": str(label), "bbox": bbox})

    for key in ("annotations", "objects", "data", "images"):
        value = data.get(key)
        if isinstance(value, (dict, list)):
            parsed.extend(_parse_parallel_json_objects(value))
    return parsed


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> list[float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    xmin = min(max(xmin, 0.0), float(width))
    xmax = min(max(xmax, 0.0), float(width))
    ymin = min(max(ymin, 0.0), float(height))
    ymax = min(max(ymax, 0.0), float(height))
    return [xmin, ymin, xmax, ymax]


def validate_bbox(bbox: Sequence[float], min_size: float = 1.0) -> bool:
    if len(bbox) != 4:
        return False
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        return False
    return (xmax - xmin) >= min_size and (ymax - ymin) >= min_size


def convert_xyxy_to_yolo(bbox: Sequence[float], width: int, height: int) -> list[float] | None:
    clipped = clip_bbox(bbox, width, height)
    if not validate_bbox(clipped):
        return None
    xmin, ymin, xmax, ymax = clipped
    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + box_w / 2.0
    y_center = ymin + box_h / 2.0
    return [
        x_center / float(width),
        y_center / float(height),
        box_w / float(width),
        box_h / float(height),
    ]


def find_matching_image_for_annotation(
    annotation_path: Path,
    image_paths: Sequence[Path],
    image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Path | None:
    """Find an image with the same stem as an annotation path."""

    annotation_path = Path(annotation_path)
    ext_set = {ext.lower() for ext in image_extensions}
    same_dir_candidates = [
        annotation_path.with_suffix(ext)
        for ext in ext_set
        if annotation_path.with_suffix(ext).exists()
    ]
    if same_dir_candidates:
        return sorted(same_dir_candidates)[0]

    by_stem: dict[str, list[Path]] = {}
    for image_path in image_paths:
        by_stem.setdefault(image_path.stem, []).append(image_path)
    matches = by_stem.get(annotation_path.stem, [])
    if matches:
        same_parent = [path for path in matches if path.parent == annotation_path.parent]
        return sorted(same_parent or matches)[0]
    return None
