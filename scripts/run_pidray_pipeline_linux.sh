#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
GDOWN_CONTINUE="${GDOWN_CONTINUE:-1}"
HARDWARE_PRESET="${HARDWARE_PRESET:-rtx5090}"
EXPERIMENT_SET="${EXPERIMENT_SET:-balanced}"
MODEL="${MODEL:-}"
EPOCHS="${EPOCHS:-}"
IMGSZ="${IMGSZ:-}"
BATCH="${BATCH:-}"
WORKERS="${WORKERS:-}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-none}"
RAW_DIR="${RAW_DIR:-datasets/pidray_raw}"
STORAGE_ROOT="${STORAGE_ROOT:-/workspace/CIXDet_data}"
DELETE_ARCHIVES_AFTER_EXTRACT="${DELETE_ARCHIVES_AFTER_EXTRACT:-1}"

if [[ "$INSTALL_REQUIREMENTS" != "0" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

CMD=("$PYTHON_BIN" scripts/run_pidray_pipeline.py \
  --raw-dir "$RAW_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --hardware-preset "$HARDWARE_PRESET" \
  --experiment-set "$EXPERIMENT_SET" \
  --cache "$CACHE" \
  --device "$DEVICE")

if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
if [[ -n "$EPOCHS" ]]; then CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$IMGSZ" ]]; then CMD+=(--imgsz "$IMGSZ"); fi
if [[ -n "$BATCH" ]]; then CMD+=(--batch "$BATCH"); fi
if [[ -n "$WORKERS" ]]; then CMD+=(--workers "$WORKERS"); fi
if [[ "$DELETE_ARCHIVES_AFTER_EXTRACT" != "0" ]]; then CMD+=(--delete-archives-after-extract); fi
if [[ "$SKIP_DOWNLOAD" != "0" ]]; then CMD+=(--skip-download); fi
if [[ "$GDOWN_CONTINUE" != "0" ]]; then CMD+=(--gdown-continue); fi

"${CMD[@]}"
