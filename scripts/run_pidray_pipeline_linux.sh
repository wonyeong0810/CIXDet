#!/usr/bin/env bash
set -euo pipefail

HARDWARE_PRESET="${HARDWARE_PRESET:-rtx5090}"
EXPERIMENT_SET="${EXPERIMENT_SET:-balanced}"
MODEL="${MODEL:-}"
EPOCHS="${EPOCHS:-}"
IMGSZ="${IMGSZ:-}"
BATCH="${BATCH:-}"
WORKERS="${WORKERS:-}"
DEVICE="${DEVICE:-0}"
RAW_DIR="${RAW_DIR:-datasets/pidray_raw}"

CMD=(python scripts/run_pidray_pipeline.py \
  --raw-dir "$RAW_DIR" \
  --hardware-preset "$HARDWARE_PRESET" \
  --experiment-set "$EXPERIMENT_SET" \
  --device "$DEVICE")

if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
if [[ -n "$EPOCHS" ]]; then CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$IMGSZ" ]]; then CMD+=(--imgsz "$IMGSZ"); fi
if [[ -n "$BATCH" ]]; then CMD+=(--batch "$BATCH"); fi
if [[ -n "$WORKERS" ]]; then CMD+=(--workers "$WORKERS"); fi

"${CMD[@]}"
