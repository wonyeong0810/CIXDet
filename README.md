# CI-XDet: Composition-Invariant X-ray Prohibited Item Detection

CI-XDet is a research-only prototype for offline object detection experiments on an already-downloaded AIHub X-ray prohibited item detection dataset.

The central question is whether detectors trained with ordinary random splits hide a gap between simple object compositions and complex baggage compositions. This project studies:

- How much detector performance drops from simple compositions to complex compositions.
- Whether `complex_with_non_target` samples cause a larger drop.
- Whether composition-balanced training reduces the gap.
- Whether object-level features cluster by object class or by composition type.

This repository does not download, scrape, collect, generate, upload, export, or redistribute AIHub data. It does not assume multiple scanners or scanner-domain labels. For this prototype, the dataset is treated as one scanner setup, and composition types are inferred from folder/file names or metadata.

## Composition Types

The useful AIHub composition/generation labels are mapped as:

- `단일기본` -> `single_basic`
- `단일비품목` -> `single_with_non_target`
- `복합품목` -> `complex_target`
- `복합비품목` -> `complex_with_non_target`

Unknown or unmatched names are kept as `unknown`.

## Hardware Defaults

Target environment:

- RAM: 32 GB
- GPU: NVIDIA RTX 5080
- VRAM: 16 GB
- Single GPU
- CUDA when available
- Mixed precision enabled by default
- Dataloader workers: 4

Recommended runs:

- Debug: `yolo11n.pt` or `yolov8n.pt`, `imgsz=640`, `batch=4`, `epochs=3`
- Baseline: `yolo11s.pt` or `yolov8s.pt`, `imgsz=640`, `batch=8`, `epochs=50`
- Stronger: `yolo11m.pt` or `yolov8m.pt`, `imgsz=640` or `768`, `batch=4` or `8`, `epochs=100`

If CUDA runs out of memory, reduce batch size to `4`, reduce image size to `512`, use a smaller model, or reduce dataloader workers.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Commands

Inspect dataset:

```bash
python scripts/inspect_aihub_dataset.py --dataset-root /path/to/aihub_xray --output-dir outputs
```

Convert to YOLO:

```bash
python scripts/convert_aihub_to_yolo.py --dataset-root /path/to/aihub_xray --output-dir dataset_yolo --class-map configs/class_map.yaml.example
```

The example class map is intentionally empty. Leave it empty to auto-discover classes, or edit it first to force a stable class order.

Convert a COCO-format public X-ray dataset such as PIDray:

```bash
python scripts/convert_coco_to_yolo.py --images-root /path/to/PIDray/images --annotation-json /path/to/PIDray/annotations/train.json --output-dir dataset_yolo --split-source train
python scripts/convert_coco_to_yolo.py --images-root /path/to/PIDray/images --annotation-json /path/to/PIDray/annotations/easy.json --output-dir dataset_yolo --split-source easy --append
python scripts/convert_coco_to_yolo.py --images-root /path/to/PIDray/images --annotation-json /path/to/PIDray/annotations/hard.json --output-dir dataset_yolo --split-source hard --append
python scripts/convert_coco_to_yolo.py --images-root /path/to/PIDray/images --annotation-json /path/to/PIDray/annotations/hidden.json --output-dir dataset_yolo --split-source hidden --append
```

For PIDray-style splits, `easy` is treated as a proxy for `single_basic`, `hard` as a proxy for `complex_target`, and `hidden` as a proxy for `complex_with_non_target`. This is not the same as AIHub's composition labels, but it preserves the simple-to-difficult generalization question.

Make splits:

```bash
python scripts/make_splits.py --metadata dataset_yolo/metadata.csv --output-dir dataset_yolo/splits
```

Debug train:

```bash
python scripts/train_yolo.py --data-yaml dataset_yolo/splits/mixed/data.yaml --model yolo11n.pt --epochs 3 --imgsz 640 --batch 4 --device 0
```

Baseline train:

```bash
python scripts/train_yolo.py --data-yaml dataset_yolo/splits/mixed/data.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --device 0 --name baseline_mixed
```

Simple-to-complex train:

```bash
python scripts/train_yolo.py --data-yaml dataset_yolo/splits/simple_to_complex/data.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --device 0 --name baseline_simple_to_complex
```

Evaluate:

```bash
python scripts/evaluate_yolo.py --weights runs/detect/baseline_mixed/weights/best.pt --data-yaml dataset_yolo/splits/mixed/data.yaml --metadata dataset_yolo/splits/mixed/metadata_test.csv --split-name mixed
```

Compute CIGG:

```bash
python scripts/compute_cigg.py --eval-csv outputs/eval_results.csv
```

Make balanced training split:

```bash
python scripts/make_balanced_training_set.py --split-dir dataset_yolo/splits/mixed --metadata dataset_yolo/splits/mixed/metadata_train.csv --output-dir dataset_yolo/splits/balanced_mixed
```

Train balanced model:

```bash
python scripts/train_yolo.py --data-yaml dataset_yolo/splits/balanced_mixed/data.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --device 0 --name balanced_mixed
```

Extract object crops:

```bash
python scripts/extract_object_crops.py --metadata dataset_yolo/metadata.csv --output-dir object_crops
```

Train contrastive encoder:

```bash
python scripts/train_object_contrastive_encoder.py --crops-metadata object_crops/crops_metadata.csv --output-dir outputs/contrastive --epochs 30 --batch 64 --device 0
```

Analyze feature bias:

```bash
python scripts/analyze_feature_bias.py --embeddings outputs/contrastive/embeddings.npy --metadata outputs/contrastive/embeddings_metadata.csv --output-dir outputs/feature_bias
```

Visualize:

```bash
python scripts/visualize_results.py --metadata dataset_yolo/metadata.csv --eval-csv outputs/eval_results.csv --output-dir outputs/figures
```

Run the PIDray workflow in one command:

```bash
python scripts/run_pidray_pipeline.py --experiment-set core --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

On a Linux training server:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pidray_pipeline.py --hardware-preset rtx5090 --experiment-set balanced --device 0
```

Or use the bash wrapper:

```bash
chmod +x scripts/run_pidray_pipeline_linux.sh
./scripts/run_pidray_pipeline_linux.sh
```

Override defaults through environment variables:

```bash
EXPERIMENT_SET=full HARDWARE_PRESET=rtx5090 DEVICE=0 ./scripts/run_pidray_pipeline_linux.sh
```

The RTX 5090 preset uses `yolo11m.pt`, `imgsz=768`, `batch=8`, `epochs=100`, and `workers=8`. If CUDA runs out of memory, keep the preset and override only the batch size:

```bash
python scripts/run_pidray_pipeline.py --hardware-preset rtx5090 --experiment-set balanced --batch 4 --device 0
```

Quick smoke test:

```bash
python scripts/run_pidray_pipeline.py --debug --experiment-set mixed --device 0
```

If PIDray is already downloaded or Google Drive blocks automated download, point the runner at the existing folder:

```bash
python scripts/run_pidray_pipeline.py --skip-download --raw-dir /path/to/PIDray --experiment-set core --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

## Research Splits

`scripts/make_splits.py` creates:

- `mixed`: random split across composition types, stratified when possible.
- `simple_to_complex`: train on `single_basic` and `single_with_non_target`, test on complex compositions.
- `complex_with_non_target_holdout`: train on all other compositions, test on `complex_with_non_target`.
- `leave_out_*`: one split per known composition type.

`scripts/make_balanced_training_set.py` creates `balanced_mixed` by oversampling underrepresented composition types in the training subset while preserving validation and test subsets from `mixed` when available.

## CIGG

Composition-Invariance Generalization Gap:

```text
CIGG = mixed_test_mAP50 - complex_holdout_mAP50
```

Lower CIGG means better composition robustness. The scripts also compute recall and `mAP50_95` gaps when the required rows exist.

Expected result tables include:

- Baseline YOLO vs YOLO + composition-balanced training.
- Mixed vs simple-to-complex performance.
- Per-composition mAP and recall.
- Object-class accuracy vs composition leakage accuracy from crop embeddings.

## Outputs

Important generated paths:

- `outputs/dataset_stats.json`
- `outputs/dataset_stats.csv`
- `dataset_yolo/metadata.csv`
- `dataset_yolo/splits/*/data.yaml`
- `outputs/eval_results.csv`
- `outputs/per_class_eval_results.csv`
- `outputs/cigg_results.json`
- `outputs/cigg_results.csv`
- `outputs/figures/*.png`
- `outputs/contrastive/*`
- `outputs/feature_bias/*`

Generated datasets, crops, model weights, runs, and reports are ignored by Git. Do not commit original AIHub data.
