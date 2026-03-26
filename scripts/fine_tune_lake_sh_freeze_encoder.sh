#!/bin/bash
# Fine-tune SAM3 on Lake SH dataset using pre-split COCO train/val annotations
# Usage:
#   bash scripts/fine_tune_lake_sh.sh
# Optional env vars:
#   PROJECT_ROOT=/home/ucas_yw/algorithm/sam3-main
#   PYTHON_BIN=/home/ucas_yw/anaconda3/envs/sam3/bin/python
#   CONFIG_NAME=configs/lake_sh_fine_tune_freeze_encoder.yaml
#   NUM_GPUS=1
#   USE_CLUSTER=0
#   DATASET_ROOT=/home/ucas_yw/algorithm/sam3-main/data/lake_sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ucas_yw/algorithm/sam3-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/ucas_yw/anaconda3/envs/sam3/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-configs/lake_sh_fine_tune_freeze_encoder.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_CLUSTER="${USE_CLUSTER:-0}"
DATASET_ROOT="${DATASET_ROOT:-/home/ucas_yw/algorithm/sam3-main/data/lake_sh}"

TRAIN_JSON="${DATASET_ROOT}/annotations/instances_train.json"
VAL_JSON="${DATASET_ROOT}/annotations/instances_val.json"
IMAGES_DIR="${DATASET_ROOT}/images"

cd "${PROJECT_ROOT}"

echo "Starting SAM3 fine-tuning..."
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_NAME}"
echo "Dataset root: ${DATASET_ROOT}"

# Basic data checks (explicitly ensure we use your already-split train/val data)
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not found: ${PYTHON_BIN}"; exit 1; }
[[ -f "${TRAIN_JSON}" ]] || { echo "[ERROR] Missing train annotation: ${TRAIN_JSON}"; exit 1; }
[[ -f "${VAL_JSON}" ]] || { echo "[ERROR] Missing val annotation: ${VAL_JSON}"; exit 1; }
[[ -d "${IMAGES_DIR}" ]] || { echo "[ERROR] Missing images dir: ${IMAGES_DIR}"; exit 1; }

echo "Verifying train/val split statistics..."
DATASET_ROOT="${DATASET_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

dataset_root = Path(os.environ["DATASET_ROOT"])
train_json = dataset_root / "annotations" / "instances_train.json"
val_json = dataset_root / "annotations" / "instances_val.json"

train = json.loads(train_json.read_text())
val = json.loads(val_json.read_text())

train_files = {x["file_name"] for x in train.get("images", [])}
val_files = {x["file_name"] for x in val.get("images", [])}

print(f"  train images: {len(train.get('images', []))}, annotations: {len(train.get('annotations', []))}")
print(f"  val   images: {len(val.get('images', []))}, annotations: {len(val.get('annotations', []))}")
print(f"  overlap file names (train∩val): {len(train_files & val_files)}")
print(f"  categories: {train.get('categories', [])}")
PY

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:."

# Run training with local single-node mode by default
"${PYTHON_BIN}" sam3/train/train.py \
    -c "${CONFIG_NAME}" \
    --use-cluster "${USE_CLUSTER}" \
    --num-gpus "${NUM_GPUS}"

echo "Training command finished."
