#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT_DIR"

DATASETS=${DATASETS:-"bank_marketing titanic wine_quality nursery"}
EXTRA_ARGS=${EXTRA_ARGS:-""}

for dataset in $DATASETS; do
  echo "[prepare_data] dataset=$dataset"

  python src/preprocess/clean/clean_${dataset}.py $EXTRA_ARGS

  if [ -f "src/preprocess/syn/syn_${dataset}.py" ]; then
    python src/preprocess/syn/syn_${dataset}.py $EXTRA_ARGS
  fi

  if [ -f "src/preprocess/expert/${dataset}_mlp.py" ]; then
    python src/preprocess/expert/${dataset}_mlp.py $EXTRA_ARGS
  fi

  python src/preprocess/gentext/gentext_dataset.py --dataset "$dataset" $EXTRA_ARGS

done
