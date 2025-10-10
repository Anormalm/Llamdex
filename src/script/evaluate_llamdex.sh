#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT_DIR"

DATASETS=${DATASETS:-"titanic wine_quality bank_marketing nursery"}
SEEDS=${SEEDS:-"0 1"}
LAYERS=${LAYERS:-"0 10 20 30"}
BATCH_SIZE=${BATCH_SIZE:-8}
EXTRA_ARGS=${EXTRA_ARGS:-""}

log_dir=${LOG_DIR:-"logs/llamdex"}
mkdir -p "$log_dir"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    dataset_log_dir="$log_dir/$dataset"
    mkdir -p "$dataset_log_dir"
    for layer in $LAYERS; do
      run_name="exp_mistral_${dataset}_layer${layer}_seed${seed}"
      model_dir="model/llm/$run_name"
      model_path="$model_dir/model_final.pt"
      if [ ! -f "$model_path" ]; then
        echo "[evaluate_llamdex] skip dataset=$dataset layer=$layer seed=$seed (missing $model_path)" >&2
        continue
      fi

      echo "[evaluate_llamdex] dataset=$dataset layer=$layer seed=$seed"
      python src/evaluate/eval_domain_mistral.py \
        --dataset "$dataset" \
        --layer "$layer" \
        --batch_size "$BATCH_SIZE" \
        --model_state_dict_dir "$model_path" \
        $EXTRA_ARGS \
        2>&1 | tee "$dataset_log_dir/eval_${run_name}.log"
    done
  done
done
