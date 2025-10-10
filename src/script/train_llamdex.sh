#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT_DIR"

DATASETS=${DATASETS:-"titanic wine_quality bank_marketing nursery"}
SEEDS=${SEEDS:-"0 1"}
LAYERS=${LAYERS:-"0 10 20 30"}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-conf/default_config.yaml}
EXTRA_ARGS=${EXTRA_ARGS:-""}

log_dir=${LOG_DIR:-"logs/llamdex"}
mkdir -p "$log_dir"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    dataset_log_dir="$log_dir/$dataset"
    mkdir -p "$dataset_log_dir"
    for layer in $LAYERS; do
      run_name="exp_mistral_${dataset}_layer${layer}_seed${seed}"
      save_dir="model/llm/$run_name"
      mkdir -p "$save_dir"

      echo "[train_llamdex] dataset=$dataset layer=$layer seed=$seed"
      accelerate launch --config_file "$ACCELERATE_CONFIG" \
        src/train/train_domain_mistral.py \
        --dataset "$dataset" \
        --no_checkpoint \
        --save_dir "$save_dir" \
        --batch_size 32 \
        --gradient_accumulation_steps 8 \
        --layer "$layer" \
        --num_epochs_enc 30 \
        --num_epochs_dec 10 \
        --learning_rate_enc 5e-5 \
        --learning_rate_dec 5e-5 \
        $EXTRA_ARGS \
        2>&1 | tee "$dataset_log_dir/${run_name}.log"
    done
  done
done
