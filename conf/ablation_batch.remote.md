# Ablation Batch (Remote)

## Policy ablation (overwrite vs router_parallel)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_policy_ablation.py \
  --server_models_path /disk1/lfhu/hf_cache \
  --model_name Qwen/Qwen3.5-9B \
  --data_root /disk1/lfhu/data \
  --run_root /disk1/lfhu/runs \
  --device cuda \
  --seed 42 \
  --train_steps 40 \
  --max_train_samples 384 \
  --max_eval_samples 192 \
  --batch_size 8 \
  2>&1 | tee /disk1/lfhu/runs/policy_ablation.remote.log
```

Outputs:
- /disk1/lfhu/runs/policy_ablation.csv
- /disk1/lfhu/runs/policy_ablation.json
- /disk1/lfhu/runs/status_report.md

## Believability task matrix (tuned uploaded bundle)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_task_matrix_eval.py \
  --config conf/task_matrix_dtd_post_attn_router_upload.tuned_loaded.believability.remote.json \
  2>&1 | tee /disk1/lfhu/runs/task_matrix_dtd_post_attn_router_upload.tuned_loaded.believability.remote.log
```

Outputs:
- /disk1/lfhu/runs/task_matrix_dtd_post_attn_router_upload.tuned_loaded.believability.remote.csv
- /disk1/lfhu/runs/task_matrix_dtd_post_attn_router_upload.tuned_loaded.believability.remote.json
- /disk1/lfhu/runs/task_matrix_dtd_post_attn_router_upload.tuned_loaded.believability.remote.log

## API baselines (expanded v2)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_api_baseline_suite.py \
  --config conf/api_baselines.remote.expanded.v2.json \
  2>&1 | tee /disk1/lfhu/runs/api_baselines.remote.expanded.v2.log
```

Outputs:
- /disk1/lfhu/runs/api_baselines.remote.expanded.v2.csv
- /disk1/lfhu/runs/api_baselines.remote.expanded.v2.json
- /disk1/lfhu/runs/api_baselines.remote.expanded.v2.summary.csv
- /disk1/lfhu/runs/api_baselines.remote.expanded.v2.summary.json
- /disk1/lfhu/runs/api_baselines.remote.expanded.v2.log

## Download HF local VLM baselines

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/download_hf_vlm_baselines.py \
  --cache_dir /disk1/lfhu/hf_cache \
  --baseline_config conf/local_arch_baselines.hf_sota.dtd.remote.json \
  --include_backbone \
  --include_captioner
```

## Local HF SOTA baseline benchmark (DTD)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_baseline_benchmark_suite.py \
  --config conf/local_arch_baselines.hf_sota.dtd.remote.json \
  --report_md /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.md \
  2>&1 | tee /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.log
```

Outputs:
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.csv
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.json
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.md
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.dtd.remote.log

## Local HF SOTA baseline benchmark (Oxford Pet)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_baseline_benchmark_suite.py \
  --config conf/local_arch_baselines.hf_sota.oxford_pet.remote.json \
  --report_md /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.md \
  2>&1 | tee /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.log
```

Outputs:
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.csv
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.json
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.md
- /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.log

## Unified benchmark packet (task matrix + local HF SOTA + API expanded)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_unified_baseline_benchmark.py \
  --config conf/unified_benchmark.hf_sota.remote.json \
  2>&1 | tee /disk1/lfhu/runs/unified_benchmark.hf_sota.remote.log
```

Outputs:
- /disk1/lfhu/runs/unified_benchmark.hf_sota.remote.csv
- /disk1/lfhu/runs/unified_benchmark.hf_sota.remote.json
- /disk1/lfhu/runs/unified_benchmark.hf_sota.remote.log
