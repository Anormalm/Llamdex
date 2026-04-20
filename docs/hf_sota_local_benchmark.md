# HF SOTA Local Baseline Benchmark (Remote)

This runbook benchmarks Llamdex against local Hugging Face VLM baselines with outputs under `/disk1/lfhu/runs`.

## 1) Environment

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
python -V
```

## 2) Download Local VLM Packs

List curated packs:

```bash
python scripts/download_hf_vlm_baselines.py --list_presets
```

Budget pack (recommended first):

```bash
PYTHONPATH=/home/anormalm/Llamdex python scripts/download_hf_vlm_baselines.py \
  --cache_dir /disk1/lfhu/hf_cache \
  --preset vlm_budget_2026q2 \
  --baseline_config /home/anormalm/Llamdex/conf/local_arch_baselines.hf_sota.dtd.remote.json \
  --include_backbone \
  --include_captioner \
  --continue_on_error
```

Full SOTA pack:

```bash
PYTHONPATH=/home/anormalm/Llamdex python scripts/download_hf_vlm_baselines.py \
  --cache_dir /disk1/lfhu/hf_cache \
  --preset vlm_sota_2026q2 \
  --continue_on_error
```

## 3) Run Unified Benchmark (DTD)

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_unified_baseline_benchmark.py \
  --config /home/anormalm/Llamdex/conf/unified_benchmark.hf_sota.remote.json \
2>&1 | tee /disk1/lfhu/runs/unified_benchmark.hf_sota.remote.log
```

Outputs:
- `/disk1/lfhu/runs/unified_benchmark.hf_sota.remote.csv`
- `/disk1/lfhu/runs/unified_benchmark.hf_sota.remote.json`

## 4) Run Local Baseline Suite Only (Oxford Pet)

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_baseline_benchmark_suite.py \
  --config /home/anormalm/Llamdex/conf/local_arch_baselines.hf_sota.oxford_pet.remote.json \
  --report_md /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.md \
2>&1 | tee /disk1/lfhu/runs/local_arch_baselines.hf_sota.oxford_pet.remote.log
```

## 5) Compare vs Llamdex

Unified rows include:
- `benchmark_family=llamdex` (our model/task-matrix runs)
- `benchmark_family=architecture` (local baseline suite rows)
- `benchmark_family=api` (API baselines)

Sort by `task`, `dataset`, and `metric_value` to inspect gaps.
