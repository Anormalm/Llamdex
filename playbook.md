# Remote Codex Playbook (Always-On Rules)

This file is the operational contract for any remote Codex session on this project.
Follow these rules at all times.

## 1) Environment Contract

- OS: Ubuntu 22.04.x
- Python: venv only
- No Docker assumptions
- No Conda assumptions

Use:

```bash
source /home/anormalm/Llamdex/.venv/bin/activate
```

## 2) Path Contract (Strict)

- **Code only**: `/home/anormalm/Llamdex`
- **Data/models/artifacts only**: `/disk1/lfhu`

Never store large files in repo paths under `/home/anormalm/Llamdex`.

Canonical roots:

- HF cache: `/disk1/lfhu/hf_cache`
- Data root: `/disk1/lfhu/data`
- Run outputs: `/disk1/lfhu/runs`

## 3) Project Priorities (Current)

Urgent workstreams:

1. Task 4: Grounded answer + rationale
2. Natural-language output tuning
3. Multiple SOTA baselines

Architecture direction:

- Qwen-first server backbone
- Frozen backbone + frozen experts
- Train connector/projector/adapters only
- Keep Plan-1++ injection controls (`layer_input/post_attn/pre_ffn/post_ffn`)

## 4) Mandatory Pre-Run Checks

Before any training/eval:

```bash
cd /home/anormalm/Llamdex
git status --short
python -V
python -c "import torch,transformers,accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)"
nvidia-smi
```

If `transformers` cannot load Qwen3.5 model type, upgrade:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -U transformers accelerate tokenizers huggingface_hub safetensors sentencepiece
```

## 5) Run Policy

- Run **one heavy job per GPU** unless explicitly parallelized.
- Avoid duplicate background jobs loading the same model/cache simultaneously.
- Use unbuffered logs and tee output to `/disk1/lfhu/runs/*.log`.

Example pattern:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/<runner>.py ... \
2>&1 | tee /disk1/lfhu/runs/<run_name>.log
```

## 6) Required CLI Path Overrides

Always pass explicit storage paths:

- model cache flag (`--server_models_path` or equivalent): `/disk1/lfhu/hf_cache`
- `--data_root /disk1/lfhu/data`
- output files to `/disk1/lfhu/runs/...`

Do not rely on default `./data` or `runs/` during remote execution.

## 7) Artifact and Logging Rules

Every run must produce:

1. machine-readable metrics (`.csv` and/or `.json`)
2. full log file (`.log`)
3. command record in markdown report

Status report location:

- `/disk1/lfhu/runs/status_quo_remote_YYYYMMDD.md`

Minimum report contents:

- commit hash
- environment versions
- exact commands
- result table (`task/model/dataset/metric/value/latency/status`)
- failures + root cause + fix
- next 72-hour plan

## 8) Task-Matrix Minimum (Do Not Skip)

When reporting progress, include:

1. single-image classification
2. yes/no QA (strict output)
3. population aggregation
4. grounded answer + rationale

Metrics:

- Accuracy/F1 for classification/yes-no
- MAE (+ calibration if available) for population
- rationale consistency + format compliance + non-empty rationale rate for grounded generation

## 9) Safety / Privacy Rules

- Never upload private raw data to server-side logs/artifacts.
- Only use sanitized evidence/description inputs on server paths.
- Do not print secrets/API keys in terminal logs.

## 10) Code Change Policy

- Keep changes minimal and scoped.
- Preserve backward compatibility unless a breaking change is explicitly requested.
- Do not remove legacy modules without explicit approval.
- Add/keep robust error traces for failed runs.

Before commit:

```bash
python -m py_compile <changed_python_files>
```

Commit message format:

- `<scope>: <what changed> (<why>)`

## 11) Git Discipline

Before push:

```bash
git add <scoped files>
git commit -m "<clear message>"
git push
```

After push, always report:

- branch name
- commit hash
- changed files

## 12) Fast Triage Rules

If run appears hung:

1. check `nvidia-smi` and process list
2. stop duplicate competing jobs
3. inspect stale HF lock files
4. rerun single process with tee logging

If OOM:

1. reduce batch size first
2. reduce model size second
3. use quantization/multi-GPU policy as configured
4. do not silently weaken evaluation protocol

---

## Session Checklist (Copy/Paste)

```bash
cd /home/anormalm/Llamdex
source /home/anormalm/Llamdex/.venv/bin/activate
git pull
python -m pip install -U transformers accelerate tokenizers huggingface_hub safetensors sentencepiece
python -c "import torch,transformers,accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)"
nvidia-smi
```

Then run with explicit `/disk1/lfhu/*` paths and log to `/disk1/lfhu/runs/*.log`.
