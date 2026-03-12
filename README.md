# Llamdex: Model-based Large Language Model Customization as Service

This is the official repository of the EMNLP'25 (Main) paper: [Llamdex: Model-based Large Language Model Customization as Service](https://aclanthology.org/2025.emnlp-main.248.pdf).

Llamdex provides a complete pipeline for customizing large language models to structured data domains. This release bundles training code, preprocessing utilities, analysis scripts, and reproducible baseline implementations so you can reproduce our experiments or adapt the workflow to new datasets.

## Project Overview

This repository currently has two production tracks:

1. `Tabular Llamdex` (original EMNLP'25 system): schema-aware LLM customization for structured/private tabular data.
2. `Multimodal Plan-1` (`src/multimodal/`): connector-only evidence injection for vision/text/population reasoning with frozen backbone LLMs.

Plan-1 core interface:

`EvidenceSource -> z -> EvidenceProjector -> token overwrite at layer k -> LLM reasoning`

Production default (privacy-first):
- `evidence_source=text` for Plan-1 train/eval scripts.
- Raw image/video can stay local; server consumes only description-derived evidence.
- Vision path remains available for benchmarking and ablations (`--evidence_source vision`).

What stays frozen:
- base LLM backbone
- client expert models (vision encoder / text encoder)

What is trainable:
- evidence builders/projections needed to produce `z`
- `EvidenceProjector` and injection connector parameters

Main entrypoints:
- Train/eval Plan-1: `scripts/train_plan1.py`, `scripts/eval_plan1.py`
- API baseline suite: `scripts/run_api_baseline_suite.py`
- Local baseline suite: `scripts/run_baseline_benchmark_suite.py`
- Own-vs-API merge table: `scripts/compare_own_vs_api.py`

## Repository Layout

```
baseline/       Reference implementations for comparison baselines
  dp-opt/       Differentially private OPT baseline with training + sweeps
  mink-plus-plus/  External MINK++ implementation (git submodule)
  MICO/            Model-based inference baseline (git submodule)
  tablediffusion/  Differentially private diffusion models for tables
conf/           YAML configuration presets for end-to-end runs
src/            Llamdex source code
  analysis/     Plotting, reporting, and DP evaluation utilities
  dataset/      Downloaders, metadata, and synthetic data recipes
  evaluate/     Model evaluation entry points
  fine_tune/    LoRA fine-tuning utilities for Llama/Mistral backbones
  model/        Core model definitions and expert routing modules
  preprocess/   Data cleaning, feature generation, and text synthesis scripts
  script/       Orchestrated experiment runners for training and evaluation
  synthesis/    Synthetic data generation pipelines
  train/        Training entry scripts for different regimes
```

## Installation

1. Create a Python 3.10+ environment.
2. Install the core dependencies:

```bash
pip install -r baseline/requirements.txt
pip install accelerate deepspeed peft sentencepiece torch torchvision torchaudio
pip install pandas scikit-learn tqdm transformers xgboost gorilla tensorboard
```

> **GPU support**: use the PyTorch wheels that match your CUDA/ROCm stack as documented on [pytorch.org](https://pytorch.org/get-started/locally/).

## Data Preparation

Download raw datasets using the scripts in `src/dataset`:

```bash
bash src/dataset/bank_marketing/download.sh
bash src/dataset/titanic/download.sh
bash src/dataset/wine_quality/download.sh
bash src/dataset/nursery/download.sh
```

Each dataset follows the same preprocessing flow:

```bash
dataset=bank_marketing
python src/preprocess/clean/clean_${dataset}.py
python src/preprocess/syn/syn_${dataset}.py
python src/preprocess/expert/${dataset}_mlp.py
python src/preprocess/gentext/gentext_dataset.py --dataset ${dataset}
```

Synthetic generation recipes (`src/preprocess/syn/*.py`) and expert models (`src/preprocess/expert/*.py`) can be customized per domain. To automate the pipeline for multiple datasets, run:

```bash
DATASETS="bank_marketing titanic" bash src/script/prepare_data.sh
```

## Training & Evaluation

The release ships lean entry points in `src/script/` that run sequentially without device-specific scheduling. Override DATASETS, SEEDS, or LAYERS to focus on particular runs:

```bash
bash src/script/train_llamdex.sh        # Train Llamdex models
bash src/script/evaluate_llamdex.sh     # Evaluate saved checkpoints
```

Adjust hyperparameters via environment variables (see the script headers) or edit the underlying trainers in `src/train/`.

## Analysis & Reporting

Use `src/analysis/` to regenerate plots and tables, including DP trade-off curves (`plot_dp.py`), expert ablations (`plot_ablation_expert_weight.py`), and membership inference studies (`mia_mico_style.py`). The utilities read logs produced by the runner scripts and emit publication-ready figures and LaTeX tables.

## Llamdex-IMG: Image Classification Extension

Llamdex-IMG extends Llamdex to image classification tasks using a late-fusion design. The vision expert (frozen ResNet-18) processes images once per sample, and its output is mapped via a trainable decoder to token embeddings that are injected into the LLM at a specified layer.

### Quick Start

1. **Install additional dependencies** (if not already installed):
```bash
pip install torchvision
```

2. **Train on CIFAR-10**:
```bash
python scripts/train_img.py \
    --mistral_models_path model/llm \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_tokens 10 \
    --layer 0 \
    --num_epochs 3 \
    --batch_size 32
```

3. **Evaluate**:
```bash
# Llamdex-IMG
python scripts/eval_img.py \
    --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt

# Baselines
python scripts/eval_img.py --baseline vision_only  # Vision-only classifier
python scripts/eval_img.py --baseline llm_only     # LLM-only (no injection)
python scripts/eval_img.py --baseline prompt       # Prompt baseline
```

### Ablation Studies

The training and evaluation scripts support ablation studies via command-line flags:

- `--layer`: Insertion layer k (default: 0)
- `--num_tokens`: Number of injected tokens (1/4/8/16, default: 10)
- `--vision_output`: Expert output mode (`logits` or `embedding`, default: `logits`)
- `--alpha`: Scaling factor for injected embeddings (default: 1.0)

Example:
```bash
python scripts/train_img.py --layer 5 --num_tokens 16 --vision_output embedding --alpha 0.5
```

### Architecture

- **Frozen Vision Expert**: ResNet-18 pretrained on ImageNet, adapted for CIFAR-10 (32x32 images)
- **Trainable Decoder**: Maps expert output (logits or embeddings) to `(num_tokens × hidden_size)` token embeddings
- **Injection Mechanism**: Reuses existing Llamdex reserved token slots with Gaussian padding + LayerNorm
- **Training**: Only the decoder is trainable; base LLM and vision expert remain frozen

### Smoke Test

Run a quick smoke test to verify the setup:
```bash
python tests/test_img_smoke.py
```

This runs 10 training steps on a tiny subset without requiring GPU.

### Plan-1 Upgrade for IMG Scripts

The IMG pipeline now supports semantic evidence injection with a unified flow:

`EvidenceSource -> evidence vector z -> EvidenceProjector -> token injection -> LLM reasoning`

This is implemented in-place in:
- `scripts/train_img.py`
- `scripts/eval_img.py`
- `src/vision/vision_domain_expert.py`
- `src/vision/evidence_builder.py`

New CLI options:
- `--evidence_dim` (default: `512`)
- `--evidence_source` (`vision` or `text`, default: `vision`)
- `--task` (`single`, `yesno`, `population`, default: `single`)

Compatibility note:
- Existing commands still work unchanged.
- Default `vision + single` behavior preserves the prior classification path.

#### Examples

Vision evidence, single-label classification (existing behavior):
```bash
python scripts/train_img.py --evidence_source vision --task single
python scripts/eval_img.py --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt --evidence_source vision --task single
```

Text evidence (description-only input), single-label classification:
```bash
python scripts/train_img.py --evidence_source text --task single --text_encoder_model distilroberta-base
python scripts/eval_img.py --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt --evidence_source text --task single
```

Yes/No query mode:
```bash
python scripts/train_img.py --task yesno --yesno_class truck
python scripts/eval_img.py --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt --task yesno --yesno_class truck
```

Population query mode:
```bash
python scripts/train_img.py --task population --population_class airplane
python scripts/eval_img.py --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt --task population --population_class airplane
```

Forward-pass validation (no training):
```bash
python scripts/test_plan1_upgrade.py
```

#### Meaningful Benchmark v1 (3 harder tasks)

We added a benchmark runner for three more meaningful task tracks:
- `single_yesno_vision`: semantic yes/no image QA (`qa_type=yesno`)
- `single_label_text`: label prediction from text evidence path (`evidence_source=text`)
- `population_fraction_vision`: grouped population fraction reasoning

Run:
```bash
python scripts/benchmark_meaningful_v1.py \
  --mistral_models_path runs/hf_cache_tiny \
  --model_name hf-internal-testing/tiny-random-MistralForCausalLM \
  --dataset_name dtd \
  --data_root ./data \
  --expert_checkpoint /path/to/dtd_resnet18_best.pt \
  --connectors_path runs/plan1_iter_lr2e4/best_connectors.pt \
  --batch_size 16 \
  --max_eval_samples 1000 \
  --device cuda
```

Recommended now:
- Use non-CIFAR datasets first (default is `dtd` in `train_plan1.py` / `eval_plan1.py`).
- Train a dataset-matched vision expert checkpoint before comparing `vision_only` vs `injection`.

Train a DTD vision expert checkpoint:
```bash
python scripts/train_vision_expert.py \
  --dataset_name dtd \
  --data_root ./data \
  --out_path runs/experts/dtd_resnet18_best.pt \
  --epochs 5 \
  --batch_size 128
```

Outputs:
- `runs/benchmark_meaningful_v1.csv`
- `runs/benchmark_meaningful_v1.json`
- `runs/benchmark_meaningful_v1_dtd.csv`
- `runs/benchmark_meaningful_v1_dtd.json`

Historical run (2026-02-19, CIFAR setup):

| Task | Baseline | Metric(s) |
|---|---|---|
| `single_yesno_vision` | `vision_only` | `accuracy=0.894` |
| `single_yesno_vision` | `llm_only` | `accuracy=0.527` |
| `single_yesno_vision` | `text_prompt` | `accuracy=0.527` |
| `single_yesno_vision` | `injection` | `accuracy=0.000` |
| `single_label_text` | `llm_only` | `accuracy=0.103` |
| `single_label_text` | `injection` | `accuracy=0.000` |
| `population_fraction_vision` | `vision_only` | `bin_accuracy=0.801`, `mae=0.0237` |
| `population_fraction_vision` | `llm_only` | `bin_accuracy=0.000`, `mae=0.000` |
| `population_fraction_vision` | `text_prompt` | `bin_accuracy=0.000`, `mae=0.000` |
| `population_fraction_vision` | `injection` | `bin_accuracy=0.000`, `mae=0.000` |

Important caveat:
- `runs/plan1_iter_lr2e4/best_connectors.pt` was trained for the original single-label setup.
- For meaningful v1 tasks, dedicated connector training per task is required; otherwise injection can collapse due to answer-format mismatch and task shift.

Latest run (2026-02-26, DTD setup):

| Task | Baseline | Metric(s) |
|---|---|---|
| `single_yesno_vision` | `vision_only` | `accuracy=0.505` |
| `single_yesno_vision` | `llm_only` | `accuracy=0.495` |
| `single_yesno_vision` | `text_prompt` | `accuracy=0.495` |
| `single_yesno_vision` | `injection` | `accuracy=0.505` |
| `single_yesno_text` | `llm_only` | `accuracy=0.495` |
| `single_yesno_text` | `injection` | `accuracy=0.505` |
| `population_fraction_vision` | `vision_only` | `bin_accuracy=0.850`, `mae=0.0265` |
| `population_fraction_vision` | `llm_only` | `bin_accuracy=0.000`, `mae=0.000` |
| `population_fraction_vision` | `text_prompt` | `bin_accuracy=0.000`, `mae=0.000` |
| `population_fraction_vision` | `injection` | `bin_accuracy=0.8475`, `mae=0.0170` |

Meaningful Benchmark v2 (2026-02-26, DTD setup; compositional + stable output):

| Task | Baseline | Metric(s) |
|---|---|---|
| `single_yesno_set2_vision` | `vision_only` | `accuracy=0.5000` |
| `single_yesno_set2_vision` | `llm_only` | `accuracy=0.5225` |
| `single_yesno_set2_vision` | `text_prompt` | `accuracy=0.5225` |
| `single_yesno_set2_vision` | `injection` | `accuracy=0.4775` |
| `single_label_code_vision` | `vision_only` | `accuracy=0.0450` |
| `single_label_code_vision` | `llm_only` | `accuracy=0.0000` |
| `single_label_code_vision` | `text_prompt` | `accuracy=0.0000` |
| `single_label_code_vision` | `injection` | `accuracy=0.4250` |
| `population_fraction_vision` | `vision_only` | `bin_accuracy=0.8600`, `mae=0.0256` |
| `population_fraction_vision` | `llm_only` | `bin_accuracy=0.0000`, `mae=0.0000` |
| `population_fraction_vision` | `text_prompt` | `bin_accuracy=0.0000`, `mae=0.0000` |
| `population_fraction_vision` | `injection` | `bin_accuracy=0.8475`, `mae=0.0170` |

Artifacts:
- `runs/benchmark_meaningful_v2_dtd_afterfix.csv`
- `runs/benchmark_plan1_upgrade_log_2026-02-26.md`

## Plan 1: Semantic Evidence Injection

Plan 1 adds a unified multimodal evidence path:
- `evidence vector z in R^D` built from either vision evidence or text descriptions
- reserved-token overwrite injection at one chosen LLM layer `k`
- frozen base LLM + frozen client expert, trainable connectors only

### Adapters + Injection Control Upgrade

Plan-1 now supports Houlsby-style bottleneck adapters and explicit injection-point control.

New controls:
- `--use_adapters` (`0/1`, default `1`)
- `--adapter_bottleneck` (default `64`)
- `--adapter_dropout` (default `0.0`)
- `--adapter_activation` (`gelu|relu`, default `gelu`)
- `--tune_layernorm` (`0/1`, default `0`)
- `--inject_location` (`layer_input|post_attn|pre_ffn|post_ffn`, default `post_attn`)

Trainable scope when adapters enabled:
- EvidenceBuilder + EvidenceProjector (existing connector params)
- Adapter params (`attn_adapter`, `ffn_adapter`)
- Optional LN params if `--tune_layernorm 1`

Base LLM and expert encoders remain frozen.

### Current Architecture (Detailed)

At runtime, Plan 1 follows a strict connector-only adaptation path:

1. Input construction:
- `single_image` tasks build an instruction prompt and carry either image tensors or text descriptions.
- `population` tasks aggregate a set of images and build summary statistics.

2. Evidence builder stage:
- Vision path: `VisionEvidenceBuilder` runs a frozen vision expert and maps logits/embeddings to `z` (`evidence_dim`).
- Text path: `TextEvidenceBuilder` encodes descriptions into `z`.
- Population path: `PopulationStatsEvidenceBuilder` converts class-probability summaries to `z`.

3. Projection stage:
- `EvidenceProjector` maps `z -> (num_tokens, hidden_size)` and applies optional scaling `alpha`.

4. Injection stage:
- `SemanticEvidenceDomainExpert` is attached to one decoder layer (`layer_to_add`).
- On forward pass, projected expert tokens overwrite reserved token slots at that layer.
- Base LLM parameters remain frozen; only evidence-builder/projector connector parameters are trainable.

5. Training / evaluation behavior:
- Training optimizes next-token loss over constrained QA targets.
- Baselines:
  - `vision_only`: direct expert prediction, no LLM reasoning.
  - `llm_only`: no expert injection.
  - `text_prompt`: expert output appended to prompt text, no injection.

Implementation anchors:
- `src/multimodal/trainers/plan1_trainer.py`
- `src/multimodal/eval/plan1_eval.py`
- `src/multimodal/evidence/*`
- `src/multimodal/injection/*`

### New package

`src/multimodal/`:
- `experts/`: frozen client expert loaders (`VisionClassifierExpert`, `VisionEmbeddingExpert`)
- `evidence/`: `VisionEvidenceBuilder`, `TextEvidenceBuilder`, `PopulationStatsEvidenceBuilder`
- `injection/`: `EvidenceProjector`, `SemanticEvidenceDomainExpert`
- `data/`: CIFAR QA and population aggregation datasets
- `tasks/`: prompt templates and answer parsing
- `trainers/`, `eval/`: train/eval entry logic
- `evidence/diffusion_builder.py`: Plan 2 placeholder (`NotImplementedError`)

### Train / Eval CLI

Description-first (recommended, privacy-first):
```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_text_privacy \
  --dataset_name dtd \
  --evidence_source text \
  --description_file data/dtd_descriptions_train.jsonl \
  --qa_type label \
  --use_adapters 1 \
  --adapter_bottleneck 64 \
  --inject_location post_attn
```

```bash
python scripts/eval_plan1.py \
  --dataset_name dtd \
  --evidence_source text \
  --description_file data/dtd_descriptions_test.jsonl \
  --qa_type label \
  --baseline injection \
  --connectors_path runs/plan1_text_privacy/best_connectors.pt
```

Supported description file formats (`--description_file`):
- `jsonl`: one object per line, with `index` and `description`
- `csv`: columns `index,description`
- `json`: either `{ "0": "...", "1": "..." }` or list format

Example JSONL row:
```json
{"index": 42, "description": "Irregular spiculated mass in upper outer quadrant, size 12 mm, increased from prior."}
```

Single-image label-only, vision evidence:
```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_label_vision \
  --task_family single_image \
  --evidence_source vision \
  --qa_type label \
  --expert_kind classifier \
  --expert_output_mode logits \
  --num_tokens 4 \
  --layer 0
```

Train with adapters + recommended post-attention injection:
```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_adapt_postattn \
  --task_family single_image \
  --evidence_source vision \
  --qa_type yesno \
  --use_adapters 1 \
  --adapter_bottleneck 64 \
  --inject_location post_attn \
  --layer 0 \
  --num_tokens 8
```

```bash
python scripts/eval_plan1.py \
  --task_family single_image \
  --evidence_source vision \
  --qa_type label \
  --baseline injection \
  --connectors_path runs/plan1_label_vision/best_connectors.pt
```

Hardening switches (recommended):
- `--model_name auto`: auto-select strongest cached backbone that fits local hardware.
- `--enforce_checkpoint_compat 1` (eval): fail fast on checkpoint/task mismatch (prevents silent output collapse).
- `--use_adapters 1 --inject_location post_attn`: stable default for stronger tasks.

Description-only mode:
```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_label_text \
  --task_family single_image \
  --evidence_source text \
  --qa_type label \
  --text_encoder_model_id distilroberta-base
```

```bash
python scripts/eval_plan1.py \
  --task_family single_image \
  --evidence_source text \
  --qa_type label \
  --baseline injection \
  --connectors_path runs/plan1_label_text/best_connectors.pt
```

Population aggregation mode (with optional summary-level noise):
```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_population_sigma0 \
  --task_family population \
  --evidence_source vision \
  --population_output_mode integer \
  --population_sigma 0.0
```

```bash
python scripts/eval_plan1.py \
  --task_family population \
  --evidence_source vision \
  --population_output_mode integer \
  --population_sigma 0.0 \
  --baseline injection \
  --connectors_path runs/plan1_population_sigma0/best_connectors.pt
```

```bash
python scripts/train_plan1.py \
  --run_dir runs/plan1_population_sigma01 \
  --task_family population \
  --evidence_source vision \
  --population_output_mode integer \
  --population_sigma 0.1
```

### Baselines

`scripts/eval_plan1.py --baseline ...` supports:
- `injection`
- `vision_only`
- `llm_only`
- `text_prompt` (expert output appended in prompt text)

### Ablation runner

```bash
python scripts/run_plan1_ablation.py \
  --tokens_grid 1 4 8 16 \
  --layers 0 8 16 24 \
  --inject_locations post_attn pre_ffn post_ffn \
  --use_adapters 1 \
  --adapter_bottlenecks 32 64 128 \
  --output_csv runs/plan1_ablation_summary.csv
```

`run_plan1_ablation.py` now writes parsed `accuracy` and also a best-row file:
- `<output_csv>.best.csv`

### Checkpoint-backed adapter/injection ablation (2026-03-05)

To avoid random-head artifacts, the runs below use dataset-matched expert checkpoints:
- DTD expert: `runs/experts/dtd_resnet18_best.pt`
- Oxford-IIIT Pet expert: `runs/experts/oxford_pet_resnet18_best.pt`

Settings:
- model: `hf-internal-testing/tiny-random-MistralForCausalLM`
- adapters: `use_adapters=1`, `adapter_bottleneck=64`, `tune_layernorm=0`
- grid: `num_tokens in {4,8,16}`, `inject_location in {post_attn, pre_ffn, post_ffn}`
- budget: `max_train_samples=256`, `max_eval_samples=256`

Results summary:

| Dataset | Best Config(s) | Best Accuracy | Pattern |
|---|---|---:|---|
| DTD | `num_tokens=8`, `post_attn` or `post_ffn` | 0.6563 | `pre_ffn` collapses (0.0039-0.0313) |
| Oxford-IIIT Pet | `num_tokens=4`, `post_attn` or `post_ffn` | 0.7461 | `pre_ffn` collapses (0.0313-0.0391) |

Artifacts:
- `runs/plan1_adapter_inject_ablation_dtd_stable_ckpt.csv`
- `runs/plan1_adapter_inject_ablation_oxford_stable_ckpt.csv`

### Evidence quality diagnostics

Before blaming injection or backbone, check whether evidence vectors `z` are separable:

```bash
python scripts/analyze_evidence_quality.py \
  --dataset_name oxford_pet \
  --evidence_source vision \
  --evidence_dim 256 \
  --expert_kind classifier \
  --expert_output_mode logits \
  --expert_checkpoint runs/experts/oxford_pet_resnet18_best.pt \
  --connectors_path runs/plan1_adapter_inject_ablation_oxford_ckpt/tokens4_layer0_post_attn_adapt1_b64_ln0/last_connectors.pt \
  --max_train_samples 512 \
  --max_eval_samples 256 \
  --out_csv runs/evidence_quality_oxford_probe.csv \
  --out_json runs/evidence_quality_oxford_probe.json
```

Reported metrics:
- `probe_acc`: linear-probe accuracy on `z`
- `centroid_margin`: average pairwise class-centroid distance

Practical default for harder tasks:
- `--use_adapters 1`
- `--adapter_bottleneck 64`
- `--inject_location post_attn` (or `post_ffn` if preferred)

### Smoke tests

```bash
python -m pytest -q tests/test_plan1_smoke.py tests/test_tabular_pipeline_smoke.py
```

### Latest Benchmarks (2026-02-14)

Evaluation setup:
- Task: `single_image`, `qa_type=label`
- Dataset: CIFAR-10 (`max_eval_samples=1000`)
- Model: `hf-internal-testing/tiny-random-MistralForCausalLM`
- Expert checkpoint: `runs/experts/cifar10_resnet18_best.pt`
- Device: `cuda`
- Batch size: `16`

| Case | Accuracy | Wall Time (s) |
|---|---:|---:|
| `vision_only` | 0.821 | 28.19 |
| `injection_old` (`runs/plan1_meaningful_vision/best_connectors.pt`) | 0.811 | 18.64 |
| `injection_iter` (`runs/plan1_iter_lr2e4/best_connectors.pt`) | 0.839 | 11.22 |
| `llm_only_raw` (`constrain_llm_only_outputs=0`) | 0.000 | 10.63 |
| `llm_only_constrained` (`constrain_llm_only_outputs=1`) | 0.103 | 10.55 |
| `text_prompt_raw` (`constrain_llm_only_outputs=0`) | 0.000 | 13.68 |
| `text_prompt_constrained` (`constrain_llm_only_outputs=1`) | 0.103 | 13.63 |

Notes:
- For tiny random LLM backbones, constrained decoding prevents punctuation collapse in `llm_only`/`text_prompt` and forces valid task-token outputs.
- Benchmark artifact CSV: `runs/benchmark_plan1_2026-02-14.csv`
- Iteration run: `runs/plan1_iter_lr2e4` with `best_eval_acc=0.839`

## API Baseline Suite (External SOTA-style Comparison)

The API suite evaluates realistic external baselines under the same constrained answer protocol.

Core files:
- Runner: `scripts/run_api_baseline_suite.py`
- Implementation: `src/multimodal/baselines/api_suite.py`
- Strong config: `conf/api_baseline_sota.strong.json`

Supported modes:
- `api_frozen_vlm` (direct image -> answer)
- `api_two_stage` (image -> caption -> answer)
- `api_llm_only` (floor baseline)
- `api_rag` (text retrieval baseline)

Runner features:
- deterministic repeats with seed control
- aggregate mean/std/CI95 summaries
- per-model endpoint and API key override (for multi-provider serving)
- retry/backoff for transient API failures

Run:
```bash
python scripts/run_api_baseline_suite.py --config conf/api_baseline_sota.strong.json
```

Artifacts:
- detailed rows: `runs/api_baseline_suite_strong.csv`, `runs/api_baseline_suite_strong.json`
- summary rows: `runs/api_baseline_suite_strong.summary.csv`, `runs/api_baseline_suite_strong.summary.json`

## Own Model vs API Leaderboard

Run local (own-model) baseline suite and merge with API summary:

```bash
python scripts/run_baseline_benchmark_suite.py --config conf/baseline_benchmark.expanded.json
python scripts/compare_own_vs_api.py \
  --local_csv runs/baseline_suite_expanded.csv \
  --api_summary_csv runs/api_baseline_suite_strong.summary.csv \
  --max_failure_rate 0.0 \
  --out_csv runs/own_vs_api_leaderboard.csv
```

Merged leaderboard artifact:
- `runs/own_vs_api_leaderboard.csv`

## Current Status Snapshot (2026-03-11)

This is the current project status for the multimodal Plan-1 track.

### Architecture and training status

Current runtime path:

`EvidenceSource -> z -> EvidenceProjector -> reserved-token overwrite at layer k -> frozen LLM reasoning`

Current implementation status:
- Frozen backbone LLM + frozen client experts are preserved.
- Trainable scope: evidence connector modules, projector, and optional Houlsby adapters.
- Explicit injection-point control is implemented:
  - `layer_input | post_attn | pre_ffn | post_ffn`
- Checkpoint compatibility guard is implemented in eval (`--enforce_checkpoint_compat 1`) to block mismatched task/checkpoint runs that previously caused collapsed metrics.
- `llm_only` scoring for yes/no baselines is semantic (not strict token-id only), so floor baselines are meaningful.

### Local matched-protocol benchmark (repeat-2)

Setup:
- backbone: `hf-internal-testing/tiny-random-MistralForCausalLM`
- repeats: `2` (`seed=42`, `seed=1042`)
- eval budget: `64`
- protocol: task-matched checkpoint + args

Artifact:
- `runs/benchmark_local_matched_repeat2_summary_2026-03-11.csv`

Results:

| Dataset | Model | Accuracy Mean | Accuracy CI95 | Latency Mean (s/sample) |
|---|---|---:|---:|---:|
| DTD | `injection` | 0.781250 | 0.000000 | 0.054981 |
| DTD | `expert_only` | 0.031250 | 0.000000 | 0.015018 |
| DTD | `llm_only` | 0.000000 | 0.000000 | 0.005328 |
| Oxford-IIIT Pet | `injection` | 0.890625 | 0.000000 | 0.060251 |
| Oxford-IIIT Pet | `expert_only` | 0.093750 | 0.000000 | 0.014535 |
| Oxford-IIIT Pet | `llm_only` | 0.000000 | 0.000000 | 0.005291 |

### Semantic llm_only benchmark (yes/no floor)

To avoid degenerate floor metrics from strict token-id matching, yes/no baselines are evaluated semantically.

Artifact:
- `runs/benchmark_llm_only_yesno_2026-03-11.csv`

Results:

| Dataset | Model | Accuracy | F1 | Latency (s/sample) |
|---|---|---:|---:|---:|
| DTD | `llm_only` | 0.480469 | 0.324538 | 0.021052 |
| Oxford-IIIT Pet | `llm_only` | 0.531250 | 0.346939 | 0.014022 |

Interpretation:
- `llm_only` is now a meaningful floor on yes/no tasks (near chance as expected on balanced binary prompts).
- Injection remains the strongest local model under matched protocols.

### Hospital text dataset support (industry path)

Plan-1 supports `dataset_name=hospital_text` for description-only clinical workflows.

Required input columns (CSV/JSONL):
- `description` (or `text`/`report`)
- `label` (or `target`/`class`)

Train/eval:
```bash
python scripts/train_plan1.py   --run_dir runs/plan1_hospital_text   --dataset_name hospital_text   --hospital_train_file runs/hospital_train.csv   --hospital_eval_file runs/hospital_eval.csv   --evidence_source text   --qa_type label_code   --use_adapters 1   --adapter_bottleneck 64   --inject_location post_attn
```

```bash
python scripts/eval_plan1.py   --dataset_name hospital_text   --hospital_eval_file runs/hospital_eval.csv   --evidence_source text   --qa_type label_code   --baseline injection   --connectors_path runs/plan1_hospital_text/best_connectors.pt
```

Text-only hospital baseline suite:
```bash
python scripts/run_text_only_baseline_suite.py   --train_file runs/hospital_train.csv   --eval_file runs/hospital_eval.csv   --qa_type yesno   --out_csv runs/hospital_text_baseline_suite_yesno.csv
```

Latest demo repeat-2 artifacts:
- `runs/hospital_text_baseline_suite_yesno_repeat2_detailed.csv`
- `runs/hospital_text_baseline_suite_yesno_repeat2_summary.csv`

Latest demo readout (`hospital_*_demo.csv`, yes/no):
- `injection_text_privacy`: `0.515`
- `llm_only`: `0.515`
- `majority_label`: `0.480`
- `tfidf_logreg`: `1.000`

Note:
- Demo hospital files are sanity datasets, not final clinical-scale benchmarks.
- For industry claims, replace with a real de-identified hospital split and rerun the same scripts.

## Baselines

- `baseline/dp-opt`: Differentially private OPT fine-tuning with ready-to-run sweep configurations.
- `baseline/tablediffusion`: DP table diffusion models with GAN, VAE, and SAINT back-ends.
- `baseline/MICO` and `baseline/mink-plus-plus`: External repositories vendored as submodules; initialize them with `git submodule update --init --recursive` before use.

## Citation

If you use Llamdex in your research, please cite:

```bibtex
@article{wu2024model,
  title={Model-based Large Language Model Customization as Service},
  author={Wu, Zhaomin and Guo, Jizhou and Hou, Junyi and He, Bingsheng and Fan, Lixin and Yang, Qiang},
    journal={EMNLP},
  year={2025}
}
```

## License

This release is distributed under the [Apache License 2.0](LICENSE). By contributing or using the software, you agree to the terms of that license.


