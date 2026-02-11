# Llamdex: Model-based Large Language Model Customization as Service

This is the official repository of the EMNLP'25 (Main) paper: [Llamdex: Model-based Large Language Model Customization as Service](https://aclanthology.org/2025.emnlp-main.248.pdf).

Llamdex provides a complete pipeline for customizing large language models to structured data domains. This release bundles training code, preprocessing utilities, analysis scripts, and reproducible baseline implementations so you can reproduce our experiments or adapt the workflow to new datasets.

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


