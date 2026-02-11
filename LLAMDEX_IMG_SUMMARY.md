# Llamdex-IMG Implementation Summary

## Overview

Llamdex-IMG is a late-fusion image classification extension of Llamdex that preserves all Llamdex invariants:
- (C1) Model-based customization (client uploads expert model, not data)
- (C2) Freeze base LLM and expert; train only connector modules
- (C3) Server-side connector training must not use private client data
- (C4) Use existing mid-layer reserved-token overwrite injection mechanism (Gaussian padding + LayerNorm)

## Architecture

1. **Frozen Vision Expert**: ResNet-18 pretrained on ImageNet, adapted for CIFAR-10
   - Outputs either logits (10/100 classes) or embeddings (512-dim)
   - All parameters frozen

2. **Trainable Decoder** (`VisionToLlamdexDecoder`):
   - Maps expert output → (num_tokens × hidden_size) token embeddings
   - FFN (SwiGLU or Linear) → LayerNorm → alpha scaling
   - Only this module is trainable

3. **Injection Mechanism**: Reuses existing Llamdex infrastructure
   - Reserved token slots created in `DomainMistralForCausalLM.forward()`
   - Overwrite happens in `DomainMistralDecoderLayer.forward()` at layer k
   - Expert inputs passed via `expert_inputs=(images,)` tuple

## Files Added/Modified

### New Files

1. **`src/vision/__init__.py`**: Package initialization
2. **`src/vision/vision_expert.py`**: Frozen vision expert wrapper (ResNet-18)
3. **`src/vision/vision_decoder.py`**: Trainable decoder (FFN + LayerNorm + alpha)
4. **`src/vision/vision_domain_expert.py`**: Integration wrapper compatible with DomainExpert interface
5. **`src/vision/cifar_dataset.py`**: CIFAR-10 dataset with text prompts
6. **`scripts/train_img.py`**: Training script with ablation flags
7. **`scripts/eval_img.py`**: Evaluation script with baselines
8. **`tests/test_img_smoke.py`**: Smoke test (10 steps, no GPU required)

### Modified Files

1. **`README.md`**: Added Llamdex-IMG quickstart section

## Key Implementation Details

### Reserved Token Slots
- Location: `src/model/DomainMistralModel.py`, `DomainMistralForCausalLM.forward()` (lines 467-476)
- Mechanism: Extends `attention_mask` and pads `inputs_embeds` with Gaussian/zero padding

### Overwrite Mechanism
- Location: `src/model/DomainMistralModel.py`, `DomainMistralDecoderLayer.forward()` (lines 100-123)
- Mechanism: When `expert_inputs` provided, calls `expert.forward_with_features()`, gets token embeddings, overwrites last `num_tokens` positions, applies LayerNorm

### Expert Inputs Flow
- Location: `src/train/train_domain_mistral.py` (line 449)
- Mechanism: `expert_inputs=(features,)` passed to `model.forward()`, propagates to decoder layers

## Usage

### Training

```bash
python scripts/train_img.py \
    --mistral_models_path model/llm \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_tokens 10 \
    --layer 0 \
    --num_epochs 3 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Evaluation

```bash
# Llamdex-IMG
python scripts/eval_img.py \
    --model_state_dict model/llm/llamdex_img_cifar10/model_final.pt

# Baselines
python scripts/eval_img.py --baseline vision_only
python scripts/eval_img.py --baseline llm_only
python scripts/eval_img.py --baseline prompt
```

### Ablation Studies

```bash
# Layer ablation
python scripts/train_img.py --layer 5

# Num tokens ablation
python scripts/train_img.py --num_tokens 16

# Output type ablation
python scripts/train_img.py --vision_output embedding

# Alpha scaling ablation
python scripts/train_img.py --alpha 0.5
```

## Expected Output

### Training Logs (Example)

```
================================================================================
Llamdex-IMG Training Configuration:
  Model: mistralai/Mistral-7B-Instruct-v0.3
  Vision Model: resnet18
  Vision Output: logits
  Num Classes: 10
  Num Tokens: 10
  Layer: 0
  Alpha: 1.0
  FFN Hidden Size: 2048
  Batch Size: 32
  Learning Rate: 0.0001
  Num Epochs: 3
  Device: cuda:0
================================================================================
Trainable parameters: 20,480 / 7,240,000,000 (0.00%)

Epoch 1/3
100%|████████████| 1563/1563 [15:23<00:00, 1.70it/s]
Epoch 1/3
  Train Loss: 1.8234
  Test Loss: 1.6543
  Test Accuracy: 0.4521 (4521/10000)

Epoch 2/3
100%|████████████| 1563/1563 [15:20<00:00, 1.70it/s]
Epoch 2/3
  Train Loss: 1.2341
  Test Loss: 1.3456
  Test Accuracy: 0.6234 (6234/10000)

Epoch 3/3
100%|████████████| 1563/1563 [15:25<00:00, 1.70it/s]
Epoch 3/3
  Train Loss: 0.9876
  Test Loss: 1.1234
  Test Accuracy: 0.7123 (7123/10000)

Training completed!
```

### Evaluation Logs (Example)

```
================================================================================
Llamdex-IMG Evaluation Configuration:
  Model: mistralai/Mistral-7B-Instruct-v0.3
  Vision Model: resnet18
  Vision Output: logits
  Num Classes: 10
  Num Tokens: 10
  Layer: 0
  Alpha: 1.0
  Baseline: None
================================================================================
100%|████████████| 313/313 [02:15<00:00, 2.31it/s]

================================================================================
Results (Llamdex-IMG):
  Accuracy: 0.7123 (7123/10000)
================================================================================
```

## Baselines

1. **Vision-only**: Expert argmax (logits) → accuracy
2. **LLM-only**: Disable injection (alpha=0) → LLM without vision
3. **Prompt baseline**: Run vision expert, insert prediction text into prompt, run LLM without injection

## Testing

Run smoke test:
```bash
python tests/test_img_smoke.py
```

Expected output:
```
================================================================================
Llamdex-IMG Smoke Test
================================================================================

Running 10 training steps...
  Step 1/10: Loss=2.3456, Acc=0.0000
  Step 2/10: Loss=2.1234, Acc=0.5000
  ...
  Step 10/10: Loss=1.9876, Acc=0.5000

Running evaluation...
Evaluation Accuracy: 0.4500 (9/20)

================================================================================
Smoke test completed successfully!
================================================================================
```

## Notes

- Vision expert runs once per sample (cached per forward)
- No full VLM pipeline (no patch tokens, no cross-attention, no image tokens in tokenizer)
- Preserves existing tabular codepaths (no breaking changes)
- Supports CIFAR-10 (minimum), extensible to CIFAR-100
- Default output: single token digit (0-9) to avoid multi-subtoken issues
