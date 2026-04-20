# Status Quo Remote Snapshot



The intended status quo is:

1. Train the expert stack on the client side.
2. Export the frozen expert artifact.
3. Upload only the frozen expert to the server.
4. Reload that expert inside the server runtime.
5. Compute `z_ctx` locally inside the runtime from sanitized server-visible input.
6. Inject that locally computed evidence through one of two supported router policies:
   - `post_attn_router_parallel`
   - `pre_ffn_router_parallel`

The privacy rule is:

`Upload expert, not per-example z.`

Per-sample evidence vectors are treated as privacy-sensitive and should not be transferred as part of the serving contract.

## Current Recommended Architecture

### Expert-Only Handoff

- Client trains the expert stack locally.
- Client uploads only the frozen expert artifact plus manifest/config metadata.
- Server validates and reloads the uploaded expert.
- Server reconstructs or trains builder / projector / router modules server-side.
- Server never receives raw private artifacts as part of the upload contract.
- Server does not receive per-example `z`.

### Injection Policies Under Comparison

#### `post_attn_router_parallel`

- Late fusion near the output side of the transformer.
- `z_ctx` is fused with a late language hidden state.
- Stronger direct control over final token logits.
- Best fit for grounded answer-code generation and final-token steering.

#### `pre_ffn_router_parallel`

- Mid-layer fusion inside the FFN branch.
- `z_ctx` perturbs hidden features before later layers finish processing.
- Less direct control over final logits, but stronger internal feature shaping.
- Best fit for classification-style adaptation and structured prediction.

## Current Status-Quo Position

The latest best design should be presented as:

- `expert-only upload`
- `local z_ctx computation after reload`
- `post_attn_router_parallel` and `pre_ffn_router_parallel` as the two main transformer injection mechanisms

This is the design baseline that downstream diagrams and benchmark summaries should assume.

## 2026-04-01 Completed Benchmark Snapshot

| Method | Finegrained | Strict Yes/No | Population ECE | Population MAE | Grounded Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM-only | 0.093 | 0.230 | 0.287 | 0.142 | 0.012 |
| LoRA finetune | 0.412 | 0.953 | 0.061 | 0.023 | 0.318 |
| API GPT/VLM | 0.914 | 0.897 | 0.031 | 0.023 | 0.842 |
| post_attn_router | 0.964 | 0.897 | 0.018 | 0.023 | 0.903 |
| pre_ffn_router | 0.977 | 0.974 | 0.009 | 0.010 | 0.781 |
