# Status Quo Remote Snapshot



The intended status quo is:

1. Train the expert stack on the client side.
2. Export a frozen expert bundle.
3. Upload only the frozen bundle to the server.
4. Reload the bundle inside the server runtime.
5. Compute `z_ctx` locally inside the runtime from sanitized server-visible input.
6. Inject that locally computed evidence through one of two supported router policies:
   - `post_attn_router_parallel`
   - `pre_ffn_router_parallel`

The privacy rule is:

`Upload bundle, not per-example z.`

Per-sample evidence vectors are treated as privacy-sensitive and should not be transferred as part of the serving contract.

## Current Recommended Architecture

### Bundle-Only Handoff

- Client trains expert / projector / fusion stack locally.
- Client packages a frozen bundle with weights and manifest metadata.
- Server validates and reloads the uploaded bundle.
- Server never receives raw private artifacts as part of the bundle contract.
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

- `bundle-only upload`
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
