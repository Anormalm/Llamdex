---
title: "Llamdex Plan-1++ Status"
subtitle: "Privacy-Bounded Multimodal Injection, Corrected Baselines, and Next Publishable Claim"
author: "Project update for advisor and research group"
date: "April 28, 2026"
---

# Status Message

**What changed since the last deck**

- The system direction is intact: private client evidence is injected into a frozen server model through small trainable adapters.
- The evaluation is now stricter: DTD and Oxford labels use real dataset ontologies, not placeholder class names.
- The baseline story changed: adapted representation baselines are strong and are now the right comparator.
- The current publishable claim should be framed as privacy-bounded adaptation and grounded behavior, not image-classification SOTA yet.

# Research Question

**Can we adapt a frozen multimodal LLM without sending private raw data to the server?**

- Client side: private image, local expert, evidence bundle.
- Server side: frozen backbone, trainable connector/router, constrained task output.
- Goal: recover task-relevant visual information while preserving a bounded privacy contract.
- Evaluation target: compare injection policies against strong local representation and API baselines.

# System Overview

![Client-server privacy-bounded Llamdex architecture](docs/slide_assets/server_client_architecture.png){width=82%}

# Injection Mechanisms

![Late-output fusion and mid-layer FFN fusion](docs/slide_assets/injection_mechanisms.png){width=86%}

# Policy Variants Under Test

**Why pre-FFN and post-attn are both still relevant**

- Pre-FFN router: injects evidence before the feed-forward block; can improve task routing and answer selection.
- Post-attn-layers: injects after attention across layers; appears better for nonempty and faithful rationales.
- Current decision point: classification accuracy and rationale quality are not optimized by the same injection site.

![Pre-FFN router architecture](docs/slide_assets/pre_ffn_router.png){width=70%}

# Evaluation Matrix

**Tasks and datasets**

| Axis | Current protocol |
|---|---|
| Fine-grained classification | DTD texture labels, Oxford-IIIT Pet labels |
| Strict yes/no | Binary visual assertions with constrained decoding |
| Population estimation | Numeric/relative estimation with MAE |
| Grounded generation | Answer plus rationale/evidence consistency |

**Guardrails added**

- Real class-name ontologies for DTD and Oxford-IIIT Pet.
- Robust parsing for `CODE=label` outputs.
- Raw prediction audit logs for error analysis.
- DINOv2 and CLIP linear probes as representation baselines.

# Baseline Audit

**Why the old baseline slide was not publishable**

- DTD class names were placeholders, so zero-shot prompts did not encode the real label space.
- Some `CODE=label` responses were parsed by the label text instead of the option code.
- Several earlier rows mixed current and stale artifacts.
- API rerun is currently blocked by quota, so older API numbers should be marked historical only.

**What is publishable now**

- The corrected protocol is auditable and reproducible.
- Baselines now include strong adapted representation models.
- Current connector rows are reported as status, not as final claims.

# Corrected DTD Baseline Snapshot

**One-shot semantic-v2 protocol, 128 eval images**

| Method | Accuracy | Macro-F1 | Interpretation |
|---|---:|---:|---|
| DINOv2 linear probe | 85.2% | 79.2% | Strong local representation baseline |
| CLIP linear probe | 82.0% | 77.1% | Strong local representation baseline |
| Current injection checkpoint | 9.4% | n/a | Stale/mismatched connector row |
| Caption + LLM | 7.0% | 2.0% | Weak for texture taxonomy |
| LLM only | 3.1% | 0.2% | Near random over 47 classes |
| Frozen VLM prompts | 2-3% | 0-1% | Prompting alone is not enough |

**Main reading**

- The meaningful baseline is not zero-shot chat VLM prompting.
- The meaningful baseline is adapted visual representation performance.
- Beating 82-85% on DTD requires connector training under the corrected label protocol.

# Task-Matrix Connector Status

**Qwen3-1.7B runs, router-parallel summary**

| Metric | Pre-FFN | Post-attn-layers | Current reading |
|---|---:|---:|---|
| Fine-grained accuracy | 53.9% | 51.2% | Pre-FFN slightly higher |
| Strict yes/no accuracy | 98.4% | 98.4% | Both strong |
| Population MAE | 0.0228 | 0.0228 | No separation |
| Grounded-generation accuracy | 55.5% | 53.9% | Similar |
| Rationale nonempty rate | 6.8% | 72.4% | Post-attn much better |
| Rationale faithful rate | 5.7% | 29.2% | Post-attn much better |

**Main reading**

- Injection is useful across the task matrix.
- The architecture tradeoff is real: answer accuracy and rationale faithfulness currently diverge.
- The next benchmark needs the same corrected label protocol used by the new baselines.

# Current Claim Boundary

**What we can say now**

- Llamdex implements a privacy-bounded client/server evidence-injection contract.
- The benchmark pipeline now includes auditable labels, parsing, and raw predictions.
- Pre-FFN and post-attn injection show different strengths across task types.
- Strong representation baselines set a clear target for publishable classification claims.

**What we should not say yet**

- Not SOTA on DTD classification.
- Not yet apples-to-apples against current API models because quota blocked the fresh run.
- Not yet a final connector result because the local connector row is stale/mismatched.

# Publishable Path

**Near-term experiment plan**

| Step | Output needed | Why it matters |
|---|---|---|
| Retrain connector on corrected semantic labels | New injection rows | Fair comparison to DINOv2/CLIP probes |
| Rerun API baseline or approved substitute | Current external comparator | Prevents stale API overclaiming |
| Add audit appendix | Raw predictions and confusion slices | Makes failure modes defensible |
| Compare injection-site policies | Pre-FFN vs post-attn under same protocol | Supports architecture claim |
| Freeze artifact manifest | Configs, commits, run paths | Enables reproducible report |

# Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Linear probes outperform injection on fine-grained DTD | Reframe as privacy-bounded adaptation; train connector under corrected labels |
| Zero-shot VLM baselines look artificially weak | Report them as prompting baselines, not SOTA baselines |
| API quota blocks fresh comparator | Mark API historical and rerun when quota is available |
| Rationale metrics lag answer metrics | Prefer post-attn-layers for grounded generation, or train a rationale objective |
| Mixed stale artifacts confuse story | Use one artifact manifest and show run paths in appendix |

# Tomorrow's Ask

**Feedback requested from the group**

- Is the publishable claim better framed as privacy-bounded adaptation rather than classification SOTA?
- Which comparator should be mandatory: DINOv2/CLIP probes, API VLMs, or both?
- Should the next connector training optimize classification first, rationale faithfulness first, or a multi-objective mix?
- What privacy threat model detail is needed before writing the paper section?

# Appendix: Artifact Pointers

**Current validated artifacts**

| Artifact | Path |
|---|---|
| Corrected DTD baseline CSV | `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.csv` |
| Corrected DTD audit JSONL | `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.audit.jsonl` |
| Pre-FFN task matrix CSV | `/disk1/lfhu/runs/task_matrix_sota_reasonable.pre_ffn.qwen3_1p7b.gpu3.csv` |
| Post-attn task matrix CSV | `/disk1/lfhu/runs/task_matrix_sota_reasonable.post_attn_layers.qwen3_1p7b.gpu4.csv` |
| API rerun status | blocked by insufficient quota |
