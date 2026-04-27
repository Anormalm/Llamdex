# Llamdex Plan-1++ Slide-Maker Prompts

Use these prompts page-by-page to revise the existing `llamdex_plan1pp_status` deck into a publishable research update for an advisor and lab audience. Preserve the original technical visual identity where possible, but make the claims more rigorous and artifact-backed.

Global rules for every slide:

- Tone: research update, not marketing.
- Avoid claiming SOTA unless a current apples-to-apples benchmark proves it.
- Distinguish validated artifacts from stale, historical, or blocked runs.
- Use exact metric labels and dates where numbers appear.
- Prefer concise claims with artifact-backed caveats.
- Emphasize that the corrected baseline story makes the project more rigorous, not weaker.

---

## Slide 1: Title / Status

Prompt:

Revise the title slide to frame the talk as a project status update, not a final performance claim. Title should be: “Llamdex Plan-1++ Status: Privacy-Bounded Multimodal Injection”. Subtitle should mention “Corrected baselines, architecture status, and next publishable claim”. Date should be April 28, 2026. Add a small status strip with three factual anchors: DINOv2 linear probe 85.2% corrected DTD accuracy, CLIP linear probe 82.0% corrected DTD accuracy, and router-parallel strict yes/no 98.4% in task-matrix runs. Do not imply Llamdex beats these baselines yet.

Fact-check:

- DINOv2 linear probe accuracy: 85.2% from `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.csv`.
- CLIP linear probe accuracy: 82.0% from the same corrected DTD run.
- Strict yes/no task-matrix accuracy: 98.4% for both pre-FFN and post-attn-layers router-parallel summaries.

Avoid:

- “SOTA”
- “final results”
- “outperforms API/VLMs”

---

## Slide 2: One-Slide Message

Prompt:

Replace the old key-updates framing with a direct status message: the architecture direction is intact, but the evaluation is now stricter and the baseline interpretation changed. Use four bullets: 1) Private client evidence is injected into a frozen server model through small trainable adapters. 2) Dataset label ontologies and output parsing were corrected. 3) Adapted representation baselines are now the right comparator. 4) The current publishable claim should be privacy-bounded adaptation and grounded behavior, not image-classification SOTA yet. Add a small “claim boundary” callout.

Fact-check:

- DTD and Oxford labels were corrected from placeholders to real label sets.
- `CODE=label` parsing was fixed so option codes are interpreted correctly.
- Current API rerun is blocked by quota; do not use old API results as current.

Avoid:

- Treating low zero-shot VLM numbers as the main baseline.
- Treating the current injection row as final.

---

## Slide 3: Research Question

Prompt:

Refocus the problem statement around the privacy boundary: “Can we adapt a frozen multimodal LLM without sending private raw data to the server?” Show three zones: Client, Boundary, Server. Client has private image, local expert, evidence bundle. Boundary has compact evidence only and manifest/contract. Server has frozen backbone, connector/router, constrained output. End with the evaluation target: compare injection policies against strong local representation and API baselines.

Fact-check:

- Llamdex design is client-side expert/evidence plus frozen server-side runtime.
- Trainable components are connector/router/adapters, not the frozen backbone.

Avoid:

- Saying privacy is fully proven; phrase as “privacy-bounded contract” or “bounded privacy design”.

---

## Slide 4: System Overview

Prompt:

Keep the existing Llamdex-IMG overview visual if it is clean, but simplify the surrounding labels. The slide should communicate the data path: private image -> client expert -> evidence bundle -> frozen server runtime -> injected hidden states/output constraints. Add labels for “raw image stays client-side” and “server receives compact evidence”. Use this as a system slide, not a results slide.

Fact-check:

- Raw image should be described as client-side/private.
- Evidence bundle crosses the boundary, not the raw image.
- Server model is frozen in the intended setup.

Avoid:

- Overclaiming formal privacy guarantees if the deck has not defined a threat model.

---

## Slide 5: Injection Mechanisms

Prompt:

Keep or redraw the injection-mechanism diagram, but make the comparison explicit: late-output fusion versus mid-layer FFN/hidden-state fusion. Add one sentence: “The research axis is where and how compact evidence should enter the frozen model.” Show that late fusion is simpler but weaker for internal reasoning, while mid-layer injection can affect token generation and rationale behavior.

Fact-check:

- The project tested pre-FFN and post-attn-layer injection variants.
- Do not collapse all injection policies into one architecture.

Avoid:

- Saying one mechanism is definitively best across all tasks.

---

## Slide 6: Policy Variants Under Test

Prompt:

Add a slide comparing pre-FFN router and post-attn-layers. Pre-FFN: injects evidence before feed-forward block, slightly higher current fine-grained and grounded-generation accuracy. Post-attn-layers: injects after attention across layers, much better rationale nonempty and faithful rates. Use a balanced tradeoff visual rather than a winner/loser framing.

Fact-check:

- Pre-FFN router-parallel means: fine-grained accuracy 53.9%, grounded-generation accuracy 55.5%.
- Post-attn-layers router-parallel means: fine-grained accuracy 51.2%, grounded-generation accuracy 53.9%.
- Post-attn-layers rationale nonempty rate 72.4% vs pre-FFN 6.8%.
- Post-attn-layers rationale faithful rate 29.2% vs pre-FFN 5.7%.

Avoid:

- Claiming post-attn is universally better.
- Claiming pre-FFN is publishably better without rerunning corrected label protocol.

---

## Slide 7: Evaluation Protocol / Task Matrix

Prompt:

Revise the task-matrix slide so it emphasizes protocol quality. Show four task axes: fine-grained classification, strict yes/no, population estimation, grounded generation. Add a “guardrails added” box: real class-name ontologies, robust `CODE=label` parser, raw prediction audit logs, DINOv2 and CLIP linear probes. This slide should explain why the new numbers are more meaningful than the old numbers.

Fact-check:

- DTD has 47 real texture labels.
- Oxford-IIIT Pet has 37 real pet breed labels.
- Corrected DTD baseline run has audit JSONL at `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.audit.jsonl`.

Avoid:

- Showing only LLM/VLM prompting baselines.
- Omitting linear probes.

---

## Slide 8: Baseline Audit / What Was Fixed

Prompt:

Replace any stale benchmark snapshot with a transparent baseline-audit slide. Title: “Baseline audit: why the old baseline was too low.” Explain: DTD class names were placeholders, `CODE=label` parsing could misread outputs, earlier rows mixed current and stale artifacts, and fresh API rerun is blocked by quota. Then state the correction: use real label ontologies, parser regression tests, raw prediction audits, and DINOv2/CLIP linear probes.

Fact-check:

- API rerun blocked by OpenAI insufficient quota in `/disk1/lfhu/runs/api_baseline_sota_urgent.log`.
- Historical API CSV/JSON from March should not be presented as current.

Avoid:

- Hiding the benchmark issue.
- Calling the previous slide “wrong” without explaining the technical reason.

---

## Slide 9: Corrected DTD Baseline Snapshot

Prompt:

Create a clean corrected-baseline table for the DTD one-shot semantic-v2 protocol. Use columns: Method, Accuracy, Macro-F1, Interpretation. Rows: DINOv2 linear probe 85.2%, 79.2%, strong local representation baseline. CLIP linear probe 82.0%, 77.1%, strong local representation baseline. Current injection checkpoint 9.4%, macro-F1 n/a or omit, stale/mismatched connector row. Caption+LLM 7.0%, 2.0%, weak for texture taxonomy. LLM-only 3.1%, 0.2%, near random over 47 classes. Frozen VLM prompts 2-3%, 0-1%, prompting alone is insufficient. Add takeaway: “The meaningful baseline is adapted representation performance, not zero-shot chat prompting.”

Fact-check:

- Corrected run path: `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.csv`.
- DINOv2 exact accuracy: 0.8515625.
- CLIP exact accuracy: 0.8203125.
- Caption+LLM exact accuracy: 0.0703125.
- LLM-only exact accuracy: 0.03125.
- Frozen VLM rows are approximately 2-3% accuracy depending on Qwen/LLaVA row.

Avoid:

- Saying Llamdex loses definitively; the current injection checkpoint is stale/mismatched.
- Saying zero-shot VLMs are SOTA baselines.

---

## Slide 10: Task-Matrix Connector Status

Prompt:

Create a concise table comparing pre-FFN and post-attn-layers router-parallel summaries. Include: fine-grained accuracy 53.9% vs 51.2%, strict yes/no 98.4% vs 98.4%, population MAE 0.0228 vs 0.0228, grounded-generation accuracy 55.5% vs 53.9%, rationale nonempty 6.8% vs 72.4%, rationale faithful 5.7% vs 29.2%. Add takeaway: “Injection is useful across the task matrix, but answer accuracy and rationale quality currently diverge.”

Fact-check:

- Pre-FFN CSV: `/disk1/lfhu/runs/task_matrix_sota_reasonable.pre_ffn.qwen3_1p7b.gpu3.csv`.
- Post-attn-layers CSV: `/disk1/lfhu/runs/task_matrix_sota_reasonable.post_attn_layers.qwen3_1p7b.gpu4.csv`.
- Both completed 36 rows.

Avoid:

- Comparing these task-matrix numbers directly to the corrected one-shot DTD linear probes as if protocols are identical.

---

## Slide 11: Current Claim Boundary

Prompt:

Add a claim-boundary slide with two columns: “Can say now” and “Should not say yet.” Can say now: Llamdex implements a privacy-bounded client/server evidence-injection contract; benchmark pipeline has auditable labels, parsing, and raw predictions; pre-FFN and post-attn injection show distinct strengths. Should not say yet: not SOTA on DTD classification; not apples-to-apples against current API models due quota; not final connector result until retrained under corrected semantic labels.

Fact-check:

- Current API baseline rerun is blocked.
- Corrected local baseline is validated.
- Connector retraining under corrected semantic labels has not yet been completed.

Avoid:

- Making this slide sound apologetic. Frame it as rigorous claim management.

---

## Slide 12: Publishable Path / Next Experiments

Prompt:

Create a next-steps slide with a table: Step, Output needed, Why it matters. Steps: retrain connector under corrected semantic labels; rerun API baseline or approved substitute; add audit appendix with raw predictions and confusion slices; compare pre-FFN vs post-attn under one protocol; freeze artifact manifest with configs, commits, and run paths. End with target claim: “privacy-bounded connector adaptation with grounded output, not universal image-classification SOTA.”

Fact-check:

- Retraining is needed before claiming injection competes with DINOv2/CLIP probes on DTD.
- API baseline should be current or explicitly historical.

Avoid:

- Saying fine-tuning the whole server model is the main plan. The Llamdex premise is small connector/router adaptation around a frozen backbone.

---

## Slide 13: Risks and Mitigations

Prompt:

Add a risks slide if space allows. Risks: linear probes outperform current injection on DTD; zero-shot VLM baselines are artificially weak; API quota blocks fresh comparator; rationale metrics lag answer metrics; mixed artifacts confuse the story. Mitigations: reframe claim as privacy-bounded adaptation, report prompting baselines separately from adapted representation baselines, mark API historical until rerun, train rationale objective or prefer post-attn-layers for grounded generation, and use one artifact manifest.

Fact-check:

- This is a forward-looking risk slide, not a results slide.
- Keep wording precise: “current injection row” not “Llamdex overall”.

Avoid:

- Listing risks without mitigation.
- Using defensive language.

---

## Slide 14: Discussion / Ask for Tomorrow

Prompt:

End with a discussion slide. Ask the group four questions: 1) Is the publishable claim better framed as privacy-bounded adaptation rather than classification SOTA? 2) Which comparator is mandatory: DINOv2/CLIP probes, API VLMs, or both? 3) Should next connector training optimize classification first, rationale faithfulness first, or a multi-objective mix? 4) What privacy threat-model detail is needed before writing the paper section? Add artifact anchors in small text: corrected DTD baseline CSV, audit JSONL, pre-FFN task-matrix CSV, post-attn task-matrix CSV, API rerun blocked by quota.

Fact-check:

- Artifact paths:
  - `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.csv`
  - `/disk1/lfhu/runs/local_arch_baselines.meaningful.dtd.semantic_v2.one_shot.audit.jsonl`
  - `/disk1/lfhu/runs/task_matrix_sota_reasonable.pre_ffn.qwen3_1p7b.gpu3.csv`
  - `/disk1/lfhu/runs/task_matrix_sota_reasonable.post_attn_layers.qwen3_1p7b.gpu4.csv`

Avoid:

- Ending with only a timeline. End with explicit research decisions needed from the audience.

---

# Optional Speaker Framing

Use this opening:

“The main update is that we made the evaluation stricter. That changed the baseline story: zero-shot VLM prompting is not the meaningful comparator for DTD; DINOv2 and CLIP linear probes are. So the project claim should be tightened. Llamdex is currently best framed as privacy-bounded connector adaptation with grounded behavior, and the next decisive experiment is connector retraining under the corrected semantic-label protocol.”

Use this answer if asked why the old baseline was low:

“The old low baseline was partly an evaluation artifact and partly a real limitation of zero-shot prompting for DTD. The label ontology and parser fixes made the benchmark more meaningful. Once we add DINOv2 and CLIP linear probes, the baseline becomes much stronger, around 82-85% accuracy. That is the bar a trained connector needs to address.”
