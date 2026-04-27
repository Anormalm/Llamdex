# Llamdex Plan-1++ Revised Deck Speaker Notes

## Core narrative

The project is not weaker because the old baseline was low; the evaluation is now more honest. The corrected result shows that simple adapted visual representations are strong on DTD, so the Llamdex connector must be compared against DINOv2/CLIP probes, not just zero-shot VLM prompts.

## Slide-by-slide talk track

1. Status Message: Lead with the correction. The architecture remains valid, but the benchmark interpretation changed.
2. Research Question: Emphasize the privacy-bounded setting: raw private images stay client-side; only compact evidence crosses the boundary.
3. System Overview: Explain client expert bundle, frozen server runtime, connector/router.
4. Injection Mechanisms: Contrast late-output fusion with mid-layer fusion. State that mid-layer fusion is the main research axis.
5. Policy Variants: Pre-FFN may help answer selection; post-attn-layers is better for rationales.
6. Evaluation Matrix: Say that the protocol is now auditable: real labels, parser fixes, raw predictions, linear probes.
7. Baseline Audit: Be direct that the prior benchmark slide should not be used for publishable claims.
8. Corrected DTD Baseline Snapshot: DINOv2/CLIP probes are the bar. Current connector row is stale/mismatched and should not be overinterpreted.
9. Task-Matrix Connector Status: The task matrix still shows useful injection behavior, especially yes/no and grounded generation, but must be rerun under the corrected protocol.
10. Current Claim Boundary: Draw the line between validated claims and pending claims.
11. Publishable Path: Ask for agreement on the next mandatory experiments.
12. Risks and Mitigations: Show that risks are understood and have concrete mitigation steps.
13. Tomorrow's Ask: Use this to drive discussion, not just report numbers.
14. Appendix: Keep artifact paths available for reproducibility questions.

## Expected questions

Q: Why are zero-shot VLMs so low?
A: DTD is a 47-way texture taxonomy. Prompting a frozen chat VLM with semantic labels is not the same as training an image representation. The DINOv2/CLIP probes demonstrate that visual features contain the signal when adapted.

Q: Does this mean Llamdex is not useful?
A: No. It means the claim should not be generic image classification SOTA yet. The stronger claim is privacy-bounded adaptation: can a small connector recover task behavior while the server model remains frozen and raw data stays client-side?

Q: What is the next decisive experiment?
A: Retrain and evaluate the connector under the corrected semantic-label protocol, with DINOv2/CLIP probes and fresh API baselines in the same artifact manifest.

Q: Why keep post-attn if pre-FFN has slightly better accuracy?
A: Post-attn-layers has much higher rationale nonempty and faithful rates. If the paper emphasizes grounded generation or explanation quality, that matters.
