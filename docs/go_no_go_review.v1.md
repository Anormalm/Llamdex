# Pilot Go/No-Go Review

## Scope

This review gates a pilot deployment of the Qwen-first Llamdex server path with frozen client-uploaded experts and connector-only adaptation.

## Go/No-Go Criteria

### Data

- Hospital/private ingestion uses the schema at `conf/hospital_ingest.schema.json`.
- Release manifests validate against `conf/dataset_manifest.schema.json`.
- Label ontology is frozen for the pilot slice.
- Text-first privacy mode is enforced for pilot data.

### Evaluation

- Task matrix is frozen at `conf/task_matrix.freeze.v1.json`.
- Grounded answer + rationale has format-compliance and rationale-nonempty metrics.
- At least one local baseline and one API baseline are available for semantic comparison.
- Error rows in benchmark outputs are reviewed and triaged before pilot.

### Model

- Server backbone is Qwen-only for the pilot run.
- Public injection surface is limited to `layer_input` plus router-parallel policy.
- Checkpoint compatibility guard passes for all promoted connector bundles.
- Bundle manifest/version checks are enabled for uploaded client experts.

### Privacy / Ops

- No raw private images or raw reports are sent to the server path.
- Logs contain trace IDs and sanitized metadata only.
- Text-first policy rejects non-deidentified or failed-quality records.
- Rollback path exists by bundle ID and config version.

## Current Review Template

| Area | Status | Evidence | Owner | Notes |
|---|---|---|---|---|
| Data contract | Pending | `scripts/validate_data_contracts.py` | | |
| Dataset release manifest | Pending | `manifest.v1.json` | | |
| Task-matrix freeze | Pending | `conf/task_matrix.freeze.v1.json` | | |
| Local baselines | Pending | `runs/baseline_suite*.json` | | |
| API baselines | Pending | `runs/api_baseline*.json` | | |
| Grounded rationale | Pending | `runs/urgent_grounded_rationale*.json` | | |
| Text privacy mode | Pending | `conf/text_privacy_mode.v1.json` | | |
| Bundle compatibility | Pending | `tests/test_server_bundle_contract.py` | | |

## Decision Rule

- `Go`: every critical line item above is `Pass` or `Accepted Risk`.
- `No-Go`: any privacy, schema, or compatibility gate is `Fail`.
- `Conditional Go`: benchmark weakness is allowed only if privacy and compatibility gates are already green and the pilot scope is explicitly narrowed.
