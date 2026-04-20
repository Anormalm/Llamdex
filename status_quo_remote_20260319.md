# Plan-1++ Benchmark Status (2026-03-19)

## 1) Git commit hash
- Repo: `/home/anormalm/Llamdex`
- Commit: `bde55e7`
- Branch state: untracked files present (`playbook.md`, `pyproject.toml`, `uv.lock`)

## 2) Environment versions
- Python: `3.10.12`
- torch: `2.10.0+cu128`
- transformers: `5.3.0`
- accelerate: `1.13.0`
- GPUs: 8x `NVIDIA A100-SXM4-80GB`
- GPU note at inspection time: GPU 4 was busy with `VLLM::EngineCore` using ~73.7 GiB; other GPUs were effectively idle

## 3) Benchmark inventory analyzed
Primary March 19 artifacts:
- `/disk1/lfhu/runs/expert_sweep.apples_to_apples.dtd.csv`
- `/disk1/lfhu/runs/expert_sweep.apples_to_apples.dtd.json`
- `/disk1/lfhu/runs/expert_sweep.apples_to_apples.dtd.log`
- `/disk1/lfhu/runs/task_matrix_router_parallel_dtd_resnet18.csv`
- `/disk1/lfhu/runs/task_matrix_router_parallel_dtd_resnet18.json`
- `/disk1/lfhu/runs/task_matrix_router_parallel_dtd_resnet18.log`
- `/disk1/lfhu/runs/task_matrix_overwrite_ablation_dtd_resnet18.csv`
- `/disk1/lfhu/runs/task_matrix_overwrite_ablation_dtd_resnet18.json`
- `/disk1/lfhu/runs/task_matrix_overwrite_ablation_dtd_resnet18.log`
- `/disk1/lfhu/runs/baseline_suite_expanded.csv`
- `/disk1/lfhu/runs/baseline_suite_expanded.json`
- `/disk1/lfhu/runs/baseline_suite_expanded.log`
- `/disk1/lfhu/runs/baseline_suite_expanded.md`

Supporting earlier artifacts used only for context:
- `/disk1/lfhu/runs/expert_sweep.remote.gpu3.csv`
- `/disk1/lfhu/runs/expert_sweep.remote.gpu3.json`
- `/disk1/lfhu/runs/expert_sweep.remote.gpu3.log`
- `/disk1/lfhu/runs/local_arch_baselines.remote.gpu3.csv`
- `/disk1/lfhu/runs/local_arch_baselines.remote.gpu3.json`
- `/disk1/lfhu/runs/local_arch_baselines.remote.gpu3.log`
- `/disk1/lfhu/runs/status_quo_remote_20260318.md`

Missing expected top-level remote artifacts:
- `/disk1/lfhu/runs/unified_benchmark.remote.csv`
- `/disk1/lfhu/runs/unified_benchmark.remote.json`
- `/disk1/lfhu/runs/task_matrix_baselines.remote.csv`
- `/disk1/lfhu/runs/task_matrix_baselines.remote.json`
- `/disk1/lfhu/runs/local_arch_baselines.remote.csv`
- `/disk1/lfhu/runs/local_arch_baselines.remote.json`
- `/disk1/lfhu/runs/api_baselines.remote.csv`
- `/disk1/lfhu/runs/api_baselines.remote.json`
- `/disk1/lfhu/runs/api_baselines.remote.summary.csv`
- `/disk1/lfhu/runs/api_baselines.remote.summary.json`

## 4) Current benchmark conclusions
1. The strongest current evidence is the March 19 DTD apples-to-apples expert sweep.
2. `dinov2` is the best current general-purpose expert for `router_parallel` on DTD.
3. `router_parallel` materially outperforms weak floor baselines where meaningful comparisons exist.
4. The benchmark set is not yet playbook-complete because the freshest run does not include `strict_yesno`, and the unified/API remote outputs are absent.
5. The Oxford Pet local architecture baseline file on GPU3 is only a floor benchmark because it used no expert checkpoint.

## 5) Collated result table
### A. Freshest comparable run: DTD apples-to-apples expert sweep
Source:
- `/disk1/lfhu/runs/expert_sweep.apples_to_apples.dtd.csv`

| task | best system | metric | supporting metrics | status |
|---|---|---:|---|---|
| finegrained | `router_parallel/dinov2` | `0.7344` | `f1=0.8309` | `ok` |
| population | `router_parallel/dinov2` and `router_parallel/siglip` and `router_parallel/resnet18_classifier` | `0.8672` | best observed `mae=0.0186`, `calibration_error=0.0186` | `ok` |
| grounded_generation | `router_parallel/dinov2` | `0.7266` | `f1=0.7035`, `format_compliance=1.0`, `rationale_nonempty_rate=1.0`, `rationale_consistency=0.0` | `ok` |

Full per-expert summary:

| expert | finegrained | grounded_generation | population |
|---|---:|---:|---:|
| `dinov2` | `0.7344` | `0.7266` | `0.8672` |
| `siglip` | `0.6523` | `0.6875` | `0.8672` |
| `clip` | `0.6445` | `0.5859` | `0.6250` |
| `resnet18_classifier` | `0.0664` | `0.0938` | `0.8672` |

Interpretation:
- `dinov2` is the strongest all-around expert.
- `siglip` is competitive, but consistently behind `dinov2` on DTD.
- `clip` is clearly worse than `dinov2` on all three measured tasks.
- `resnet18_classifier` is highly specialized: it works for population aggregation but is poor for finegrained and grounded generation.

### B. Overwrite vs router_parallel on DTD
Sources:
- `/disk1/lfhu/runs/task_matrix_router_parallel_dtd_resnet18.csv`
- `/disk1/lfhu/runs/task_matrix_overwrite_ablation_dtd_resnet18.csv`

| task | overwrite/resnet18 | router_parallel/resnet18 | delta |
|---|---:|---:|---:|
| finegrained | `0.1562` | `0.0664` | `-0.0898` |
| population | `0.0000` | `0.8672` | `+0.8672` |

Interpretation:
- `router_parallel` is decisively better for population aggregation.
- `overwrite` does slightly better on DTD finegrained with this weak resnet18 expert, but both results are poor in absolute terms.
- The better comparison is across stronger experts, where `router_parallel/dinov2` is already the leading result.

### C. Older broader sweep on Oxford Pet
Source:
- `/disk1/lfhu/runs/expert_sweep.remote.gpu3.csv`

Best observed rows by task:

| task | best system | metric | note |
|---|---|---:|---|
| finegrained | `router_parallel/dinov2` | `0.7109` | meaningful |
| strict_yesno | `router_parallel/siglip` or `router_parallel/dinov2` | `0.9609` | easiest task in current matrix |
| population | `router_parallel/clip` or `router_parallel/siglip` or `router_parallel/dinov2` | `0.8333` | floor baselines were `0.0` |
| grounded_generation | all rows effectively `0.0` | `0.0000` | likely stale or pre-fix artifact; not consistent with March 19 DTD sweep |

Interpretation:
- This file is still useful for confirming that `strict_yesno` can be very strong under `router_parallel`.
- It is not fully trustworthy for comparing experts on all tasks because several rows that are now clearly plausible on DTD were exactly zero here.

### D. Local baseline suite
Sources:
- `/disk1/lfhu/runs/baseline_suite_expanded.csv`
- `/disk1/lfhu/runs/local_arch_baselines.remote.gpu3.csv`

DTD local benchmark:

| task | best system | metric | note |
|---|---|---:|---|
| single_image | `injection` | `0.3047` | best row in local baseline suite |
| single_image | `rag_structured` | `0.0391` | weak floor |
| single_image | `llm_only` | `0.0312` | weak floor |
| single_image | `two_stage_caption_llm` | `0.0234` | weak floor |

Oxford Pet GPU3 local baselines:

| task | best system | metric | note |
|---|---|---:|---|
| single_image | `two_stage_caption_llm` | `0.0547` | floor only |
| single_image | `llm_only` | `0.0469` | floor only |
| single_image | `rag_structured` | `0.0469` | floor only |

Interpretation:
- The DTD local suite supports the core claim that injection beats the non-injection baselines by a wide margin.
- The Oxford Pet GPU3 local baseline file should not be used as a serious architecture comparison because its log states the vision classifier head was randomly initialized due to missing checkpoint.

## 6) Playbook compliance check
Required task-matrix coverage from `playbook.md`:
1. single-image classification
2. yes/no QA
3. population aggregation
4. grounded answer + rationale

Current state:
- single-image classification: covered
- yes/no QA: covered only in the older Oxford Pet sweep, not in the freshest March 19 DTD sweep
- population aggregation: covered
- grounded answer + rationale: covered

Required artifacts:
1. machine-readable metrics
2. full log file
3. command record in markdown report

Current state:
- machine-readable metrics: present for the runs analyzed
- full logs: present for the runs analyzed
- markdown report: only partial or outdated until this file

Result:
- Partial compliance only
- Main missing piece is a fresh unified run or fresh task-matrix run that includes all four required tracks, especially `strict_yesno`

## 7) Important caveats
1. The March 18 report claiming API failure from missing `src.multimodal.data` is stale relative to the current repository snapshot, because `src/multimodal/data/*` exists now.
2. The freshest apples-to-apples DTD sweep is the most reliable source for expert comparison, but it omits `strict_yesno`.
3. `rationale_consistency` remains `0.0` even where `format_compliance=1.0` and `rationale_nonempty_rate=1.0`; the model is producing formatted rationale text, but not evidence-consistent rationales by the current scorer.
4. The current DTD local injection result (`0.3047`) is materially below the best task-matrix DTD result (`0.7344`), so these files are not directly interchangeable benchmarks. They use different runners, tasks, and comparison sets.
5. The current environment versions differ from the March 18 status report. Any rerun should record versions again because reproducibility may drift across `torch`/`transformers` changes.

## 8) Root-cause analysis
### Why `dinov2` is winning
- It is the strongest and most stable general visual encoder in the current sweep.
- Its gains hold across both finegrained recognition and grounded generation, not just one task.
- `siglip` is close but consistently behind.

### Why the resnet18 expert underperforms
- It is a narrow supervised classifier checkpoint with a 47-way output head.
- That representation appears brittle for tasks that need richer semantics, especially grounded generation.
- Its strong population result suggests it carries coarse class-frequency evidence but not the right representation for label-grounding language generation.

### Why floor baselines are so weak
- `llm_only`, `rag_structured`, and caption-then-LLM baselines are not seeing the visual signal in a way that matches the constrained answer protocol.
- On Oxford Pet GPU3, the local baseline log explicitly warns the vision classifier was randomly initialized, which collapses the quality of the comparison.

### Why grounded-generation still has a quality gap
- Format compliance is solved, but rationale consistency is not.
- This suggests the decoding/output contract is learned better than the explanation grounding itself.
- That aligns with the project priority in `playbook.md` to focus on grounded answer + rationale and natural-language tuning.

## 9) Exact commands associated with current artifacts
These commands are the artifact-producing entrypoints implied by the saved configs and logs.

```bash
cd /home/anormalm/Llamdex

PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_expert_encoder_sweep.py \
  --config conf/expert_sweep.apples_to_apples.dtd.json \
  2>&1 | tee /disk1/lfhu/runs/expert_sweep.apples_to_apples.dtd.log

PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_task_matrix_eval.py \
  --config conf/task_matrix_router_parallel.dtd_resnet18.json \
  2>&1 | tee /disk1/lfhu/runs/task_matrix_router_parallel_dtd_resnet18.log

PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_task_matrix_eval.py \
  --config conf/task_matrix_overwrite_ablation.dtd_resnet18.json \
  2>&1 | tee /disk1/lfhu/runs/task_matrix_overwrite_ablation_dtd_resnet18.log

PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_baseline_benchmark_suite.py \
  --config conf/baseline_benchmark.expanded.json \
  2>&1 | tee /disk1/lfhu/runs/baseline_suite_expanded.log
```

Older contextual commands:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_expert_encoder_sweep.py \
  --config conf/expert_sweep.remote.gpu3.json \
  2>&1 | tee /disk1/lfhu/runs/expert_sweep.remote.gpu3.log

PYTHONUNBUFFERED=1 PYTHONPATH=/home/anormalm/Llamdex \
python -u scripts/run_baseline_benchmark_suite.py \
  --config conf/local_arch_baselines.remote.gpu3.json \
  2>&1 | tee /disk1/lfhu/runs/local_arch_baselines.remote.gpu3.log
```

## 10) Recommended next 72-hour plan
1. Add `strict_yesno` to the DTD apples-to-apples config so the freshest benchmark satisfies the playbook’s minimum task matrix.
2. Run the configured unified benchmark path and produce the missing:
   - `unified_benchmark.remote.csv`
   - `unified_benchmark.remote.json`
   - `api_baselines.remote.*`
3. Fix the local Oxford Pet baseline config to use a real expert checkpoint or drop it from headline comparisons.
4. Treat `dinov2` as the default expert for further connector/policy work unless a narrower task-specific result justifies otherwise.
5. Focus model work on grounded rationale quality, because rationale formatting is solved but rationale consistency remains at zero.

## 11) Bottom line
- Best current overall system: `Qwen/Qwen3.5-9B + router_parallel + dinov2`
- Strongest current evidence: March 19 DTD apples-to-apples sweep
- Headline result: `dinov2` leads on finegrained and grounded generation and ties for best population performance
- Main benchmark gap: no fresh all-four-task remote benchmark with unified/API outputs
