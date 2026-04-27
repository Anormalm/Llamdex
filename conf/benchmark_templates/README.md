# Industry Benchmark Templates

This folder contains CSV templates for reporting industry/academia-recognized benchmark results.

## Files

- `industry_benchmark_run_template.csv`
  - One row per benchmark run/split.
  - Use for raw reporting and auditability.

- `industry_benchmark_leaderboard_template.csv`
  - One row per model variant.
  - Use for summary tables in docs/slides.

## Recommended Macros

- `vision_macro`: mean of MMMU, MMBench-EN, MMVet, ScienceQA, TextVQA, DocVQA, ChartQA, POPE
- `text_macro`: mean of MMLU-Pro, GPQA, HumanEval
- `safety_macro`: normalized safety score from HarmBench
- `overall_macro`: weighted mean, e.g. `0.5 * vision_macro + 0.3 * text_macro + 0.2 * safety_macro`

## Model Variant Naming

Use one of:

- `default_preffn`
- `specialized_postattn`
- `hybrid_task_routed`

## Reporting Hygiene

Always include:

- exact benchmark version/split
- zero-shot/few-shot setting
- decode settings
- hardware
- latency and throughput
- cost
- run ID and UTC date
