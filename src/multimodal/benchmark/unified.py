from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.multimodal.baselines.api_suite import load_api_baseline_config, run_api_baseline_suite
from src.multimodal.baselines.suite import load_baseline_suite_config, run_baseline_suite
from src.multimodal.task_matrix.runner import load_task_matrix_config, run_task_matrix


@dataclass
class UnifiedBenchmarkConfig:
    task_matrix_config: Optional[str] = None
    local_baseline_config: Optional[str] = None
    api_baseline_config: Optional[str] = None
    out_csv: str = "/disk1/lfhu/runs/unified_benchmark.csv"
    out_json: str = "/disk1/lfhu/runs/unified_benchmark.json"


def _save_rows(rows: List[Dict], out_csv: str, out_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _metric_name(row: Dict) -> str:
    if "metric_name" in row and row["metric_name"]:
        return str(row["metric_name"])
    if "mae" in row:
        return "mae"
    return "accuracy"


def _normalize_task_matrix_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        mode = str(row.get("baseline_mode", "unknown"))
        family = "llamdex" if mode in {"overwrite", "router_parallel"} else "architecture"
        out.append(
            {
                "benchmark_family": family,
                "benchmark_source": "task_matrix",
                "baseline_name": mode,
                "task": row.get("task"),
                "dataset": row.get("dataset"),
                "backbone": row.get("backbone"),
                "fusion_policy": row.get("fusion_policy"),
                "bundle_id": row.get("bundle_id"),
                "layer_idx": row.get("layer_idx"),
                "k": row.get("k"),
                "evidence_dim": row.get("evidence_dim"),
                "metric": row.get("metric", 0.0),
                "metric_name": _metric_name(row),
                "status": row.get("status", "ok"),
                **row,
            }
        )
    return out


def _normalize_local_suite_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        out.append(
            {
                "benchmark_family": "architecture",
                "benchmark_source": "local_suite",
                "baseline_name": row.get("model_type"),
                "task": row.get("task"),
                "dataset": row.get("dataset"),
                "backbone": row.get("backbone"),
                "fusion_policy": row.get("fusion_policy"),
                "bundle_id": row.get("bundle_id"),
                "layer_idx": row.get("layer_idx"),
                "k": row.get("k"),
                "evidence_dim": row.get("evidence_dim"),
                "metric": row.get("metric", 0.0),
                "metric_name": _metric_name(row),
                "status": row.get("status", "ok"),
                **row,
            }
        )
    return out


def _normalize_api_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        out.append(
            {
                "benchmark_family": "api",
                "benchmark_source": "api_suite",
                "baseline_name": row.get("model_type"),
                "task": row.get("task_family"),
                "dataset": row.get("dataset"),
                "backbone": row.get("model_id"),
                "fusion_policy": None,
                "bundle_id": None,
                "layer_idx": None,
                "k": 0,
                "evidence_dim": 0,
                "metric": row.get("metric", 0.0),
                "metric_name": _metric_name(row),
                "status": row.get("status", "ok"),
                **row,
            }
        )
    return out


def run_unified_benchmark(cfg: UnifiedBenchmarkConfig) -> List[Dict]:
    rows: List[Dict] = []

    if cfg.task_matrix_config:
        tm_cfg = load_task_matrix_config(cfg.task_matrix_config)
        rows.extend(_normalize_task_matrix_rows(run_task_matrix(tm_cfg)))

    if cfg.local_baseline_config:
        local_cfg = load_baseline_suite_config(cfg.local_baseline_config)
        rows.extend(_normalize_local_suite_rows(run_baseline_suite(local_cfg)))

    if cfg.api_baseline_config:
        api_cfg = load_api_baseline_config(cfg.api_baseline_config)
        rows.extend(_normalize_api_rows(run_api_baseline_suite(api_cfg)))

    _save_rows(rows, cfg.out_csv, cfg.out_json)
    return rows


def load_unified_benchmark_config(path: str) -> UnifiedBenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return UnifiedBenchmarkConfig(**raw)
