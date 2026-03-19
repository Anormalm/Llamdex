from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.multimodal.task_matrix.runner import TaskMatrixConfig, run_task_matrix


@dataclass
class ExpertSweepSpec:
    name: str
    expert_type: str
    expert_model_id: Optional[str] = None
    expert_model_path: Optional[str] = None
    expert_output_dim: Optional[int] = None
    enabled: bool = True


@dataclass
class ExpertSweepConfig:
    base_task_matrix_config: str
    experts: List[ExpertSweepSpec] = field(
        default_factory=lambda: [
            ExpertSweepSpec(name="clip", expert_type="clip", expert_model_id="openai/clip-vit-base-patch32", expert_output_dim=512),
            ExpertSweepSpec(name="siglip", expert_type="siglip", expert_model_id="google/siglip-base-patch16-224", expert_output_dim=768),
            ExpertSweepSpec(name="dinov2", expert_type="dinov2", expert_model_id="facebook/dinov2-base", expert_output_dim=768),
        ]
    )
    out_csv: str = "/disk1/lfhu/runs/expert_sweep.csv"
    out_json: str = "/disk1/lfhu/runs/expert_sweep.json"


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


def _load_task_matrix_config(path: str) -> TaskMatrixConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    from src.multimodal.task_matrix.runner import load_task_matrix_config

    return load_task_matrix_config(path)


def run_expert_sweep(cfg: ExpertSweepConfig) -> List[Dict]:
    base_cfg = _load_task_matrix_config(cfg.base_task_matrix_config)
    rows: List[Dict] = []

    for spec in cfg.experts:
        if not spec.enabled:
            continue
        try:
            run_cfg = TaskMatrixConfig(**base_cfg.__dict__)
            run_cfg.expert_type = spec.expert_type
            if spec.expert_model_id is not None:
                run_cfg.expert_model_id = spec.expert_model_id
            if spec.expert_model_path is not None:
                run_cfg.expert_model_path = spec.expert_model_path
            if spec.expert_output_dim is not None:
                run_cfg.expert_output_dim = int(spec.expert_output_dim)
            stem = os.path.splitext(os.path.basename(cfg.out_csv))[0]
            out_dir = os.path.dirname(cfg.out_csv)
            run_cfg.out_csv = os.path.join(out_dir, f"{stem}.{spec.name}.csv")
            run_cfg.out_json = os.path.join(out_dir, f"{stem}.{spec.name}.json")
            subrows = run_task_matrix(run_cfg)
            for row in subrows:
                row = dict(row)
                row["expert_sweep_name"] = spec.name
                row["expert_type"] = spec.expert_type
                row["expert_model_id"] = spec.expert_model_id
                row["expert_model_path"] = spec.expert_model_path
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "expert_sweep_name": spec.name,
                    "expert_type": spec.expert_type,
                    "expert_model_id": spec.expert_model_id,
                    "expert_model_path": spec.expert_model_path,
                    "metric": 0.0,
                    "status": "error",
                    "error": str(exc) or repr(exc),
                }
            )
        _save_rows(rows, cfg.out_csv, cfg.out_json)

    return rows


def load_expert_sweep_config(path: str) -> ExpertSweepConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "base_config" in raw and "base_task_matrix_config" not in raw:
        raw["base_task_matrix_config"] = raw.pop("base_config")
    experts = [ExpertSweepSpec(**x) for x in raw.pop("experts", [])]
    cfg = ExpertSweepConfig(**raw)
    if experts:
        cfg.experts = experts
    return cfg
