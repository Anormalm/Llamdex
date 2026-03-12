import argparse
import csv
import os
from typing import Dict, List


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _read_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_local_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        if str(r.get("status", "")).lower() != "ok":
            continue
        metric_name = r.get("metric_name") or ("accuracy" if r.get("accuracy") is not None else "metric")
        metric = _to_float(r.get("metric", r.get(metric_name, 0.0)))
        out.append(
            {
                "source": "local",
                "model_type": str(r.get("model_type", "")),
                "dataset": str(r.get("dataset", "")),
                "metric_name": str(metric_name),
                "metric": metric,
                "metric_ci95": 0.0,
                "latency": _to_float(r.get("inference_latency", 0.0)),
                "failure_rate": 0.0,
                "status": "ok",
            }
        )
    return out


def _normalize_api_summary_rows(rows: List[Dict], max_failure_rate: float) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        fr = _to_float(r.get("failure_rate", 1.0), 1.0)
        if fr > max_failure_rate:
            continue
        metric = _to_float(r.get("metric_mean", 0.0))
        out.append(
            {
                "source": "api",
                "model_type": str(r.get("model_type", "")),
                "dataset": str(r.get("dataset", "")),
                "metric_name": str(r.get("metric_name", "metric")),
                "metric": metric,
                "metric_ci95": _to_float(r.get("metric_ci95", 0.0)),
                "latency": _to_float(r.get("latency_mean", 0.0)),
                "failure_rate": fr,
                "status": "ok" if fr <= max_failure_rate else "filtered",
            }
        )
    return out


def _rank(rows: List[Dict]) -> List[Dict]:
    by_ds: Dict[str, List[Dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    ranked: List[Dict] = []
    for ds, gr in by_ds.items():
        gr_sorted = sorted(gr, key=lambda x: x["metric"], reverse=True)
        for i, row in enumerate(gr_sorted, start=1):
            x = dict(row)
            x["rank"] = i
            ranked.append(x)
    return ranked


def _write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Merge local (own model) baselines with API summary baselines into one ranked leaderboard.")
    p.add_argument("--local_csv", type=str, required=True, help="CSV from local benchmark suite (e.g., runs/baseline_suite_expanded.csv).")
    p.add_argument("--api_summary_csv", type=str, required=True, help="CSV from API summary (e.g., runs/api_baseline_suite_strong.summary.csv).")
    p.add_argument("--max_failure_rate", type=float, default=0.0, help="Keep only API rows with failure_rate <= threshold.")
    p.add_argument("--out_csv", type=str, default="runs/own_vs_api_leaderboard.csv")
    return p.parse_args()


def main():
    a = parse_args()
    local_rows = _normalize_local_rows(_read_csv(a.local_csv))
    api_rows = _normalize_api_summary_rows(_read_csv(a.api_summary_csv), a.max_failure_rate)
    merged = _rank(local_rows + api_rows)
    _write_csv(a.out_csv, merged)
    print(f"Saved leaderboard: {a.out_csv}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()

