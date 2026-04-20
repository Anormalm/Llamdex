import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Ops hardening: verify required artifacts and triage failed rows.")
    p.add_argument("--csv", type=str, default=None, help="Metrics CSV artifact path.")
    p.add_argument("--json", type=str, default=None, help="Metrics JSON artifact path.")
    p.add_argument("--log", type=str, default=None, help="Run log path.")
    p.add_argument("--out_json", type=str, default=None, help="Optional output JSON report path.")
    return p.parse_args()


def _status(path: str) -> Dict:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }


def _triage_from_csv(path: Path) -> List[Dict]:
    failures: List[Dict] = []
    if not path.exists():
        return failures
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("status", "ok")).lower() != "ok":
                failures.append(
                    {
                        "task": row.get("task"),
                        "dataset": row.get("dataset"),
                        "baseline_mode": row.get("baseline_mode") or row.get("model_type"),
                        "status": row.get("status"),
                        "error": row.get("error", ""),
                    }
                )
    return failures


def main():
    a = parse_args()
    checks = []
    if a.csv:
        checks.append(_status(a.csv))
    if a.json:
        checks.append(_status(a.json))
    if a.log:
        checks.append(_status(a.log))

    csv_failures = _triage_from_csv(Path(a.csv)) if a.csv else []
    out = {
        "checks": checks,
        "all_present_nonempty": all(c["exists"] and c["size_bytes"] > 0 for c in checks) if checks else False,
        "failure_count": len(csv_failures),
        "failures": csv_failures,
    }

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    if not out["all_present_nonempty"]:
        raise SystemExit(2)
    if out["failure_count"] > 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
