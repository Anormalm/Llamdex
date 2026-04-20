import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _read_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _run(cmd: List[str], cwd: str) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return ""


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _detect_branch(repo_root: str, override: Optional[str]) -> str:
    return override or _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root) or "unknown"


def _detect_commit(repo_root: str, override: Optional[str]) -> str:
    return override or _run(["git", "rev-parse", "--short", "HEAD"], repo_root) or "unknown"


def _python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


def _library_versions() -> str:
    versions = []
    for name in ("torch", "transformers", "accelerate"):
        try:
            mod = __import__(name)
            versions.append(str(getattr(mod, "__version__", "unknown")))
        except Exception:
            versions.append("unavailable")
    return " / ".join(versions)


def _timestamp_string(override: Optional[str]) -> str:
    if override:
        return override
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %z")


def _collect_result_rows(
    task_rows: List[Dict[str, str]],
    grounded_rows: List[Dict[str, str]],
    unified_rows: List[Dict[str, str]],
    api_rows: List[Dict[str, str]],
    review: Optional[Dict],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for row in unified_rows:
        metric_name = str(row.get("metric_name") or row.get("metric") or "metric")
        metric_value = row.get("metric_value", row.get("metric", ""))
        rows.append(
            {
                "task": str(row.get("task", "")),
                "model_system": str(row.get("baseline_name") or row.get("backbone") or row.get("model_type") or ""),
                "dataset": str(row.get("dataset", "")),
                "metric": metric_name,
                "value": str(metric_value),
                "latency": str(row.get("latency", "-") or "-"),
                "status": str(row.get("status", "ok")),
            }
        )

    if not rows:
        for row in task_rows:
            task = str(row.get("task", "")).strip()
            if not task:
                continue
            metric = "mae" if _to_float(row.get("mae")) is not None else "accuracy"
            value = row.get(metric, "")
            if task == "grounded_generation" and _to_float(row.get("f1")) is not None:
                metric = "accuracy/f1"
                value = f"{row.get('accuracy', '')} / {row.get('f1', '')}"
            rows.append(
                {
                    "task": task,
                    "model_system": str(row.get("baseline_mode") or row.get("backbone") or ""),
                    "dataset": str(row.get("dataset", "")),
                    "metric": metric,
                    "value": str(value),
                    "latency": str(row.get("latency", "-") or "-"),
                    "status": str(row.get("status", "ok")),
                }
            )

    for row in grounded_rows:
        if str(row.get("task", "")).strip() != "grounded_generation":
            continue
        for metric_name in ("rationale_consistency", "format_compliance", "rationale_nonempty_rate"):
            value = row.get(metric_name)
            if _to_float(value) is None:
                continue
            rows.append(
                {
                    "task": "grounded_generation",
                    "model_system": str(row.get("baseline_mode") or row.get("backbone") or "grounded_report"),
                    "dataset": str(row.get("dataset", "")),
                    "metric": metric_name,
                    "value": str(value),
                    "latency": "-",
                    "status": str(row.get("status", "ok")),
                }
            )

    if api_rows and not unified_rows:
        for row in api_rows:
            rows.append(
                {
                    "task": str(row.get("task", "api_baseline")),
                    "model_system": str(row.get("model_type") or row.get("model_id") or "api"),
                    "dataset": str(row.get("dataset", "")),
                    "metric": str(row.get("metric_name") or "metric"),
                    "value": str(row.get("metric", row.get("metric_value", ""))),
                    "latency": str(row.get("latency", "-") or "-"),
                    "status": str(row.get("status", "ok")),
                }
            )

    if review:
        rows.append(
            {
                "task": "pilot_gate",
                "model_system": "Go/No-Go review",
                "dataset": "benchmark packet",
                "metric": "decision",
                "value": str(review.get("decision", "")),
                "latency": "-",
                "status": "ok",
            }
        )

    return rows


def build_report(args) -> Dict:
    task_rows = _read_csv(args.task_matrix_csv)
    grounded_rows = _read_csv(args.grounded_csv)
    unified_rows = _read_csv(args.unified_csv)
    api_rows = _read_csv(args.api_summary_csv)
    review = _read_json(args.go_no_go_json)
    commands = _read_text(args.commands_file)

    repo_root = os.path.abspath(args.repo_root)
    report = {
        "title": f"Remote Status Quo ({args.report_date})",
        "repo_root": repo_root,
        "branch": _detect_branch(repo_root, args.branch),
        "commit": _detect_commit(repo_root, args.commit),
        "python_version": args.python_version or _python_version(),
        "library_versions": args.library_versions or _library_versions(),
        "timestamp": _timestamp_string(args.timestamp),
        "commands": commands,
        "results": _collect_result_rows(task_rows, grounded_rows, unified_rows, api_rows, review),
        "failures": list(args.failure or []),
        "next_plan": list(args.next_step or []),
        "artifacts": list(args.artifact or []),
        "context": {
            "task_rows": len(task_rows),
            "grounded_rows": len(grounded_rows),
            "unified_rows": len(unified_rows),
            "api_rows": len(api_rows),
            "go_no_go_decision": None if not review else review.get("decision"),
        },
    }
    return report


def _render_md(report: Dict) -> str:
    lines = []
    lines.append(f"# {report['title']}")
    lines.append("")
    lines.append("## Commit and Branch")
    lines.append(f"- Repo: `{report['repo_root']}`")
    lines.append(f"- Branch: `{report['branch']}`")
    lines.append(f"- Commit: `{report['commit']}`")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Python: `{report['python_version']}`")
    lines.append(f"- torch / transformers / accelerate: `{report['library_versions']}`")
    lines.append(f"- Timestamp: `{report['timestamp']}`")
    lines.append("")
    lines.append("## Commands Executed")
    if report["commands"]:
        lines.append("```bash")
        lines.append(report["commands"])
        lines.append("```")
    else:
        lines.append("- none recorded")
    lines.append("")
    lines.append("## Result Table")
    lines.append("| task | model/system | dataset | metric | value | latency | status |")
    lines.append("|---|---|---|---|---:|---:|---|")
    if report["results"]:
        for row in report["results"]:
            lines.append(
                f"| {row['task']} | {row['model_system']} | {row['dataset']} | {row['metric']} | {row['value']} | {row['latency']} | {row['status']} |"
            )
    else:
        lines.append("| none | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Failures / Root Cause / Fix")
    if report["failures"]:
        for idx, failure in enumerate(report["failures"], start=1):
            lines.append(f"{idx}. {failure}")
    else:
        lines.append("1. No failures recorded.")
    lines.append("")
    lines.append("## Artifacts Produced")
    if report["artifacts"]:
        for item in report["artifacts"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none recorded")
    lines.append("")
    lines.append("## Next 72-Hour Plan")
    if report["next_plan"]:
        for idx, step in enumerate(report["next_plan"], start=1):
            lines.append(f"{idx}. {step}")
    else:
        lines.append("1. Add next steps before publication.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Generate a playbook-compliant remote status markdown report.")
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--report_date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--task_matrix_csv", type=str, default=None)
    p.add_argument("--grounded_csv", type=str, default=None)
    p.add_argument("--unified_csv", type=str, default=None)
    p.add_argument("--api_summary_csv", type=str, default=None)
    p.add_argument("--go_no_go_json", type=str, default=None)
    p.add_argument("--commands_file", type=str, default=None)
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--failure", action="append", default=[])
    p.add_argument("--next_step", action="append", default=[])
    p.add_argument("--branch", type=str, default=None)
    p.add_argument("--commit", type=str, default=None)
    p.add_argument("--python_version", type=str, default=None)
    p.add_argument("--library_versions", type=str, default=None)
    p.add_argument("--timestamp", type=str, default=None)
    p.add_argument("--out_md", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    report = build_report(args)
    _ensure_parent_dir(args.out_md)
    _ensure_parent_dir(args.out_json)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(report))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"MD: {args.out_md}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
