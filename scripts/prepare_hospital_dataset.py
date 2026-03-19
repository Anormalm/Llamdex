import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.multimodal.data import (
    build_dataset_manifest,
    load_jsonl,
    load_label_ontology,
    load_text_privacy_policy,
    normalize_label_record,
    summarize_split_counts,
    validate_hospital_record,
    validate_text_privacy_record,
)


def parse_args():
    p = argparse.ArgumentParser(description="Normalize hospital/private records into the schema-compliant JSONL release format.")
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--manifest_out", type=str, default=None)
    p.add_argument("--rejects_out", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="hospital_dataset_v1")
    p.add_argument("--track", type=str, default="medical")
    p.add_argument("--source_url", type=str, default="local/private")
    p.add_argument("--ontology", type=str, default="conf/label_ontology.medical.v1.json")
    p.add_argument("--label_set", type=str, default=None)
    p.add_argument("--privacy_policy", type=str, default="conf/text_privacy_mode.v1.json")
    p.add_argument("--question_template", type=str, default="Based on the sanitized description, answer Yes or No: {label_name}?")
    p.add_argument("--default_split", type=str, default="train")
    p.add_argument("--default_modality", type=str, default="text")
    p.add_argument("--min_description_chars", type=int, default=24)
    return p.parse_args()


def _read_rows(path: str) -> List[Dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        raise ValueError("JSON input must contain a top-level list.")
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {suffix}")


def _choose_first(record: Dict, keys: Iterable[str], default=""):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _normalize_row(record: Dict, ontology: Dict, args) -> Dict:
    row = dict(record)
    row = normalize_label_record(row, ontology, label_set=args.label_set)

    description = str(_choose_first(row, ["description", "text", "report", "summary"], default="")).strip()
    split = str(_choose_first(row, ["split", "partition"], default=args.default_split)).strip().lower()
    if split == "valid":
        split = "val"
    question = str(_choose_first(row, ["question"], default="")).strip()
    if not question:
        question = args.question_template.format(
            label_name=row["label_name"],
            label_code=row["label_code"],
        )

    normalized = {
        "record_id": str(_choose_first(row, ["record_id", "id", "study_id", "example_id"], default="")),
        "patient_hash": str(_choose_first(row, ["patient_hash", "patient_id_hash", "subject_id"], default="")),
        "study_hash": str(_choose_first(row, ["study_hash", "study_id_hash", "study_uid"], default="")),
        "split": split or args.default_split,
        "question": question,
        "answer": str(_choose_first(row, ["answer"], default=row["label_name"])),
        "label_code": str(row["label_code"]),
        "label_name": str(row["label_name"]),
        "label_id": int(row["label_id"]),
        "description": description,
        "modality": str(_choose_first(row, ["modality"], default=args.default_modality)),
        "deid_status": str(_choose_first(row, ["deid_status"], default="deidentified")),
        "quality_pass": bool(_choose_first(row, ["quality_pass"], default=bool(len(description) >= args.min_description_chars))),
        "source_dataset": str(_choose_first(row, ["source_dataset", "dataset"], default=args.dataset_name)),
        "notes": str(_choose_first(row, ["notes"], default="")),
    }
    if not normalized["record_id"]:
        normalized["record_id"] = f"{normalized['source_dataset']}::{normalized['split']}::{normalized['patient_hash'] or normalized['study_hash'] or normalized['label_code']}"
    return normalized


def main():
    args = parse_args()
    rows = _read_rows(args.input_path)
    ontology = load_label_ontology(args.ontology)
    policy = load_text_privacy_policy(args.privacy_policy)
    policy["description_min_chars"] = int(args.min_description_chars)

    accepted: List[Dict] = []
    rejects: List[Dict] = []
    for raw in rows:
        try:
            normalized = _normalize_row(raw, ontology, args)
        except Exception as exc:
            rejects.append({"record": raw, "errors": [f"normalization_failed: {exc}"]})
            continue

        errors = []
        errors.extend(validate_hospital_record(normalized))
        errors.extend(validate_text_privacy_record(normalized, policy=policy))
        if errors:
            rejects.append({"record": normalized, "errors": errors})
            continue
        accepted.append(normalized)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    payload_bytes = Path(args.output_jsonl).read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    label_distribution: Dict[str, int] = {}
    for row in accepted:
        key = str(row["label_code"])
        label_distribution[key] = label_distribution.get(key, 0) + 1

    if args.rejects_out:
        os.makedirs(os.path.dirname(args.rejects_out) or ".", exist_ok=True)
        with open(args.rejects_out, "w", encoding="utf-8") as f:
            json.dump(rejects, f, indent=2)

    manifest_out = args.manifest_out or str(Path(args.output_jsonl).with_suffix(".manifest.v1.json"))
    manifest = build_dataset_manifest(
        {
            "name": args.dataset_name,
            "track": args.track,
            "tasks": ["yesno", "grounded_generation"],
            "download_mode": "local_ingest",
            "source_url": args.source_url,
            "license": "private/owner-controlled",
            "access": "restricted",
            "ingest_contract": {
                "schema_path": "conf/hospital_ingest.schema.json",
                "privacy_policy_path": args.privacy_policy,
                "ontology_path": args.ontology,
            },
        },
        root_dir=str(Path(args.output_jsonl).parent),
        download_status="prepared",
        artifacts=[
            {"path": args.output_jsonl, "type": "normalized_jsonl", "records": len(accepted), "sha256": payload_sha256},
            {"path": manifest_out, "type": "manifest"},
        ],
        notes=f"accepted={len(accepted)} rejected={len(rejects)}",
        record_count=len(accepted),
        split_counts=summarize_split_counts(accepted),
        label_distribution=label_distribution,
        release_status="v1_candidate",
    )
    manifest["privacy_policy"] = policy
    manifest["ontology_version"] = ontology.get("version", "unknown")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(accepted)} accepted rows.")
    print(f"Rejected {len(rejects)} rows.")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Manifest: {manifest_out}")
    if args.rejects_out:
        print(f"Rejects: {args.rejects_out}")


if __name__ == "__main__":
    main()
