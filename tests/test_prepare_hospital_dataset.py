import json
import subprocess
import sys
from pathlib import Path


def test_prepare_hospital_dataset_normalizes_and_writes_manifest(tmp_path):
    src = tmp_path / "raw.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "case-1",
                        "split": "train",
                        "answer": "Yes",
                        "description": "De-identified summary of the case with no direct identifiers present.",
                        "deid_status": "deidentified",
                        "quality_pass": True,
                    }
                ),
                json.dumps(
                    {
                        "id": "case-2",
                        "split": "test",
                        "answer": "No",
                        "description": "short",
                        "deid_status": "deidentified",
                        "quality_pass": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "prepared.jsonl"
    rejects = tmp_path / "rejects.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_hospital_dataset.py",
            "--input_path",
            str(src),
            "--output_jsonl",
            str(out_jsonl),
            "--rejects_out",
            str(rejects),
            "--manifest_out",
            str(manifest),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    prepared = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(prepared) == 1
    assert prepared[0]["label_code"] == "Y"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["split_counts"] == {"train": 1}
    assert manifest_payload["record_count"] == 1
    assert manifest_payload["label_distribution"] == {"Y": 1}
    assert manifest_payload["artifacts"][0]["sha256"]
    assert "privacy_qa_summary" in manifest_payload
    reject_payload = json.loads(rejects.read_text(encoding="utf-8"))
    assert len(reject_payload) == 1
    assert "qa" in reject_payload[0]
