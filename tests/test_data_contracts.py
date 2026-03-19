import json


def test_validate_hospital_record_accepts_valid_row():
    from src.multimodal.data import validate_hospital_record

    row = {
        "record_id": "r1",
        "split": "train",
        "question": "Is there edema?",
        "answer": "No",
        "label_code": "N",
        "label_name": "No",
        "description": "De-identified report summary.",
        "deid_status": "deidentified",
        "quality_pass": True,
    }
    assert validate_hospital_record(row) == []


def test_validate_hospital_jsonl_reports_missing_fields(tmp_path):
    from src.multimodal.data import validate_hospital_jsonl

    p = tmp_path / "hospital.jsonl"
    p.write_text(json.dumps({"record_id": "r1"}) + "\n", encoding="utf-8")
    errs = validate_hospital_jsonl(str(p))
    assert errs
    assert "missing fields" in errs[0]


def test_validate_task_matrix_freeze_passes():
    from src.multimodal.data import validate_task_matrix_freeze

    assert validate_task_matrix_freeze("conf/task_matrix.freeze.v1.json") == []


def test_validate_label_ontologies_pass():
    from src.multimodal.data import validate_label_ontology

    assert validate_label_ontology("conf/label_ontology.open.v1.json") == []
    assert validate_label_ontology("conf/label_ontology.medical.v1.json") == []


def test_validate_text_privacy_record_rejects_forbidden_fields():
    from src.multimodal.data import load_text_privacy_policy, validate_text_privacy_record

    row = {
        "record_id": "r1",
        "split": "train",
        "question": "Is there edema?",
        "answer": "No",
        "label_code": "N",
        "label_name": "No",
        "description": "De-identified report summary long enough for the policy gate.",
        "deid_status": "deidentified",
        "quality_pass": True,
        "raw_report": "forbidden raw content",
    }
    errs = validate_text_privacy_record(row, policy=load_text_privacy_policy("conf/text_privacy_mode.v1.json"))
    assert errs
    assert "forbidden field present: raw_report" in errs


def test_validate_dataset_manifest_passes(tmp_path):
    from src.multimodal.data import validate_dataset_manifest

    manifest = {
        "name": "demo",
        "track": "medical",
        "version": "v1",
        "schema_version": "dataset-manifest-v1",
        "created_at_utc": "2026-03-19T12:00:00Z",
        "tasks": ["yesno"],
        "source_url": "local/private",
        "root_dir": "/tmp/demo",
        "download_status": "prepared",
        "license": "private",
        "access": "restricted",
        "ingest_contract": {"schema_path": "conf/hospital_ingest.schema.json"},
        "artifacts": [{"path": "/tmp/demo.jsonl", "type": "normalized_jsonl"}],
        "record_count": 1,
        "split_counts": {"train": 1},
        "label_distribution": {"Y": 1},
        "release_status": "v1_candidate",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_dataset_manifest(str(path)) == []
