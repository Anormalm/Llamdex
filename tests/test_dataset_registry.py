import json


def test_dataset_registry_loads():
    from src.multimodal.data import load_dataset_registry

    reg = load_dataset_registry("conf/datasets.registry.json")
    names = {item["name"] for item in reg["datasets"]}
    assert {
        "oxford_pet",
        "dtd",
        "gqa",
        "vqa_v2",
        "resisc45",
        "fgvc_aircraft",
        "mimic_cxr_jpg",
        "chexpert",
        "chexpert_plus",
        "medical_cxr_vqa",
        "medical_diff_vqa",
        "private_doc_ocr_synth",
        "privacy_risk",
    } <= names


def test_build_dataset_manifest_contains_required_fields():
    from src.multimodal.data import build_dataset_manifest

    entry = {
        "name": "oxford_pet",
        "track": "open",
        "tasks": ["single_image", "population"],
        "download_mode": "torchvision",
        "source_url": "https://example.com",
        "license": "dataset-specific",
        "access": "public",
        "ingest_contract": {"modality": "image"},
    }
    manifest = build_dataset_manifest(entry, root_dir="/tmp/oxford_pet", download_status="downloaded", artifacts=[])
    assert manifest["name"] == "oxford_pet"
    assert manifest["download_status"] == "downloaded"
    assert manifest["root_dir"] == "/tmp/oxford_pet"
    assert manifest["schema_version"] == "dataset-manifest-v1"
    assert manifest["record_count"] == 0
    assert manifest["split_counts"] == {}


def test_cross_validate_registry_and_freeze_passes():
    from src.multimodal.data import cross_validate_registry_and_freeze, validate_dataset_registry

    assert validate_dataset_registry("conf/datasets.registry.json") == []
    assert cross_validate_registry_and_freeze("conf/datasets.registry.json", "conf/task_matrix.freeze.v1.json") == []
