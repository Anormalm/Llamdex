import argparse

from src.multimodal.data import (
    cross_validate_registry_and_freeze,
    load_text_privacy_policy,
    validate_dataset_registry,
    validate_dataset_manifest,
    validate_hospital_jsonl,
    validate_label_ontology,
    validate_task_matrix_freeze,
    validate_text_privacy_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="Validate dataset/task-matrix contract files.")
    p.add_argument("--hospital_jsonl", type=str, default=None)
    p.add_argument("--dataset_manifest", type=str, default=None)
    p.add_argument("--dataset_registry", type=str, default="conf/datasets.registry.json")
    p.add_argument("--open_ontology", type=str, default="conf/label_ontology.open.v1.json")
    p.add_argument("--medical_ontology", type=str, default="conf/label_ontology.medical.v1.json")
    p.add_argument("--privacy_policy", type=str, default="conf/text_privacy_mode.v1.json")
    p.add_argument("--task_matrix_freeze", type=str, default="conf/task_matrix.freeze.v1.json")
    return p.parse_args()


def main():
    args = parse_args()
    errors = []
    privacy_policy = load_text_privacy_policy(args.privacy_policy) if args.privacy_policy else None
    if args.hospital_jsonl:
        errors.extend(validate_hospital_jsonl(args.hospital_jsonl))
        errors.extend(validate_text_privacy_jsonl(args.hospital_jsonl, policy=privacy_policy))
    if args.dataset_manifest:
        errors.extend(validate_dataset_manifest(args.dataset_manifest))
    if args.dataset_registry:
        errors.extend(validate_dataset_registry(args.dataset_registry))
    if args.task_matrix_freeze:
        errors.extend(validate_task_matrix_freeze(args.task_matrix_freeze))
    if args.open_ontology:
        errors.extend(validate_label_ontology(args.open_ontology))
    if args.medical_ontology:
        errors.extend(validate_label_ontology(args.medical_ontology))
    if args.dataset_registry and args.task_matrix_freeze:
        errors.extend(cross_validate_registry_and_freeze(args.dataset_registry, args.task_matrix_freeze))
    if errors:
        for err in errors:
            print(err)
        raise SystemExit(1)
    print("Contracts valid.")


if __name__ == "__main__":
    main()
