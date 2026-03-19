import argparse
import json
from pathlib import Path

from torchvision import datasets, transforms

from src.multimodal.data import build_dataset_manifest, load_dataset_registry


AUTO_DOWNLOADERS = {
    "oxford_pet": lambda root: datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=True,
        transform=transforms.ToTensor(),
    ),
    "dtd": lambda root: datasets.DTD(
        root=root,
        split="train",
        download=True,
        transform=transforms.ToTensor(),
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap open-track and manual-access datasets from the dataset registry.")
    p.add_argument("--registry", type=str, default="conf/datasets.registry.json")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--names", nargs="*", default=None, help="Optional dataset names to bootstrap.")
    return p.parse_args()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _manual_notice(entry: dict, dataset_dir: Path):
    notice = dataset_dir / "MANUAL_DOWNLOAD_REQUIRED.txt"
    notice.write_text(
        "\n".join(
            [
                f"Dataset: {entry['name']}",
                f"Source: {entry['source_url']}",
                f"Access: {entry['access']}",
                f"License: {entry['license']}",
                "",
                "Download this dataset manually according to the source instructions,",
                "then update the manifest status from pending_manual_download to ready_for_ingestion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    registry = load_dataset_registry(args.registry)
    requested = set(args.names or [])
    for entry in registry["datasets"]:
        if requested and entry["name"] not in requested:
            continue
        dataset_dir = Path(args.data_root) / entry["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        status = "pending_manual_download"
        notes = ""
        if entry["download_mode"] == "torchvision":
            AUTO_DOWNLOADERS[entry["name"]](str(dataset_dir))
            status = "downloaded"
            notes = "Bootstrapped via torchvision download helpers."
        else:
            _manual_notice(entry, dataset_dir)
            artifacts.append({"path": str(dataset_dir / "MANUAL_DOWNLOAD_REQUIRED.txt"), "type": "notice"})
            notes = "Manual-access dataset scaffolded; download must be completed out of band."

        manifest = build_dataset_manifest(
            entry=entry,
            root_dir=str(dataset_dir),
            download_status=status,
            artifacts=artifacts,
            notes=notes,
        )
        _write_json(dataset_dir / "manifest.v1.json", manifest)
        print(f"{entry['name']}: {status} -> {dataset_dir}")


if __name__ == "__main__":
    main()
