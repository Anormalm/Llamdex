import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class HFImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        transform,
        cache_dir: Optional[str] = None,
        image_field: str = "image",
        label_field: str = "label",
    ):
        from datasets import ClassLabel, load_dataset

        self.ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.transform = transform
        self.image_field = image_field
        self.label_field = label_field
        feature = self.ds.features[label_field]
        if isinstance(feature, ClassLabel):
            self.classes = list(feature.names)
            self._label_to_idx = None
        else:
            labels = sorted({str(row[label_field]).strip() for row in self.ds})
            self.classes = labels
            self._label_to_idx = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        image = row[self.image_field]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        raw_label = row[self.label_field]
        label = int(raw_label) if self._label_to_idx is None else self._label_to_idx[str(raw_label).strip()]
        return image, label


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_classes_for_dataset(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "dtd":
        return 47
    if dataset_name == "oxford_pet":
        return 37
    if dataset_name == "resisc45":
        return 45
    if dataset_name == "fgvc_aircraft":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_model(num_classes: int) -> nn.Module:
    # Keep architecture aligned with src/multimodal/experts/vision_experts.py
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transforms(dataset_name: str, train: bool):
    if dataset_name in {"cifar10", "cifar100"}:
        if train:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    if dataset_name in {"dtd", "oxford_pet", "resisc45", "fgvc_aircraft"}:
        # DTD images are variable-sized; resize to a stable input shape.
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_datasets(dataset_name: str, data_root: str):
    train_tf = build_transforms(dataset_name, train=True)
    eval_tf = build_transforms(dataset_name, train=False)

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        eval_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
        return train_ds, eval_ds
    if dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
        eval_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_tf)
        return train_ds, eval_ds
    if dataset_name == "dtd":
        train_ds = datasets.DTD(root=data_root, split="train", download=True, transform=train_tf)
        eval_ds = datasets.DTD(root=data_root, split="test", download=True, transform=eval_tf)
        return train_ds, eval_ds
    if dataset_name == "oxford_pet":
        train_ds = datasets.OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=True, transform=train_tf)
        eval_ds = datasets.OxfordIIITPet(root=data_root, split="test", target_types="category", download=True, transform=eval_tf)
        return train_ds, eval_ds
    if dataset_name == "resisc45":
        train_ds = HFImageClassificationDataset("timm/resisc45", split="train", transform=train_tf, cache_dir=data_root)
        eval_ds = HFImageClassificationDataset("timm/resisc45", split="test", transform=eval_tf, cache_dir=data_root)
        return train_ds, eval_ds
    if dataset_name == "fgvc_aircraft":
        train_ds = datasets.FGVCAircraft(root=data_root, split="trainval", annotation_level="variant", download=True, transform=train_tf)
        eval_ds = datasets.FGVCAircraft(root=data_root, split="test", annotation_level="variant", download=True, transform=eval_tf)
        return train_ds, eval_ds
    raise ValueError(f"Unsupported dataset: {dataset_name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vision expert checkpoint for Plan-1 image datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dtd",
        choices=["cifar10", "cifar100", "dtd", "oxford_pet", "resisc45", "fgvc_aircraft"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_path", type=str, default="runs/experts/dtd_resnet18_best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True

    train_ds, eval_ds = build_datasets(args.dataset_name, args.data_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = build_model(num_classes_for_dataset(args.dataset_name)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"dataset={args.dataset_name} device={device} train={len(train_ds)} eval={len(eval_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        eval_acc = evaluate(model, eval_loader, device, use_amp)
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} eval_acc={eval_acc:.4f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), str(out_path))
            print(f"saved {out_path}")

    print(f"best_eval_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
