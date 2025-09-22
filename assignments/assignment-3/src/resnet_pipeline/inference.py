import argparse
from pathlib import Path
import json

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import prepare_model


def build_transform(resize=(224, 224), mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(tuple(resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def resolve_actual_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    # If user passed a split (e.g., root/train) that's fine
    if path.is_dir():
        # If this dir has image files directly, wrap by creating temp? Simpler: require class folders.
        return path
    raise FileNotFoundError(f"{data_dir} is not a directory")


def load_dataset(data_dir: str, resize=(224, 224), mean=None, std=None):
    base = resolve_actual_dir(data_dir)
    tfm = build_transform(resize=resize, mean=mean, std=std)
    dataset = datasets.ImageFolder(root=str(base), transform=tfm)
    return dataset


def run_inference(model_name: str, data_dir: str | None = None, batch_size: int = 16, limit: int | None = None, device: str | None = None, dataset=None,
                  resize=(224, 224), mean=None, std=None):
    if dataset is None:
        if data_dir is None:
            raise ValueError("Either dataset or data_dir must be provided")
        dataset = load_dataset(data_dir, resize=resize, mean=mean, std=std)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Support Subset carrying underlying ImageFolder
    base = getattr(dataset, 'dataset', dataset)
    classes = getattr(base, 'classes', None)
    if classes is None:
        raise AttributeError("Provided dataset lacks 'classes' attribute")
    num_classes = len(classes)
    model = prepare_model(model_name, device=device, num_classes=num_classes, pretrained=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    count = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Batches", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            for i in range(images.size(0)):
                item = {
                    "index": count,
                    "pred_class": classes[preds[i].item()],
                    "confidence": float(confs[i].item()),
                    "true_class": classes[labels[i].item()] if labels is not None else None
                }
                predictions.append(item)
                count += 1
                if limit is not None and count >= limit:
                    break
            if limit is not None and count >= limit:
                break
    return predictions, classes


def main():
    parser = argparse.ArgumentParser(description="Simple ResNet inference over an ImageFolder dataset")
    parser.add_argument("--model", required=True, choices=["resnet34", "resnet50", "resnet101", "resnet152"], help="ResNet architecture")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root, split (train/test), or direct class folders root")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=16, help="Number of images to process (default 16, -1 for all)")
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--output", default=None, help="Optional JSON file to save predictions")

    args = parser.parse_args()
    limit = None if args.limit == -1 else args.limit

    preds, classes = run_inference(
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=limit,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Classes: {classes}")
    for p in preds:
        print(p)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({"classes": classes, "predictions": preds}, f, indent=2)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
