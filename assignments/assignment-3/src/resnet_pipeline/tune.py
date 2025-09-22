import argparse
import json
import time
from pathlib import Path
import tomllib
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .inference import build_transform
from .models import prepare_model


def load_grid(grid_path: str):
    with open(grid_path, 'r') as f:
        return json.load(f)


def load_data_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_model_config(path: str):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def get_inference_dir(data_source: dict) -> Path:
    root = Path(data_source['path'])
    if (root / 'train').exists():
        return root / 'train'
    return root


def stratified_sample(dataset, per_class: int = 32, seed: int = 42):
    random.seed(seed)
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)
    sampled = []
    for lbl, idxs in class_to_indices.items():
        random.shuffle(idxs)
        sampled.extend(idxs[:per_class])
    return Subset(dataset, sampled)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            pred_labels = preds.argmax(1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='Simple grid search (3e)')
    parser.add_argument('--grid', default='configs/hparam_grid.json')
    parser.add_argument('--data-config', default='configs/data_config.json')
    parser.add_argument('--model-config', default='configs/model_params.toml')
    parser.add_argument('--output', default='outputs/tuning_results.json')
    parser.add_argument('--per-class-train', type=int, default=32)
    parser.add_argument('--per-class-val', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    grid = load_grid(args.grid)
    data_cfg = load_data_config(args.data_config)
    model_cfg = load_model_config(args.model_config)

    arch = grid.get('architecture', model_cfg.get('defaults', {}).get('architecture', 'resnet34'))

    data_source = data_cfg['data_source']
    infer_dir = get_inference_dir(data_source)
    prep = data_cfg.get('preprocessing', {})
    resize = prep.get('resize', [224, 224])
    norm = prep.get('normalization', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

    transform = build_transform(resize=resize, mean=norm['mean'], std=norm['std'])
    full_dataset = datasets.ImageFolder(str(infer_dir), transform=transform)

    # stratified train/val sample from the same source (for demonstration)
    train_subset = stratified_sample(full_dataset, per_class=args.per_class_train, seed=0)
    val_subset = stratified_sample(full_dataset, per_class=args.per_class_val, seed=1)

    train_loader = DataLoader(train_subset, batch_size=model_cfg.get('defaults', {}).get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    results = []

    for lr in grid['learning_rates']:
        for opt_name in grid['optimizers']:
            for mom in grid['momentums']:
                start = time.time()
                model = prepare_model(arch, device=device, num_classes=len(full_dataset.classes), pretrained=True)
                # Optimizer setup
                if opt_name.lower() == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
                else:
                    # Adam ignores momentum; include for grid completeness
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                # Train
                for _ in range(args.epochs):
                    train_one_epoch(model, train_loader, criterion, optimizer, device)
                # Evaluate
                val_acc = evaluate(model, val_loader, device)
                elapsed = time.time() - start
                results.append({
                    'architecture': arch,
                    'lr': lr,
                    'optimizer': opt_name,
                    'momentum': mom,
                    'val_accuracy': val_acc,
                    'train_samples': len(train_subset),
                    'val_samples': len(val_subset),
                    'seconds': round(elapsed, 2)
                })
                print(f"lr={lr} opt={opt_name} mom={mom} val_acc={val_acc:.4f} time={elapsed:.1f}s")

    # Rank by val_accuracy desc then time asc
    results.sort(key=lambda r: (-r['val_accuracy'], r['seconds']))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"Saved tuning results to {args.output}")


if __name__ == '__main__':
    main()
