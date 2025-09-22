import argparse
import json
import random
from pathlib import Path
import time
import tomllib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .inference import build_transform
from .models import prepare_model


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def load_toml(path: str):
    with open(path, 'rb') as f:
        return tomllib.load(f)

def pick_dirs(root: Path):
    train_dir = root / 'train'
    test_dir = root / 'test'
    if train_dir.exists() and test_dir.exists():
        return train_dir, test_dir
    # fallback: single folder acts as both; we'll split later if needed
    return root, root

def stratified_subset(dataset, per_class: int, seed: int = 0):
    if per_class == -1:
        return dataset
    random.seed(seed)
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)
    selected = []
    for lbl, idxs in class_to_indices.items():
        random.shuffle(idxs)
        selected.extend(idxs[:per_class])
    return Subset(dataset, selected)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0

def train_loop(model, train_loader, criterion, optimizer, device, epochs: int):
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def main():
    parser = argparse.ArgumentParser(description='Config-driven training + evaluation (extended 3d)')
    parser.add_argument('--data-config', default='configs/data_config.json')
    parser.add_argument('--model-config', default='configs/model_params.toml')
    parser.add_argument('--arch', default=None)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--output', default=None, help='Optional metrics JSON path (default outputs/train_result.json)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_cfg = load_json(args.data_config)
    model_cfg = load_toml(args.model_config)

    defaults = model_cfg.get('defaults', {})
    arch = args.arch or defaults.get('architecture', 'resnet34')

    arch_params = model_cfg.get(arch, {})
    lr = arch_params.get('learning_rate', 0.001)
    optimizer_name = arch_params.get('optimizer', 'adam').lower()
    momentum = arch_params.get('momentum', 0.0)

    batch_size = defaults.get('batch_size', 32)
    epochs = defaults.get('epochs', 1)
    per_class_train = defaults.get('per_class_train', 32)
    per_class_val = defaults.get('per_class_val', 16)

    data_source = data_cfg['data_source']
    root = Path(data_source['path'])
    train_dir, test_dir = pick_dirs(root)

    prep = data_cfg.get('preprocessing', {})
    resize = prep.get('resize', [224, 224])
    norm = prep.get('normalization', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    transform = build_transform(resize=resize, mean=norm['mean'], std=norm['std'])

    full_train = datasets.ImageFolder(str(train_dir), transform=transform)
    full_test = datasets.ImageFolder(str(test_dir), transform=transform)

    train_subset = stratified_subset(full_train, per_class_train, seed=0)
    val_subset = stratified_subset(full_train, per_class_val, seed=1)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    model = prepare_model(arch, device=device, num_classes=len(full_train.classes), pretrained=True)

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    start = time.time()
    train_loop(model, train_loader, criterion, optimizer, device, epochs)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    elapsed = round(time.time() - start, 2)

    output_dir = Path(defaults.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        model_path = output_dir / f"{data_source.get('name','model')}_{arch}.pt"
        torch.save({'state_dict': model.state_dict(), 'arch': arch}, model_path)
        print(f"Saved model to {model_path}")

    metrics = {
        'architecture': arch,
        'learning_rate': lr,
        'optimizer': optimizer_name,
        'momentum': momentum,
        'epochs': epochs,
        'train_samples': len(train_subset) if isinstance(train_subset, Subset) else len(full_train),
        'val_samples': len(val_subset) if isinstance(val_subset, Subset) else len(full_train),
        'test_samples': len(full_test),
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'seconds': elapsed
    }

    metrics_path = args.output or str(output_dir / 'train_result.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(f"Validation Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f} | Time: {elapsed}s")


if __name__ == '__main__':
    main()
