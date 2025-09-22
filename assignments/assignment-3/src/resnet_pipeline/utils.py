import json
import sys
import tomllib
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random

from .inference import build_transform


def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return tomllib.load(f)


def validate_dataset_path(data_source: Dict[str, Any]) -> None:
    path = Path(data_source['path'])
    name = data_source.get('name', 'dataset')
    
    if not path.exists() or not any(p.is_dir() for p in path.iterdir()):
        print(f"Dataset path '{path}' not found or empty.")
        if 'cifar' in name.lower():
            print("Download with: uv run download-cifar10")
        elif 'mnist' in name.lower():
            print("Download with: uv run download-mnist")
        else:
            print("Available: uv run download-cifar10 | uv run download-mnist")
        sys.exit(1)


def pick_train_test_dirs(root: Path) -> Tuple[Path, Path]:
    train, test = root / 'train', root / 'test'
    if train.exists() and test.exists():
        return train, test
    return (train if train.exists() else root), (test if test.exists() else root)


def pick_inference_dir(data_source: Dict[str, Any]) -> Path:
    root = Path(data_source['path'])
    for subdir in ['test', 'train']:
        if (root / subdir).exists():
            return root / subdir
    return root


def stratified_subset(dataset, per_class: int, seed: int = 0):
    if per_class == -1:
        return dataset
    random.seed(seed)
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)
    selected = []
    for idxs in class_indices.values():
        random.shuffle(idxs)
        selected.extend(idxs[:per_class])
    return Subset(dataset, selected)


def balanced_sample_indices(dataset, limit: int):
    if limit is None or limit < 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    for lst in class_indices.values():
        random.shuffle(lst)
    ordered = []
    while len(ordered) < limit and any(class_indices.values()):
        for lst in class_indices.values():
            if lst and len(ordered) < limit:
                ordered.append(lst.pop())
    return ordered


def create_dataset_with_transform(data_dir: Path, data_cfg: Dict[str, Any]) -> datasets.ImageFolder:
    prep = data_cfg.get('preprocessing', {})
    resize = prep.get('resize', [224, 224])
    norm = prep.get('normalization', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    transform = build_transform(resize=resize, mean=norm['mean'], std=norm['std'])
    return datasets.ImageFolder(str(data_dir), transform=transform)


def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def get_dataset_classes(dataset):
    base = getattr(dataset, 'dataset', dataset)
    classes = getattr(base, 'classes', None)
    if not classes:
        raise AttributeError("Dataset lacks 'classes' attribute")
    return classes