import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .models import prepare_model
from .utils import (
    load_json_config, load_toml_config, validate_dataset_path,
    pick_inference_dir, stratified_subset, create_dataset_with_transform,
    evaluate_model, get_dataset_classes
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='Grid search tuning')
    parser.add_argument('--grid', default='configs/hparam_grid.json')
    parser.add_argument('--data-config', default='configs/data_config.json')
    parser.add_argument('--model-config', default='configs/model_params.toml')
    parser.add_argument('--output', default='outputs/tuning_results.json')
    parser.add_argument('--per-class-train', type=int, default=32)
    parser.add_argument('--per-class-val', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(args.grid, 'r') as f:
        grid = json.load(f)
    data_cfg = load_json_config(args.data_config)
    model_cfg = load_toml_config(args.model_config)

    dataset_path = validate_dataset_path(data_cfg['data_source']['path'])
    arch = grid.get('architecture', model_cfg.get('defaults', {}).get('architecture', 'resnet34'))

    infer_dir = pick_inference_dir(dataset_path)
    full_dataset = create_dataset_with_transform(infer_dir, data_cfg['preprocessing'])
    num_classes = len(get_dataset_classes(full_dataset))

    train_subset = stratified_subset(full_dataset, args.per_class_train, seed=0)
    val_subset = stratified_subset(full_dataset, args.per_class_val, seed=1)

    batch_size = model_cfg.get('defaults', {}).get('batch_size', 32)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    results = []

    for lr in grid['learning_rates']:
        for opt_name in grid['optimizers']:
            for mom in grid['momentums']:
                start = time.time()
                model = prepare_model(arch, device=device, num_classes=num_classes, pretrained=True)
                
                if opt_name.lower() == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                
                for _ in range(args.epochs):
                    train_one_epoch(model, train_loader, criterion, optimizer, device)
                
                val_acc = evaluate_model(model, val_loader, device)
                elapsed = round(time.time() - start, 2)
                
                results.append({
                    'architecture': arch,
                    'lr': lr,
                    'optimizer': opt_name,
                    'momentum': mom,
                    'val_accuracy': val_acc,
                    'seconds': elapsed
                })
                print(f"lr={lr} opt={opt_name} mom={mom} val_acc={val_acc:.4f} time={elapsed}s")

    results.sort(key=lambda r: (-r['val_accuracy'], r['seconds']))
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"Saved tuning results to {args.output}")
