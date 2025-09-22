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
    pick_train_test_dirs, stratified_subset, create_dataset_with_transform,
    evaluate_model
)


def train_loop(model, loader, criterion, optimizer, device, epochs):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='Config-driven training')
    parser.add_argument('--data-config', default='configs/data_config.json')
    parser.add_argument('--model-config', default='configs/model_params.toml')
    parser.add_argument('--arch', default=None)
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_cfg = load_json_config(args.data_config)
    model_cfg = load_toml_config(args.model_config)

    data_source = data_cfg['data_source']
    validate_dataset_path(data_source)

    defaults = model_cfg.get('defaults', {})
    arch = args.arch or defaults.get('architecture', 'resnet34')
    arch_params = model_cfg.get(arch, {})

    root = Path(data_source['path'])
    train_dir, test_dir = pick_train_test_dirs(root)
    full_train = create_dataset_with_transform(train_dir, data_cfg)
    full_test = create_dataset_with_transform(test_dir, data_cfg)

    train_subset = stratified_subset(full_train, defaults.get('per_class_train', 32), 0)
    val_subset = stratified_subset(full_train, defaults.get('per_class_val', 16), 1)

    batch_size = defaults.get('batch_size', 32)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    model = prepare_model(arch, device=device, num_classes=len(full_train.classes), pretrained=True)
    
    lr = arch_params.get('learning_rate', 0.001)
    if arch_params.get('optimizer', 'adam').lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=arch_params.get('momentum', 0.0))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    train_loop(model, train_loader, nn.CrossEntropyLoss(), optimizer, device, defaults.get('epochs', 1))
    val_acc = evaluate_model(model, val_loader, device)
    test_acc = evaluate_model(model, test_loader, device)
    elapsed = round(time.time() - start, 2)

    output_dir = Path(defaults.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        torch.save(model.state_dict(), output_dir / f"{data_source.get('name','model')}_{arch}.pt")

    metrics = {
        'architecture': arch,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'seconds': elapsed
    }
    
    with open(output_dir / 'train_result.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {elapsed}s")


if __name__ == '__main__':
    main()
