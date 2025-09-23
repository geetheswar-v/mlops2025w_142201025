import os
import json

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .helper.data import balanced_subset_indices


def train_model(
    train_dataset,
    test_dataset,
    model: nn.Module,
    training_cfg: dict,
    data_cfg: dict,
    save: bool = True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    limit = training_cfg.get("limit", -1)
    if limit != -1:
        train_indices = balanced_subset_indices(train_dataset, limit)
        test_indices = balanced_subset_indices(test_dataset, limit)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    batch_size = training_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lr = training_cfg.get("lr", 0.001)
    momentum = training_cfg.get("momentum", 0.0)
    optimizer_name = training_cfg.get("optimizer", "sgd").lower()
    epochs = training_cfg.get("epochs", 10)

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    epoch_metrics = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        epoch_loss = running_loss / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Record metrics for this epoch
        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "train_accuracy": train_acc
        })

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    if hasattr(test_dataset, "dataset"):
        class_names = test_dataset.dataset.classes
    else:
        class_names = test_dataset.classes

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Sample predictions
    sample_predictions = []
    indices = np.random.choice(len(all_preds), min(20, len(all_preds)), replace=False)
    for i in indices:
        sample_predictions.append({
            "pred": class_names[all_preds[i]],
            "true": class_names[all_labels[i]]
        })

    # Prepare JSON results
    results = {
        "epochs": epoch_metrics,
        "test_accuracy": test_acc,
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "classes": class_names,
        "sample_predictions": sample_predictions
    }

    if save:
        output_dir = training_cfg.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        data_name = data_cfg.get("name", "dataset")
        architecture = training_cfg.get("architecture", "model")

        # Save model
        model_file = os.path.join(output_dir, f"{data_name}_{architecture}_trained.pth")
        torch.save(model.state_dict(), model_file)
        print(f"Trained model saved to {model_file}")

        # Save JSON results
        json_file = os.path.join(output_dir, f"{data_name}_{architecture}_training.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Training results saved to {json_file}")

    return model, results