import os
import json

import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .helper.data import balanced_subset_indices

def run_inference(
    test_dataset,
    model: torch.nn.Module,
    inference_cfg: dict,
    data_cfg: dict
):
    device = next(model.parameters()).device
    model.eval()

    limit = inference_cfg.get("limit", -1)
    if limit != -1:
        indices = balanced_subset_indices(test_dataset, limit)
        test_dataset = Subset(test_dataset, indices)

    batch_size = inference_cfg.get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if hasattr(test_dataset, "dataset"):
        class_names = test_dataset.dataset.classes
    else:
        class_names = test_dataset.classes

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    print(f"Inference Accuracy: {accuracy*100:.2f}%")

    # Sample predictions
    sample_predictions = []
    indices = np.random.choice(len(all_preds), min(20, len(all_preds)), replace=False)
    for i in indices:
        sample_predictions.append({
            "pred": class_names[all_preds[i]],
            "true": class_names[all_labels[i]]
        })

    results = {
        "accuracy": accuracy,
        "num_samples": total,
        "classes": class_names,
        "sample_predictions": sample_predictions
    }

    output_dir = inference_cfg.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    data_name = data_cfg.get("name", "dataset")
    architecture = inference_cfg.get("architecture", "model")
    output_file = os.path.join(output_dir, f"{data_name}_{architecture}_inference.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Inference results saved to {output_file}")
    return results
