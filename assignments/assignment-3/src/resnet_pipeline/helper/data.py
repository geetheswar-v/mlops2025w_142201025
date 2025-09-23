import os
import numpy as np
from collections import defaultdict

from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset, Subset


def get_transforms(preprocessing: dict | None) -> transforms.Compose:
    transform_list = []

    if preprocessing is None:
        return transforms.ToTensor()

    if "resize" in preprocessing:
        w, h = preprocessing["resize"]
        transform_list.append(transforms.Resize((w, h)))
    
    transform_list.append(transforms.ToTensor())

    if "normalization" in preprocessing:
        mean = preprocessing["normalization"]["mean"]
        std = preprocessing["normalization"]["std"]
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)

def prepare_dataset(
    data_source: dict,
    preprocessing: dict | None,
    split_ratio: float = 0.8,
    only_test: bool = False
) -> dict[str, Dataset]:
    dataset_path = data_source["path"]
    transform = get_transforms(preprocessing)

    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    else:
        full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
        train_size = int(split_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    if only_test:
        return {"test": test_dataset}

    return {"train": train_dataset, "test": test_dataset}

def balanced_subset_indices(dataset: Dataset | Subset, limit: int) -> list[int]:
    if limit == -1 or limit >= len(dataset):
        return list(range(len(dataset)))
    
    if hasattr(dataset, "dataset"):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
        base_indices = dataset.indices
    else:
        targets = np.array(dataset.targets)
        base_indices = np.arange(len(dataset))
    
    class_to_indices = defaultdict(list)
    for idx, t in zip(base_indices, targets):
        class_to_indices[t].append(idx)
    
    num_classes = len(class_to_indices)
    per_class_limit = limit // num_classes
    
    selected_indices = []
    for cls, idxs in class_to_indices.items():
        np.random.shuffle(idxs)
        selected_indices.extend(idxs[:per_class_limit])
    
    # If leftover samples (due to integer division), add randomly
    remaining = limit - len(selected_indices)
    if remaining > 0:
        leftover = []
        for idxs in class_to_indices.values():
            leftover.extend(idxs[per_class_limit:])
        np.random.shuffle(leftover)
        selected_indices.extend(leftover[:remaining])
    
    np.random.shuffle(selected_indices)
    return selected_indices
