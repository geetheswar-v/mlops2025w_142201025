import sys
from pathlib import Path

from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


def download_cifar10(data_dir: str = "data"):
    print("Downloading CIFAR-10 dataset")
    
    # Create main data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create CIFAR-10 specific directory
    cifar10_path = data_path / "cifar10"
    cifar10_path.mkdir(exist_ok=True)
    
    # CIFAR-10 class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Create train and test directories
    train_dir = cifar10_path / "train"
    test_dir = cifar10_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Create class directories for both train and test
    for class_name in class_names:
        (train_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
    
    print("Setting up datasets")
    
    # Download CIFAR-10 datasets
    transform = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(
        root=str(data_path / "temp_cifar10"),
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=str(data_path / "temp_cifar10"),
        train=False,
        download=False,
        transform=transform
    )
    
    print("Extracting training images")
    train_counts = {class_name: 0 for class_name in class_names}
    
    for idx, (image, label) in enumerate(tqdm(train_dataset, desc="Training images")):
        # Convert tensor to numpy array and then to PIL Image
        img_array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        # Save image to appropriate class folder
        class_name = class_names[label]
        img_filename = f"{train_counts[class_name]:05d}.png"
        img_path = train_dir / class_name / img_filename
        img.save(img_path)
        
        train_counts[class_name] += 1
    
    print("Extracting test images")
    test_counts = {class_name: 0 for class_name in class_names}
    
    for idx, (image, label) in enumerate(tqdm(test_dataset, desc="Test images")):
        # Convert tensor to numpy array and then to PIL Image
        img_array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        # Save image to appropriate class folder
        class_name = class_names[label]
        img_filename = f"{test_counts[class_name]:05d}.png"
        img_path = test_dir / class_name / img_filename
        img.save(img_path)
        
        test_counts[class_name] += 1
    
    print(f"\nDataset extracted to: {cifar10_path}")
    print("\nClass distribution:")
    print("Training set:")
    for class_name in class_names:
        print(f"  {class_name}: {train_counts[class_name]} images")
    
    print("Test set:")
    for class_name in class_names:
        print(f"  {class_name}: {test_counts[class_name]} images")
    
    # Clean up temporary directory
    import shutil
    temp_dir = data_path / "temp_cifar10"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def update_data_config(data_dir: str, dataset_name: str = "cifar10"):
    """Update data_config.json with the downloaded dataset path."""
    import json
    from pathlib import Path
    
    config_path = Path("configs/data_config.json")
    if not config_path.exists():
        print(f"Config file {config_path} not found, skipping update.")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['data_source']['name'] = dataset_name
        config['data_source']['path'] = str(Path(data_dir) / dataset_name)
        config['data_source']['num_classes'] = 10
        config['data_source']['input_shape'] = [32, 32, 3]
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Updated {config_path} with dataset path: {config['data_source']['path']}")
    except Exception as e:
        print(f"Warning: Could not update config file: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and extract CIFAR-10 dataset")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to save the dataset (default: data)"
    )
    parser.add_argument(
        "--update-data-config",
        action="store_true",
        default=True,
        help="Update data_config.json with dataset path (default: True)"
    )
    parser.add_argument(
        "--no-update-data-config",
        dest="update_data_config",
        action="store_false",
        help="Skip updating data_config.json"
    )
    
    args = parser.parse_args()
    
    try:
        download_cifar10(data_dir=args.data_dir)
        if args.update_data_config:
            update_data_config(args.data_dir, "cifar10")
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()