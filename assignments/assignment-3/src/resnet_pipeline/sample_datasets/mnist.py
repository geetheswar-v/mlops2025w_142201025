import sys
from pathlib import Path

from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


def download_mnist(data_dir: str = "data"):
    print("Downloading MNIST dataset")
    
    # Create main data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create MNIST specific directory
    mnist_path = data_path / "mnist"
    mnist_path.mkdir(exist_ok=True)
    
    # Create train and test directories
    train_dir = mnist_path / "train"
    test_dir = mnist_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Create class directories (0-9) for both train and test
    for i in range(10):
        (train_dir / str(i)).mkdir(exist_ok=True)
        (test_dir / str(i)).mkdir(exist_ok=True)
    
    print("Setting up datasets")
    
    # Download MNIST datasets
    transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(
        root=str(data_path / "temp_mnist"),
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=str(data_path / "temp_mnist"),
        train=False,
        download=False,
        transform=transform
    )
    
    print("Extracting training images")
    train_counts = {str(i): 0 for i in range(10)}
    
    for idx, (image, label) in enumerate(tqdm(train_dataset, desc="Training images")):
        # Convert tensor to numpy array and then to PIL Image
        img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Save image to appropriate class folder
        class_label = str(label)
        img_filename = f"{train_counts[class_label]:05d}.png"
        img_path = train_dir / class_label / img_filename
        img.save(img_path)
        
        train_counts[class_label] += 1
    
    print("Extracting test images")
    test_counts = {str(i): 0 for i in range(10)}
    
    for idx, (image, label) in enumerate(tqdm(test_dataset, desc="Test images")):
        # Convert tensor to numpy array and then to PIL Image
        img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Save image to appropriate class folder
        class_label = str(label)
        img_filename = f"{test_counts[class_label]:05d}.png"
        img_path = test_dir / class_label / img_filename
        img.save(img_path)
        
        test_counts[class_label] += 1
    
    print(f"\nDataset extracted to: {mnist_path}")
    print("\nClass distribution:")
    print("Training set:")
    for class_label in range(10):
        print(f"  Class {class_label}: {train_counts[str(class_label)]} images")
    
    print("Test set:")
    for class_label in range(10):
        print(f"  Class {class_label}: {test_counts[str(class_label)]} images")
    
    # Clean up temporary directory
    import shutil
    temp_dir = data_path / "temp_mnist"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def update_data_config(data_dir: str, dataset_name: str = "mnist"):
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
        config['data_source']['input_shape'] = [28, 28, 1]
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Updated {config_path} with dataset path: {config['data_source']['path']}")
    except Exception as e:
        print(f"Warning: Could not update config file: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and extract MNIST dataset")
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
        download_mnist(data_dir=args.data_dir)
        if args.update_data_config:
            update_data_config(args.data_dir, "mnist")
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
