import argparse
from pathlib import Path

from . import cifar10, mnist

_DATASET_LOADERS = {
    "cifar10": (cifar10.download_cifar10, cifar10.update_data_config),
    "mnist": (mnist.download_mnist, mnist.update_data_config),
}


def _download_dataset(name: str, data_dir: Path):
    download_fn, update_fn = _DATASET_LOADERS[name]
    download_fn(data_dir=str(data_dir))
    update_fn(str(data_dir), name)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Download and prepare sample datasets for the ResNet pipeline."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory to store downloaded datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--cifar10",
        action="store_true",
        help="Download the CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--mnist",
        action="store_true",
        help="Download the MNIST dataset.",
    )

    args = parser.parse_args(argv)
    
    if not (args.cifar10 or args.mnist):
        parser.error("Select at least one dataset to download: --cifar10 or --mnist.")
    
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.cifar10:
        _download_dataset("cifar10", data_dir)

    if args.mnist:
        _download_dataset("mnist", data_dir)


if __name__ == "__main__":
    main()