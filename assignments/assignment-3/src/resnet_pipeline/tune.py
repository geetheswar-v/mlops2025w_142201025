from .helper.model import get_resnet
from .train import train_model
import torch
from torch.utils.data import Dataset
import os


def grid_search(grid_list: dict, train_dataset: Dataset, test_dataset: Dataset, data_cfg: dict):
    best_test_acc = 0.0
    best_model = None
    best_cfg = None
    
    for cfg in grid_list:
        print(f"\nRunning grid search for {cfg['architecture']} with params: "
              f"lr={cfg.get('lr')}, optimizer={cfg.get('optimizer')}, "
              f"momentum={cfg.get('momentum')}, epochs={cfg.get('epochs')}")
    
        # Load model
        model = get_resnet(cfg["architecture"], num_classes=data_cfg.get("num_classes"))
    
        # Train model
        trained_model, results = train_model(train_dataset, test_dataset, model, cfg, data_cfg, save=False)
    
        test_acc = results["test_accuracy"]
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = trained_model
            best_cfg = cfg
            
    # Save best model
    output_dir = best_cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    best_model_file = os.path.join(output_dir, f"{data_cfg['name']}_best_model.pth")
    torch.save(best_model.state_dict(), best_model_file)
    
    print(f"\nBest model saved at {best_model_file}")
    print(f"Best architecture: {best_cfg['architecture']}, Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"Best hyperparameters: {best_cfg}")