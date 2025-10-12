from .helper.model import get_resnet
from .train import train_model
import torch
from torch.utils.data import Dataset
import os
import gc


def _state_dict_to_cpu(model: torch.nn.Module) -> dict:
    """Detach and move model state dict to CPU for safe serialization."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def grid_search(grid_list: dict, train_dataset: Dataset, test_dataset: Dataset, data_cfg: dict):
    best_test_acc = 0.0
    best_cfg = None
    best_state_dict = None
    best_results = None
    
    for cfg in grid_list:
        print(f"\nRunning grid search for {cfg['architecture']} with params: "
              f"lr={cfg.get('lr')}, optimizer={cfg.get('optimizer')}, "
              f"momentum={cfg.get('momentum')}, epochs={cfg.get('epochs')}, "
              f"batch_size={cfg.get('batch_size')}")
    
        # Clear GPU memory before loading new model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
        try:
            # Load model
            model = get_resnet(cfg["architecture"], num_classes=data_cfg.get("num_classes"))
        
            # Train model
            trained_model, results = train_model(train_dataset, test_dataset, model, cfg, data_cfg, save=False)
        
            test_acc = results["test_accuracy"]
            state_dict_cpu = _state_dict_to_cpu(trained_model)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_cfg = cfg
                best_state_dict = state_dict_cpu
                best_results = results

            # Clean up GPU memory after training
            del trained_model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory for {cfg['architecture']} with batch_size={cfg.get('batch_size')}. Skipping this configuration.")
            print(f"Error: {e}")
            # Clean up and continue with next configuration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"Error training {cfg['architecture']}: {e}")
            # Clean up and continue with next configuration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

    if best_cfg is None or best_state_dict is None:
        print("No valid configurations were evaluated. Skipping model save.")
        return
            
    # Save best model
    output_dir = best_cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    best_model_file = os.path.join(output_dir, f"{data_cfg['name']}_best_model.pth")
    torch.save(best_state_dict, best_model_file)

    if best_results is not None:
        best_result_file = os.path.join(output_dir, f"{data_cfg['name']}_best_results.json")
        with open(best_result_file, "w") as f:
            import json

            json.dump(best_results, f, indent=4)
        print(f"Best results saved at {best_result_file}")
    
    print(f"\nBest model saved at {best_model_file}")
    print(f"Best architecture: {best_cfg['architecture']}, Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"Best hyperparameters: {best_cfg}")