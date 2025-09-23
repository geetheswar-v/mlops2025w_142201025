import argparse, sys
from pathlib import Path
from typing import Iterable

from .helper.config import data_config, inference_config, load_hparam_grid, training_config
from .helper.data import prepare_dataset
from .helper.model import get_resnet
from .inference import run_inference
from .train import train_model
from .tune import grid_search

DEFAULTS = {
    "data": Path("configs/data_config.json"),
    "model": Path("configs/model_params.toml"),
    "grid": Path("configs/hparam_grid.json"),
}

def _resolve(path): 
    return Path(path).expanduser().resolve()

def _prepare(cfg, only_test=False):
    data_cfg, prep = data_config(str(cfg))
    return data_cfg, prepare_dataset(data_cfg, prep, only_test=only_test)

def train_pipeline(data_cfg_p, model_cfg_p):
    data_cfg, dsets = _prepare(data_cfg_p); 
    train, test = dsets["train"], dsets["test"]
    
    cfg = training_config(str(model_cfg_p))
    
    model = get_resnet(cfg["architecture"], data_cfg["num_classes"], cfg.get("pretrained", True))

    train_model(train, test, model, cfg, data_cfg, save=True)

def inference_pipeline(data_cfg_p, model_cfg_p):
    data_cfg, dsets = _prepare(data_cfg_p, only_test=True); 
    test = dsets["test"]
    
    cfg = inference_config(str(model_cfg_p))
    model = get_resnet(cfg["architecture"], data_cfg["num_classes"], cfg.get("pretrained", False))
    run_inference(test, model, cfg, data_cfg)

def tune_pipeline(data_cfg_p, model_cfg_p, grid_p):
    data_cfg, dsets = _prepare(data_cfg_p); 
    train, test = dsets["train"], dsets["test"]
    
    grid = load_hparam_grid(str(grid_p), str(model_cfg_p))
    
    if not grid: 
        raise ValueError("Empty hyperparameter grid.")
    
    grid_search(grid, train, test, data_cfg)

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="ResNet Training/Inference/Tuning")
    parser.add_argument("--data-config", default=str(DEFAULTS["data"]))
    parser.add_argument("--model-config", default=str(DEFAULTS["model"]))
    parser.add_argument("--grid-config", default=str(DEFAULTS["grid"]))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args(argv)
    if not (args.train or args.inference or args.tune): parser.error("Need --train or --inference or --tune")
    try:
        if args.train: 
            train_pipeline(_resolve(args.data_config), _resolve(args.model_config))
            
        if args.inference: 
            inference_pipeline(_resolve(args.data_config), _resolve(args.model_config))
        if args.tune: 
            tune_pipeline(_resolve(args.data_config), _resolve(args.model_config), _resolve(args.grid_config))
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr); raise SystemExit(1) from e

def main_train():
    main(["--train"])

def main_inference():
    main(["--inference"])

def main_tune():
    main(["--tune"])

if __name__ == "__main__": 
    main()
