import json
import tomllib
import itertools
import os

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The Config {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_toml(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The Config {file_path} does not exist.")
    
    with open(file_path, 'rb') as file:
        data = tomllib.load(file)
    return data


def check_data_source(data_source: dict) -> bool:
    if not os.path.exists(data_source["path"]):
        print(f"Data source {data_source['name']} does not exist.")
        if "mnist" in data_source["name"].lower():
            print("use uv run download --mnist to download the sample data")
        elif "cifar10" in data_source["name"].lower():
            print("use uv run download --cifar10 to download the sample data")
        else:
            print("Data source is not in sample datasets.")
            print("Need to be manually downloaded or")
            print("use uv run download --mnist or uv run download --cifar10 to download the sample data")
        return False
    return True


import os
import json

def data_config(config_path: str) -> tuple[dict, dict | None]:
    config = load_json(config_path)
    data_source = config.get("data_source")
    if data_source is None:
        raise ValueError("data_source not found in the configuration.")

    path = data_source["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data source {path} does not exist.")

    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "test")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        classes = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
    else:
        classes = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]

    data_source.setdefault("num_classes", len(classes))
    data_source.setdefault("classes", classes)

    if not check_data_source(data_source):
        raise FileNotFoundError(f"Data source {data_source['name']} not found.")

    return data_source, config.get("preprocessing")

def training_config(config_path: str) -> dict:
    config = load_toml(config_path)
    model_cfg = config.get("model", {})
    default_cfg = config.get("defaults", {})
    training_cfg = config.get("training", {})
    if not model_cfg:
        raise ValueError("model configuration is empty.")

    training_cfg = {**default_cfg, **training_cfg, "architecture": model_cfg.get("architecture", "resnet34")}
    return training_cfg

def inference_config(config_path: str) -> dict:
    config = load_toml(config_path)
    model_cfg = config.get("model", {})
    default_cfg = config.get("defaults", {})
    inference_cfg = config.get("inference", {})
    if not model_cfg:
        raise ValueError("model configuration is empty.")

    inference_cfg = {**default_cfg, **inference_cfg, "architecture": model_cfg.get("architecture", "resnet34")}
    return inference_cfg

def model_hparms(config_path: str, architecture: str) -> dict:
    config = load_toml(config_path)
    model_cfg = config.get("model", {})
    default_cfg = config.get("defaults", {})
    tuning_cfg = config.get("tuning", {})
    if not model_cfg:
        raise ValueError("model configuration is empty.")

    arch_params = model_cfg.get(architecture, {})
    hparms = {**default_cfg,**arch_params, **tuning_cfg}
    return hparms

def load_hparam_grid(grid_path: str, model_config_path: str) -> list[dict]:
    with open(grid_path, "r") as f:
        grid = json.load(f)

    includes = grid.get("includes", [])
    override_params = grid.get("override_params", {})
    default_override = override_params.get("default", {})

    grid_list = []

    for arch in includes:
        toml_params = model_hparms(model_config_path, arch)

        full_params = {**default_override, **toml_params}

        arch_override = override_params.get(arch, {})
        full_params.update(arch_override)

        for k, v in full_params.items():
            if not isinstance(v, list):
                full_params[k] = [v]

        keys, values = zip(*full_params.items()) if full_params else ([], [])
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]

        for comb in combinations:
            comb["architecture"] = arch
            grid_list.append(comb)

    return grid_list

