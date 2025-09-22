import argparse
import json
import random
from pathlib import Path
import tomllib
from torchvision import datasets

from .inference import run_inference, build_transform


def load_configs(data_json_path: str, model_toml_path: str):
    with open(data_json_path, "r") as f:
        data_cfg = json.load(f)
    with open(model_toml_path, "rb") as f:
        model_cfg = tomllib.load(f)
    return data_cfg, model_cfg



def pick_inference_dir(data_source: dict):
    root = Path(data_source["path"])
    if (root / "test").exists():
        return root / "test"
    if (root / "train").exists():  # fallback if only train exists
        return root / "train"
    return root

def balanced_sample_indices(dataset, limit):
    if limit is None or limit < 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    # group by class
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)
    for lst in class_to_indices.values():
        random.shuffle(lst)
    # round-robin draw
    ordered = []
    while len(ordered) < limit:
        progressed = False
        for lbl, lst in class_to_indices.items():
            if lst:
                ordered.append(lst.pop())
                progressed = True
                if len(ordered) >= limit:
                    break
        if not progressed:  # exhausted
            break
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Unified config-driven inference")
    parser.add_argument("--data-config", default="configs/data_config.json")
    parser.add_argument("--model-config", default="configs/model_params.toml")
    parser.add_argument("--arch", default=None, help="Override architecture (else use defaults.architecture)")
    args = parser.parse_args()

    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)
    defaults = model_cfg.get("defaults", {})
    arch = args.arch or defaults.get("architecture", "resnet34")
    params = model_cfg.get(arch, {})

    data_source = data_cfg["data_source"]
    infer_dir = pick_inference_dir(data_source)

    prep = data_cfg.get("preprocessing", {})
    resize = prep.get("resize", [224, 224])
    norm = prep.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})


    base_dataset = datasets.ImageFolder(str(infer_dir))
    indices = balanced_sample_indices(base_dataset, defaults.get("limit", -1))
    from torch.utils.data import Subset
    sampled = Subset(base_dataset, indices)
    base_dataset.transform = build_transform(resize=resize, mean=norm["mean"], std=norm["std"])

    # Transform already applied to base_dataset; sampled will reference it.

    batch_size = defaults.get("batch_size", 16)
    limit = defaults.get("limit", -1)

    preds, classes = run_inference(model_name=arch, dataset=sampled, batch_size=batch_size, limit=None, device=None,
                                   resize=resize, mean=norm["mean"], std=norm["std"]) 

    output_dir = Path(defaults.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{data_source.get('name','dataset')}.json"

    result = {
        "architecture": arch,
        "params": params,
        "classes": classes,
        "count": len(preds),
        "limit_applied": limit if limit != -1 else None,
        "predictions": preds[:10],
        "source_dir": str(infer_dir),
        "class_count_config": data_source.get("num_classes"),
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
