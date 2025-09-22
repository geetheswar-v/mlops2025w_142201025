import argparse
import json
from pathlib import Path
from torch.utils.data import Subset

from .inference import run_inference
from .utils import (
    load_json_config, load_toml_config, validate_dataset_path,
    pick_inference_dir, balanced_sample_indices, create_dataset_with_transform
)


def main():
    parser = argparse.ArgumentParser(description="Run inference with configs")
    parser.add_argument("--data-config", default="configs/data_config.json")
    parser.add_argument("--model-config", default="configs/model_params.toml")
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()

    data_cfg = load_json_config(args.data_config)
    model_cfg = load_toml_config(args.model_config)
    defaults = model_cfg.get("defaults", {})
    arch = args.arch or defaults.get("architecture", "resnet34")

    data_source = data_cfg["data_source"]
    validate_dataset_path(data_source)
    
    base_dataset = create_dataset_with_transform(pick_inference_dir(data_source), data_cfg)
    sampled = Subset(base_dataset, balanced_sample_indices(base_dataset, defaults.get("limit", -1)))

    prep = data_cfg.get("preprocessing", {})
    preds, classes = run_inference(
        model_name=arch, dataset=sampled, batch_size=defaults.get("batch_size", 16),
        limit=None, device=None, resize=prep.get("resize", [224, 224]),
        mean=prep.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]),
        std=prep.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
    )

    output_dir = Path(defaults.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "architecture": arch,
        "params": model_cfg.get(arch, {}),
        "classes": classes,
        "count": len(preds),
        "predictions": preds[:10],
    }
    
    out_file = output_dir / f"{data_source.get('name','dataset')}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
