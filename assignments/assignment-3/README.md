# Assignment 3

| Roll Number | Name                  |
|-------------|-----------------------|
| 142201025   | Vedachalam Geetheswar |

## Overview
Learning Json/Toml and Configuration system through this assignment

- Assignment Answers (non-coding/informative): [Assignment.md](./Assignment.md)

## Inference (Question 3a)

Run pretrained ResNet (34/50/101/152) on a folder structured like `ImageFolder`.

### Install (editable) using uv

```bash
uv sync
```

### Example: CIFAR10 airplane test subset

```bash
# (optional) download dataset structure already present in repo
uv run inference --model resnet34 --data-dir data/cifar10/test --limit 8 --batch-size 4
```

### Options

```bash
uv run inference \
	--model resnet50 \
	--data-dir data/cifar10/test \
	--batch-size 16 \
	--limit 32 \
	--output preds.json
```

Produces JSON (if --output provided):

```json
{
	"classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
	"predictions": [
		{"index":0, "pred_class":"airplane", "confidence":0.75, "true_class":"airplane"}
	]
}
```

Limit -1 processes entire dataset.

## Config Pipeline (Questions 3b, 3c, 3d)

JSON (`configs/data_config.json`) specifies dataset path, split, architecture.
TOML (`configs/model_params.toml`) specifies per-architecture parameters.

Run integrated pipeline (reads both configs):

```bash
uv run run_inference \
	--data-config configs/data_config.json \
	--model-config configs/model_params.toml \
	--output cfg_preds.json
```

Override architecture without editing JSON:
```bash
uv run run_inference --override-arch resnet50
```

Difference:
- `uv run inference` : direct manual arguments (3a)
- `uv run run_inference` : config-driven (3d)