# Residual Policy based on smolVLA

## Installation
```bash
git submodule update --init --recursive
uv sync --all-extras
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

## Training
```bash
python src/lerobot/scripts/train_residualact.py --policy.type residualact --policy.repo_id test_residual --batch_size 64 --steps 100000 --dataset.repo_id k1000dai/libero --output_dir output --wandb.enable True
```