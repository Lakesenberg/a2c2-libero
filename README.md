# Residual Policy based on smolVLA

## Installation
```bash
git submodule update --init --recursive
uv venv -p 3.10
uv pip install -e ".[smolvla]"
```
# Install libero and mujoco for evaluation
```bash
uv pip install -e third_party/libero
uv pip install mujoco==3.2.3
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```
### Docker 
```bash
cd docker
(chmod +x initialize-docker-container.sh BUILD_DOCKER_IMAGE.sh RUN_DOCKER_CONTAINER.sh )
./BUILD_DOCKER_IMAGE.sh
./RUN_DOCKER_CONTAINER.sh
```

## Training
### smolVLA
```bash
python src/lerobot/scripts/train.py  \
--policy.type=smolvla   \
--policy.load_vlm_weights True  \
--dataset.repo_id=k1000dai/libero \
--batch_size=64 \
--steps=100000 \
--policy.repo_id=k1000dai/smolvla_libero_scratch \
--output_dir=libero_smolvla_scratch  \
--job_name=libero_smolvla_scratch \
--wandb.enable=true
```
### Convert Data for Residual ACT
```bash
python eval_libero/create_residual_dataset.py 
```
See the [create_residual_dataset.py](eval_libero/create_residual_dataset.py) for more details. Need to specify the base dataset path, base policy path and new dataset path.

### Residual ACT
```bash
python src/lerobot/scripts/train_residualact.py \ 
--policy.type residualact \
--policy.repo_id test_residual \
--batch_size 64 \
--steps 100000 \
--dataset.repo_id k1000dai/libero-addinfo \
--output_dir output_residualact \
--wandb.enable True
```

## Evaluation
```bash
MUJOCO_GL=glx python eval_libero/evaluation_residual.py 
```