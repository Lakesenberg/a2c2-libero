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
uv pip install mujoco==3.3.2
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
### Residual Transformer
```bash
python src/lerobot/scripts/train_residual_transformer.py \
--policy.type residual_transformer \
--policy.repo_id k1000dai/residual_transformer_libero_spatial \
--batch_size 64 \
--num_workers 16 \
--steps 400000 \
--dataset.repo_id k1000dai/libero-spatial-smolvla \
--output_dir output_residual_transformer_spatial \
--job_name residual_transformer_libero_spatial \
--wandb.enable True
```
The script internally samples single time steps, caches language tokens, and trains a lightweight transformer head that predicts the final action conditioned on the base SmolVLA rollout. Ensure your residual dataset was generated with the helpers in `eval_libero/create_dataset_for_residualpolicy.py` (or the conveyor variant) so that it now stores both `vla_actions` and the SmolVLA VLM hidden vector (`vlm_hidden`) captured during rollout; older datasets should be regenerated. Adjust chunk size or language prompt caching in `src/lerobot/scripts/train_residual_transformer.py` if your dataset layout differs.

The current residual transformer consumes the entire base-policy action chunk at every step. During training the dataset yields a normalized `base_action_chunk` tensor alongside the single-step target; at inference time the evaluation scripts automatically forward the same chunk (and the corresponding sinusoidal `time_feature`) to the residual head. If you deploy a custom inference loop, remember to include:

1. The base policy action you intend to execute (`action[:, 0]`).
2. The full chunk predicted by the base policy under the key `base_action_chunk`.
3. The phase feature matching `train_residual_transformer.py` (sine/cosine of the chunk index).

Without these inputs the transformer will fall back to the base action token only and refinement quality will degrade.

### Residual MLP
```bash
python src/lerobot/scripts/train_residual_transformer.py \
--policy.type residual_mlp \
--policy.repo_id k1000dai/residual_mlp_libero_spatial \
--batch_size 64 \
--num_workers 16 \
--steps 400000 \
--dataset.repo_id k1000dai/libero-spatial-smolvla \
--output_dir output_residual_mlp_spatial \
--job_name residual_mlp_libero_spatial \
--wandb.enable True
```
This uses the same training loop but instantiates the MLP head (`--policy.type residual_mlp`) that directly regresses the refined action. Vision encoders remain frozen by default; tune `--policy.dim_model` or `--policy.hidden_dims` as needed for your dataset scale.

## Evaluation
```bash
MUJOCO_GL=glx python eval_libero/evaluation_libero.py 
```
