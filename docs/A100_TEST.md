# A100 inference benchmark — how to run

This benchmark loads a trained SmolVLA (default
`lerobot/smolvla_libero`), wraps it with the asynchronous A2C2 engine, and
measures per-tick latency on an A100 GPU.

## What it does

* Loads SmolVLA in eval mode and freezes parameters.
* Spawns the background SmolVLA worker; the main loop runs the A2C2 head
  every tick.
* Three workload modes:

  | mode      | observations come from              | env step? |
  |-----------|-------------------------------------|-----------|
  | synthetic | random tensors with LIBERO obs keys | no        |
  | dataset   | `lerobot/libero_spatial` parquet    | no        |
  | env       | `LiberoEnv` simulator               | **yes**   |

* Records latency percentiles (mean / p50 / p95 / p99 / max), GPU peak
  memory, and SmolVLA forward count. Saves a JSON report.

## Single-machine run (no SLURM)

```bash
# in your conda env with torch + lerobot installed
pip install -e ".[test,libero]"
export MUJOCO_GL=egl
python scripts/a100_inference_test.py --mode synthetic --ticks 1000
```

Synthetic mode does not need LIBERO; useful for first GPU smoke test.

## SLURM submit (typical A100 cluster)

```bash
mkdir -p logs
sbatch scripts/run_on_a100.sh                       # synthetic
sbatch scripts/run_on_a100.sh dataset                # real LIBERO obs
sbatch scripts/run_on_a100.sh env libero_spatial    # full env, scored
```

Adjust the SBATCH header in `scripts/run_on_a100.sh` to match your cluster
(partition, account, time limit). Logs land in `logs/a2c2_inf_<jobid>.{out,err}`.

## Expected numbers on a single A100 (40 GB)

| Metric                         | Expected            | Notes                       |
|--------------------------------|---------------------|-----------------------------|
| SmolVLA forward (450M, fp32)   | ~50–80 ms           | one chunk per call          |
| A2C2 head MLP (~0.3M)          | < 1 ms              | per-tick                    |
| A2C2 head Transformer (~32M)   | 3–6 ms              | per-tick                    |
| Async main-loop step (p99)     | < 8 ms              | with MLP head, 5 ms tick    |
| Sync step (p99)                | < (SmolVLA + head)  | when chunk is exhausted     |
| GPU peak memory                | ~3–4 GB (fp32)      | SmolVLA-450M + small head   |

If `p99` for the async step exceeds 15 ms with the MLP head, suspect:

* SmolVLA forward blocking the main thread (verify the worker thread is
  actually `start_async`'d);
* CPU-bound observation preprocessing (move it to GPU);
* Python GIL contention with overly large observation tensors copied via
  `copy.deepcopy`.

## Plugging in a trained A2C2 head checkpoint

```bash
HEAD_CKPT=/path/to/a2c2_head.pt sbatch scripts/run_on_a100.sh dataset
```

Or directly:

```bash
python scripts/a100_inference_test.py \
    --mode dataset \
    --head-ckpt /path/to/a2c2_head.pt \
    --policy-path lerobot/smolvla_libero
```

The head's `input_dim` must match what the runtime feeder produces. For the
synthetic / dataset modes this is computed from
`flatten_state_for_mlp({obs, a_base, tau_k, z})`. If you trained on a
different observation schema, update `_synthetic_obs` (or pass a custom
provider) so that the dimensions agree before loading the checkpoint.

## Caveats

* The latent extractor uses a forward hook on a heuristic module path
  (`model.vlm_backbone.layers[-1]` etc.). If SmolVLA's internals differ from
  the expected layout the latent will fall back to zeros and the test
  becomes a SmolVLA-only benchmark. Set the hook explicitly for your
  SmolVLA version when running real evaluation.
* `lerobot/smolvla_libero` is fp32 ~1.7 GB. First load downloads from HF.
* Env mode requires `pip install -e ".[libero]"` and `MUJOCO_GL=egl`.
* Random head weights make accuracy meaningless. Use this benchmark for
  *latency*; load a trained head for accuracy numbers.
