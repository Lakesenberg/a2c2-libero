# a2c2-libero · Inference test harness

Asynchronous SmolVLA + A2C2 inference utilities and tests for the LIBERO
benchmark.

This branch (`Inference_test`) adds:

* `src/a2c2_libero/inference/async_smolvla.py` — background SmolVLA worker
* `src/a2c2_libero/inference/a2c2_engine.py` — per-tick A2C2 correction engine
  (synchronous and asynchronous flavours)
* `src/a2c2_libero/inference/utils.py` — sin/cos chunk-index encoding,
  state builder, latent forward-hook
* `src/a2c2_libero/inference/mocks.py` — mock SmolVLA / A2C2 head for tests
* `src/a2c2_libero/heads/a2c2_head.py` — minimal MLP residual head
* `tests/` — unit tests covering threading, per-tick correctness, and a
  LIBERO smoke test (skipped if LeRobot is not installed)
* `scripts/run_inference_demo.py` — CLI demo (no GPU, no LIBERO needed)

## Install

```bash
git clone https://github.com/Lakesenberg/a2c2-libero.git
cd a2c2-libero
git switch Inference_test
pip install -e ".[test]"
```

For the LIBERO smoke test you also need:

```bash
pip install -e ".[libero]"
export MUJOCO_GL=egl
```

## Run tests

```bash
# All non-slow tests (no GPU, no LIBERO needed)
pytest -v

# Including LIBERO smoke test
pytest -v -m slow
```

## Try the demo

```bash
# Async pipeline, default 200 ticks at 5 ms (1 s wall clock)
python scripts/run_inference_demo.py

# Synchronous version (blocks on each SmolVLA refresh)
python scripts/run_inference_demo.py --sync

# Make SmolVLA twice as slow to see chunks lasting longer
python scripts/run_inference_demo.py --smolvla-latency-ms 200
```

## Async timing model

```
Background SmolVLA:   [forward#0===][#1===][#2===][#3===]
                                  ↓      ↓      ↓      ↓
SharedBoard chunk:    None ────── chunk_0 chunk_1 chunk_2 chunk_3
SharedBoard z:        None ────── z_0     z_1     z_2     z_3
SharedBoard k:        0    ────── 0..N-1  0..N-1  0..N-1  0..N-1

Main thread A2C2:     hold-pose   step    step    step    step
                                  every tick (5 ms), reads board snapshot
```

The main thread never blocks on SmolVLA. It always uses the most recently
published chunk, applies the per-tick A2C2 correction
`a_exec = chunk[k] + π_C(o_t, chunk[k], τ_k, z, …)`, and advances `k`. When
SmolVLA finishes a new forward, the worker writes the new chunk to the board
and resets `k = 0`.

## Plugging in real models

`MockSmolVLA` and `MockA2C2Head` in `mocks.py` are placeholders. To run with
real models, replace `smolvla_fn` and `head_fn` in `A2C2Engine`:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from a2c2_libero.heads import A2C2MLPHead
from a2c2_libero.inference.utils import LatentHook

smolvla = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").cuda().eval()
hook    = LatentHook(smolvla.model.vlm_backbone.layers[-1])

@torch.no_grad()
def smolvla_fn(obs, language):
    chunk = smolvla.predict_action_chunk(obs, language)
    return chunk, hook.latest

head = A2C2MLPHead(input_dim=..., action_dim=7).cuda().eval()
head.load_state_dict(torch.load("a2c2_head.pt"))

@torch.no_grad()
def head_fn(state):
    return head(state)

engine = A2C2Engine(smolvla_fn, head_fn, chunk_size=50, action_dim=7)
engine.start_async()
```

## References

* A2C2: *Leave No Observation Behind* — arXiv:2509.23224
* SmolVLA — arXiv:2506.01844
* LeRobot — https://github.com/huggingface/lerobot
* LIBERO — https://github.com/Lifelong-Robot-Learning/LIBERO
