from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import time
import torch
import numpy as np
import os 
import copy


BATCH_SIZE = 32


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME = "k1000dai/so101_put_on_conveyor_slow_training"
UPLOAD_REPO_NAME = "k1000dai/so101_put_on_conveyor_slow_training-smolvla"
BASE_POLICY_NAME = "k1000dai/smolvla_conveyor_slow_finetune"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

base_policy = SmolVLAPolicy.from_pretrained(BASE_POLICY_NAME)
base_policy.to("cuda")
base_policy.eval()

new_features = copy.deepcopy(base_dataset.features)
new_features["vla_actions"] = { 
                "dtype": "float32",
                "shape": (50,6),
                "names": ["vla_actions"],
            }
context_dim = base_policy.model.vlm_with_expert.config.text_config.hidden_size
new_features["vlm_context"] = {
                "dtype": "float32",
                "shape": (context_dim,),
                "names": ["vlm_context"],
            }
 
print(f"Base dataset features: {base_dataset.features}")
print(f"New dataset features: {new_features}")
new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
    robot_type="panda",
    fps=base_dataset.fps,   
    features=new_features,
    image_writer_threads=20,
    image_writer_processes=10,
)

time_index = 0
current_episode = None
buffer: list[dict] = []

def process_buffer(samples: list[dict]) -> None:
    global time_index, current_episode
    if not samples:
        return

    device = next(base_policy.parameters()).device
    observations = {
        "observation.images.wrist": torch.stack([s["observation.images.wrist"] for s in samples]).to(device),
        "observation.images.top": torch.stack([s["observation.images.top"] for s in samples]).to(device),
        "observation.state": torch.stack([s["observation.state"] for s in samples]).to(device),
        "task": [s["task"] for s in samples],
    }

    with torch.no_grad():
        predicted_actions = base_policy.predict_action_chunk(observations)

    vlm_context = getattr(base_policy, "vlm_context", None)
    if vlm_context is None:
        raise RuntimeError("Expected SmolVLA policy to expose `vlm_context` after inference.")

    predicted_actions = predicted_actions.cpu()
    vlm_context = vlm_context.cpu()

    for sample, action_chunk, context in zip(samples, predicted_actions, vlm_context, strict=False):
        episode_idx = sample["episode_index"]
        if current_episode is None:
            current_episode = episode_idx
        elif episode_idx != current_episode:
            print("Saving episode", current_episode)
            new_dataset.save_episode()
            current_episode = episode_idx

        new_dataset.add_frame(
            {
                "observation.images.wrist": sample["observation.images.wrist"].permute(1, 2, 0),
                "observation.images.top": sample["observation.images.top"].permute(1, 2, 0),
                "observation.state": sample["observation.state"],
                "action": sample["action"],
                "vla_actions": action_chunk,
                "vlm_context": context,
            },
            task=sample["task"],
        )

        if time_index % 1000 == 0:
            print(
                f"Processed episode {episode_idx}, frame {time_index} / {len(base_dataset)}, time index {time_index}"
            )
        time_index += 1


for idx in range(len(base_dataset)):
    sample = base_dataset[idx]
    buffer.append(sample)
    if len(buffer) >= BATCH_SIZE:
        process_buffer(buffer)
        buffer = []

process_buffer(buffer)

# Save the last episode
new_dataset.save_episode()
print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["so101"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)
    
