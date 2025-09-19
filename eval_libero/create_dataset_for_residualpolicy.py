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

BASE_REPO_NAME = "k1000dai/libero-spatial"
UPLOAD_REPO_NAME = "k1000dai/libero-smolvla"
BASE_POLICY_NAME = "k1000dai/smolvla_libero_scratch"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

base_policy = SmolVLAPolicy.from_pretrained(BASE_POLICY_NAME)
base_policy.to("cuda")
base_policy.eval()

device = next(base_policy.parameters()).device

first_sample = base_dataset[0]
with torch.no_grad():
    initial_observation = {
        "observation.images.image": first_sample["observation.images.image"].unsqueeze(0).to(device),
        "observation.images.wrist_image": first_sample["observation.images.wrist_image"].unsqueeze(0).to(device),
        "observation.state": first_sample["observation.state"].unsqueeze(0).to(device),
        "task": first_sample["task"],
    }
    base_policy.predict_action_chunk(initial_observation)

initial_vlm_hidden = getattr(base_policy, "vlm_hidden", None)
if initial_vlm_hidden is None:
    raise RuntimeError("SmolVLA policy did not return `vlm_hidden`. Ensure the model exposes hidden states.")

hidden_dim = int(initial_vlm_hidden.squeeze(0).shape[-1])

base_policy.reset()

new_features = copy.deepcopy(base_dataset.features)
new_features["vla_actions"] = { 
                "dtype": "float32",
                "shape": (50,7),
                "names": ["vla_actions"],
            }
new_features["vlm_hidden"] = {
                "dtype": "float32",
                "shape": (hidden_dim,),
                "names": ["vlm_hidden"],
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

    observations = {
        "observation.images.image": torch.stack([s["observation.images.image"] for s in samples]).to(device),
        "observation.images.wrist_image": torch.stack([s["observation.images.wrist_image"] for s in samples]).to(device),
        "observation.state": torch.stack([s["observation.state"] for s in samples]).to(device),
        "task": [s["task"] for s in samples],
    }

    with torch.no_grad():
        predicted_actions = base_policy.predict_action_chunk(observations)

    vlm_hidden = getattr(base_policy, "vlm_hidden", None)
    if vlm_hidden is None:
        raise RuntimeError("Expected SmolVLA policy to expose `vlm_hidden` after inference.")

    predicted_actions = predicted_actions.cpu()
    vlm_hidden = vlm_hidden.cpu()

    for sample, action_chunk, hidden_vec in zip(samples, predicted_actions, vlm_hidden, strict=False):
        episode_idx = sample["episode_index"]
        if current_episode is None:
            current_episode = episode_idx
        elif episode_idx != current_episode:
            print("Saving episode", current_episode)
            new_dataset.save_episode()
            current_episode = episode_idx

        new_dataset.add_frame(
            {
                "observation.images.image": sample["observation.images.image"].permute(1, 2, 0),
                "observation.images.wrist_image": sample["observation.images.wrist_image"].permute(1, 2, 0),
                "observation.state": sample["observation.state"],
                "action": sample["action"],
                "vla_actions": action_chunk,
                "vlm_hidden": hidden_vec.clone(),
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
    tags=["libero", "panda", "rlds"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)
    
