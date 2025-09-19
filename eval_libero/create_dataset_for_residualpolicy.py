from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import time
import torch
import numpy as np
import os 
import copy


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME = "k1000dai/libero"
UPLOAD_REPO_NAME = "k1000dai/libero-smolvla"
BASE_POLICY_NAME = "k1000dai/smolvla_libero_scratch"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

base_policy = SmolVLAPolicy.from_pretrained(BASE_POLICY_NAME)
base_policy.to("cuda")
base_policy.eval()

new_features = copy.deepcopy(base_dataset.features)
new_features["vla_actions"] = { 
                "dtype": "float32",
                "shape": (50,7),
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
episode_index = 0

for i in range(len(base_dataset)):
    # save the episode i
    if episode_index != base_dataset[i]["episode_index"]:
        print("Saving episode", episode_index)
        new_dataset.save_episode()
        episode_index = base_dataset[i]["episode_index"]

    predicted_action = base_policy.predict_action_chunk(
        {
            "observation.images.image": base_dataset[i]["observation.images.image"].unsqueeze(0).to("cuda"),
            "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].unsqueeze(0).to("cuda"),
            "observation.state": base_dataset[i]["observation.state"].unsqueeze(0).to("cuda"),
            "task": base_dataset[i]["task"]
        }
    )
    vlm_context = getattr(base_policy, "vlm_context", None)
    if vlm_context is None:
        raise RuntimeError("Expected SmolVLA policy to expose `vlm_context` after inference.")

    predicted_action = predicted_action.squeeze(0)  # Remove batch dimension
    predicted_action = predicted_action.cpu()  # Move to CPU
    vlm_context = vlm_context.squeeze(0).cpu()

    new_dataset.add_frame(
        {
            "observation.images.image": base_dataset[i]["observation.images.image"].permute(1, 2, 0),
            "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].permute(1, 2, 0),
            "observation.state": base_dataset[i]["observation.state"],
            "action": base_dataset[i]["action"],
            "vla_actions": predicted_action,
            "vlm_context": vlm_context,
        },
        task=base_dataset[i]["task"],
    )
    if i % 1000 == 0:
        print(f"Processed episode {episode_index}, frame {i} / {len(base_dataset)}, time index {time_index}")
    time_index += 1
    
# Save the last episode
new_dataset.save_episode()
print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["libero", "panda", "rlds"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)
    
