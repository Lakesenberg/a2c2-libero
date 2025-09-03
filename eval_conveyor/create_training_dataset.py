from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import time
import torch
import numpy as np
import os 
import copy


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME = "k1000dai/so101_put_yellow_block_on_conveyor_slow"
UPLOAD_REPO_NAME = "k1000dai/so101_put_yellow_block_on_conveyor_slow_training"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

new_features = copy.deepcopy(base_dataset.features)
new_features.pop("observation.images.eval")
 
print(f"Base dataset features: {base_dataset.features}")
print(f"New dataset features: {new_features}")
new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
    robot_type="so101",
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
    
    
    new_dataset.add_frame(
        {
            "observation.images.wrist": base_dataset[i]["observation.images.wrist"].permute(1, 2, 0),
            "observation.images.top": base_dataset[i]["observation.images.top"].permute(1, 2, 0),

            "observation.state": base_dataset[i]["observation.state"],
            "action": base_dataset[i]["action"],
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
    tags=["so101"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)
    