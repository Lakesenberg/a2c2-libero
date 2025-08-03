from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import time
import torch
import numpy as np
import os 

if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

# BASE_REPO_NAME = "k1000dai/libero"
BASE_REPO_NAME = "k1000dai/libero"
UPLOAD_REPO_NAME = "k1000dai/libero-addinfo"
BASE_POLICY_NAME = "k1000dai/smolvla_libero_scratch"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

base_policy = SmolVLAPolicy.from_pretrained(BASE_POLICY_NAME)
base_policy.to("cuda")
base_policy.eval()

new_features = {**base_dataset.features,
                "predicted_action" :{
                "dtype": "float32",
                "shape": (7,),
                "names": ["predicted_action"],
            },
                "elapsed_time": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["elapsed_time"],
            },
                "language_embedding": {
                "dtype": "float32",
                "shape": (960,),
                "names": ["language_embedding"],
            }   
}  
print(f"Base dataset features: {base_dataset.features}")
print(f"New dataset features: {new_features}")
new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
    robot_type="panda",
    fps=base_dataset.fps,   
    features=new_features,
    image_writer_threads=10,
    image_writer_processes=5,
)

predicted_action = []
time_index = 0
episode_index = 0
for i in range(len(base_dataset)):
    # save the episode i
    if episode_index != base_dataset[i]["episode_index"]:
        print("Saving episode", episode_index)
        new_dataset.save_episode()
        episode_index = base_dataset[i]["episode_index"]
        print(f"Starting episode {episode_index}")
        # Reset predicted action
        # Reset time index
        time_index = 0
        predicted_action = [] 

    # show the first 10 images
    if time_index == 50 or len(predicted_action) == 0:
        predicted_action = base_policy.predict_action_chunk(
            {
                "observation.images.image": base_dataset[i]["observation.images.image"].unsqueeze(0).to("cuda"),
                "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].unsqueeze(0).to("cuda"),
                "observation.state": base_dataset[i]["observation.state"].unsqueeze(0).to("cuda"),
                "task": base_dataset[i]["task"]
            }
        )
        predicted_action = predicted_action.squeeze(0)  # Remove batch dimension
        predicted_action = predicted_action.cpu()  # Move to CPU
        language_embedding = base_policy.model.language_embeddings.unsqueeze(0).cpu()  # Move to CPU
        time_index = 0
    
    new_dataset.add_frame(
        {
            "observation.images.image": base_dataset[i]["observation.images.image"].permute(1, 2, 0),
            "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].permute(1, 2, 0),
            "observation.state": base_dataset[i]["observation.state"],
            "action": base_dataset[i]["action"],
            "predicted_action": predicted_action[time_index],
            "elapsed_time": np.array([time_index], dtype=np.int64),  # Create array with shape (1,)
            "language_embedding": language_embedding  # Remove batch dimension
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

        
        
        

    