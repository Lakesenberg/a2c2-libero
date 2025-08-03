from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import time
import torch

# BASE_REPO_NAME = "k1000dai/libero"
BASE_REPO_NAME = "k1000dai/libero_pick_up_the_black_bowl"
UPLOAD_REPO_NAME = "k1000dai/libero-residual-act_test"
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
}  
print(f"Base dataset features: {base_dataset.features}")
print(f"New dataset features: {new_features}")
new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
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
        time_index = 0
    # show the first 10 images
    if time_index == 49 or len(predicted_action) == 0:
        predicted_action = base_policy.predict_action_chunk(
            {
                "observation.images.image": base_dataset[i]["observation.images.image"].unsqueeze(0).to("cuda"),
                "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].unsqueeze(0).to("cuda"),
                "observation.state": base_dataset[i]["observation.state"].unsqueeze(0).to("cuda"),
                "task": base_dataset[i]["task"]
            }
        )
        predicted_action = predicted_action.squeeze(0)  # Remove batch dimension
        time_index = 0
    print(predicted_action.shape)
    
    new_dataset.add_frame(
        {
            "observation.images.image": base_dataset[i]["observation.images.image"].permute(2, 0, 1),
            "observation.images.wrist_image": base_dataset[i]["observation.images.wrist_image"].permute(2, 0, 1),
            "observation.state": base_dataset[i]["observation.state"],
            "action": base_dataset[i]["action"],
            "predicted_action": predicted_action[time_index].cpu(),
            "elapsed_time": time_index,
        },
        task=base_dataset[i]["task"],
    )
    print(f"Processed episode {episode_index}, frame {i}, time index {time_index}")
    time_index += 1

# Save the last episode
new_dataset.save_episode()
print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["libero", "panda", "rlds"],
    private=True,
    push_videos=True,
    license="apache-2.0",
)

        
        
        

    