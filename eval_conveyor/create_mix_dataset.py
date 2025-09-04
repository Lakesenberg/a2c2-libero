from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os 
import copy
from tqdm import tqdm


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME_SLOW = "k1000dai/so101_put_yellow_block_on_conveyor_slow"
BASE_REPO_NAME_FAST = "k1000dai/so101_put_yellow_block_on_conveyor_fast"
UPLOAD_REPO_NAME = "k1000dai/so101_put_yellow_block_on_conveyor_mix"

base_dataset_slow = LeRobotDataset(
    repo_id=BASE_REPO_NAME_SLOW,
)
base_dataset_fast = LeRobotDataset(
    repo_id=BASE_REPO_NAME_FAST,
)

new_features = copy.deepcopy(base_dataset_slow.features)
new_features.pop("observation.images.eval")
 
print(f"Base dataset features: {base_dataset_slow.features}")
print(f"New dataset features: {new_features}")
new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
    robot_type="so101",
    fps=base_dataset_slow.fps,   
    features=new_features,
    image_writer_threads=20,
    image_writer_processes=10,
)

episode_index = 0
for i in tqdm(range(len(base_dataset_fast))):
    if episode_index != base_dataset_fast[i]["episode_index"]:
        print("Saving episode", episode_index)
        new_dataset.save_episode()
        episode_index = base_dataset_fast[i]["episode_index"]
    
    new_dataset.add_frame(
        {
            "observation.images.wrist": base_dataset_fast[i]["observation.images.wrist"].permute(1, 2, 0),
            "observation.images.top": base_dataset_fast[i]["observation.images.top"].permute(1, 2, 0),
            "observation.state": base_dataset_fast[i]["observation.state"],
            "action": base_dataset_fast[i]["action"],
        },
        task=base_dataset_fast[i]["task"],
    )

episode_index = 0
for i in tqdm(range(len(base_dataset_slow))):
    # save the episode i
    if episode_index != base_dataset_slow[i]["episode_index"]:
        print("Saving episode", episode_index)
        new_dataset.save_episode()
        episode_index = base_dataset_slow[i]["episode_index"]
    
    
    new_dataset.add_frame(
        {
            "observation.images.wrist": base_dataset_slow[i]["observation.images.wrist"].permute(1, 2, 0),
            "observation.images.top": base_dataset_slow[i]["observation.images.top"].permute(1, 2, 0),
            "observation.state": base_dataset_slow[i]["observation.state"],
            "action": base_dataset_slow[i]["action"],
        },
        task=base_dataset_slow[i]["task"],
    )


# Save the last episode
new_dataset.save_episode()
print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["so101"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)