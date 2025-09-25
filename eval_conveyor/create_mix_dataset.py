from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os 
import copy
from tqdm import tqdm


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME_SLOW = "dataset/to/so101_put_yellow_block_on_conveyor_slow"
BASE_REPO_NAME_FAST = "dataset/to/so101_put_yellow_block_on_conveyor_fast"
UPLOAD_REPO_NAME = "dataset/to/so101_put_yellow_block_on_conveyor_mix"

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
    batch_encoding_size=8,
)

def _copy_dataset_by_episode(src: LeRobotDataset, dst: LeRobotDataset, desc: str) -> None:
    ep_index_map = src.episode_data_index
    num_eps = src.num_episodes
    for ep_idx in tqdm(range(num_eps), desc=desc):
        start = ep_index_map["from"][ep_idx].item()
        end = ep_index_map["to"][ep_idx].item()
        for i in range(start, end):
            item = src[i]
            wrist = item["observation.images.wrist"].permute(1, 2, 0).contiguous()
            top = item["observation.images.top"].permute(1, 2, 0).contiguous()
            dst.add_frame(
                {
                    "observation.images.wrist": wrist,
                    "observation.images.top": top,
                    "observation.state": item["observation.state"],
                    "action": item["action"],
                },
                task=item["task"],
            )
        dst.save_episode()


_copy_dataset_by_episode(base_dataset_fast, new_dataset, desc="Copy fast episodes")
_copy_dataset_by_episode(base_dataset_slow, new_dataset, desc="Copy slow episodes")


# Ensure any pending image writes are flushed
new_dataset.stop_image_writer()
print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["so101"],
    private=False,
    push_videos=True,
    license="apache-2.0",
    upload_large_folder=True,
)