from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os 


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME = "k1000dai/libero"
UPLOAD_REPO_NAME = "k1000dai/libero-object"

base_dataset = LeRobotDataset(
    repo_id=BASE_REPO_NAME,
)

new_dataset = LeRobotDataset.create(
    repo_id=UPLOAD_REPO_NAME,
    robot_type="panda",
    fps=base_dataset.fps,   
    features=base_dataset.features,
    image_writer_threads=20,
    image_writer_processes=10,
)

object_task_list = [
    "pick up the alphabet soup and place it in the basket", 
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
]

# Speedup: iterate per-episode and skip non-target episodes early.
allowed_tasks = set(object_task_list)
ep_from = base_dataset.episode_data_index["from"].tolist()
ep_to = base_dataset.episode_data_index["to"].tolist()

for start, end in zip(ep_from, ep_to, strict=True):
    # Check task on the first frame of the episode
    first_item = base_dataset[start]
    if first_item["task"] not in allowed_tasks:
        continue

    print(f"Processing episode {int(first_item['episode_index'])}")
    print(f"Task: {first_item['task']}")

    # Add all frames of this episode
    for i in range(start, end):
        item = base_dataset[i]
        new_dataset.add_frame(
            {
                "observation.images.image": item["observation.images.image"].permute(1, 2, 0),
                "observation.images.wrist_image": item["observation.images.wrist_image"].permute(1, 2, 0),
                "observation.state": item["observation.state"],
                "action": item["action"],
            },
            task=item["task"],
        )

    print("Saving episode", int(first_item["episode_index"]))
    new_dataset.save_episode()

print("\nAll episodes processed and saved.")
new_dataset.push_to_hub(
    tags=["libero", "panda", "rlds"],
    private=False,
    push_videos=True,
    license="apache-2.0",
)
    