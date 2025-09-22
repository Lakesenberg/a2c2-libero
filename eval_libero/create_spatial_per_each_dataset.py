from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os 


if os.environ.get("HF_TOKEN") is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

BASE_REPO_NAME = "k1000dai/libero-object-smolvla-add-vlm-context"


spatial_task_list = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]

for spatial_task in spatial_task_list:
    print(f"Processing spatial task: {spatial_task}")
    spatial_task_sanitized = spatial_task.replace(" ", "_").replace(",", "").replace(".", "").replace("(", "").replace(")", "") 
    UPLOAD_REPO_NAME = f"k1000dai/libero-{spatial_task_sanitized}"

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
    
    for start, end in zip(base_dataset.episode_data_index["from"].tolist(), base_dataset.episode_data_index["to"].tolist(), strict=True):
        # Check task on the first frame of the episode
        first_item = base_dataset[start]
        if first_item["task"] != spatial_task:
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
                    "vla_actions": item["vla_actions"],
                    "vlm_hidden": item["vlm_hidden"],
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