"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import os

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

CHUNK_SIZE = 50
NUM_STEPS_WAIT = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def eval() -> None:
    base_policy_path: str = "k1000dai/smolvla_libero_scratch"
    residual_policy_path: str | None = "k1000dai/residual_transformer_libero_spatial"
    num_trials_per_task: int = 10 # Number of rollouts per task.
    out_base_path = "data/libero"
    
    task_suite_name_list = [ "libero_spatial", "libero_object", "libero_goal", "libero_10"]
    time_pair = [(0,1),(0,5),(0,10),(0,30),(0,40),(0,50),(1,10),(3,10),(5,10),(10,10),(1,40),(3,40),(5,40),(10,40)]
    eval_without_residual = False
    eval_with_residual = True
    time_pair.reverse()  # Reverse to start with the smallest execute_horizon and inference_delay
    seed = 7
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Load Policy ---
    base_policy = SmolVLAPolicy.from_pretrained(base_policy_path)
    base_policy.to(DEVICE)
    base_policy.eval()
    
    residual_policy = None
    if residual_policy_path and eval_with_residual:
        residual_policy = ResidualTransformerPolicy.from_pretrained(residual_policy_path)
        residual_policy.to(DEVICE)
        residual_policy.eval()
    

    logging.info("=== Evaluating without residual policy ===")
    for task_suite_name in task_suite_name_list:
        video_out_base_path:pathlib.Path = pathlib.Path(os.path.join(out_base_path,f"videos_{task_suite_name}"))
        json_out_path :pathlib.Path = pathlib.Path(os.path.join(out_base_path,f"results_{task_suite_name}.jsonl"))
        for inference_delay, execute_horizon in time_pair:
            if execute_horizon < inference_delay:
                continue
            if execute_horizon+ inference_delay > CHUNK_SIZE:
                continue
            logging.info(f"Evaluating with execute_horizon={execute_horizon}, inference_delay={inference_delay}")
            
            if eval_without_residual:
                # Evaluate without residual policy
                video_out_path = pathlib.Path(video_out_base_path) / f"execute_horizon_{execute_horizon}_inference_delay_{inference_delay}"
                results = eval_libero(
                    base_policy=base_policy,
                    residual_policy=None,
                    use_residual_policy=False,
                    task_suite_name=task_suite_name,
                    num_trials_per_task=num_trials_per_task,
                    seed=seed,
                    execute_horizon=execute_horizon,
                    inference_delay=inference_delay,
                    video_out_path=str(video_out_path)
                )
                with open(json_out_path, "a") as f:
                    f.write(f"{results}\n")
                logging.info(f"Results without residual policy: {results}")

            if eval_with_residual:
                # Evaluate with residual policy
                video_out_path = pathlib.Path(video_out_base_path) / f"execute_horizon_{execute_horizon}_inference_delay_{inference_delay}_residual"
                results = eval_libero(
                    base_policy=base_policy,
                    residual_policy=residual_policy,
                    use_residual_policy=True,
                    task_suite_name=task_suite_name,
                    num_trials_per_task=num_trials_per_task,
                    seed=seed,
                    execute_horizon=execute_horizon,
                    inference_delay=inference_delay,
                    video_out_path=str(video_out_path)
                )
                with open(json_out_path, "a") as f:
                    f.write(f"{results}\n")
                logging.info(f"Results with residual policy: {results}")

    # Log final results
    logging.info("=== Evaluation completed ===")

def eval_libero(base_policy: SmolVLAPolicy,
                residual_policy: ResidualTransformerPolicy | None,
                use_residual_policy: bool = True,
                task_suite_name: str = "libero_spatial", 
                num_trials_per_task: int = 10, 
                seed: int = 7,
                execute_horizon: int = 10, 
                inference_delay: int = 0, 
                video_out_path: str = "data/libero/videos"
                ) -> dict:
    if use_residual_policy:
        if residual_policy is None:
            raise ValueError("Residual policy requested but not provided.")
        print("====  USE RESIDUAL POLICY ====")
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[task_suite_name]()
    except KeyError:
        raise ValueError(
            f"Unknown task suite: {task_suite_name}. "
            f"Available options are: {list(benchmark_dict.keys())}"
        )
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {task_suite_name}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        # Fallback for custom task suites
        max_steps = 520

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm(
            range(min(num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and policy
            env.reset()
            base_policy.reset()
            if residual_policy is not None:
                residual_policy.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Setup
            t = 0
            frames = []
            done = False
            action_plan = collections.deque()
            pending_actions = collections.deque()
            # Add initial frame
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(agentview_image)
            logging.info(f"Starting episode {task_episodes+1}...")
            
            while t < max_steps:
                try:
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(DEVICE).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(DEVICE).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(DEVICE).unsqueeze(0),
                        "task": task_description,
                    }

                    if len(action_plan) == 0:
                        new_action_chunk = base_policy.predict_action_chunk(observation)
                        new_action_chunk = new_action_chunk.squeeze(0).cpu().numpy()
                        chunk_size = new_action_chunk.shape[0]
                        new_time_offsets = np.arange(chunk_size, dtype=np.int64)

                        chunk_entries = [
                            {
                                "action": new_action_chunk[i],
                                "time_offset": int(new_time_offsets[i]),
                                "chunk": new_action_chunk,
                            }
                            for i in range(chunk_size)
                        ]

                        available_prev = min(len(pending_actions), inference_delay)
                        exec_entries = [pending_actions.popleft() for _ in range(available_prev)]

                        start_new = min(inference_delay, execute_horizon)
                        exec_entries.extend(chunk_entries[start_new:execute_horizon])

                        action_plan = collections.deque(exec_entries)

                        if start_new > 0:
                            pending_actions.extend(chunk_entries[:start_new])
                        if execute_horizon < chunk_size:
                            pending_actions.extend(chunk_entries[execute_horizon:])

                    if action_plan:
                        plan_entry = action_plan.popleft()
                        action = plan_entry["action"]
                        time_offset = plan_entry["time_offset"]
                        source_chunk = plan_entry["chunk"]
                    else:
                        # Fallback - should not happen with correct logic
                        logging.warning("No actions in plan, using zero action")
                        action = np.zeros(7, dtype=np.float32)
                        time_offset = 0
                        source_chunk = np.zeros((1, action.shape[0]), dtype=np.float32)

                    if use_residual_policy and residual_policy is not None:
                        base_chunk_np = np.asarray(source_chunk, dtype=np.float32)

                        base_action_tensor = (
                            torch.from_numpy(action)
                            .to(torch.float32)
                            .to(DEVICE)
                            .unsqueeze(0)
                        )
                        observation["action"] = base_action_tensor
                        observation["base_action_chunk"] = (
                            torch.from_numpy(base_chunk_np)
                            .to(torch.float32)
                            .to(DEVICE)
                            .unsqueeze(0)
                        )

                        denom = max(CHUNK_SIZE - 1, 1)
                        phase = 2 * math.pi * float(time_offset % CHUNK_SIZE) / denom
                        time_feature = torch.tensor([[math.sin(phase), math.cos(phase)]], dtype=torch.float32, device=DEVICE)
                        observation["time_feature"] = time_feature

                        vlm_hidden = getattr(base_policy, "vlm_hidden", None)
                        if vlm_hidden is None:
                            vlm_hidden = getattr(base_policy.model, "vlm_hidden", None)
                        if vlm_hidden is not None:
                            observation["vlm_hidden"] = vlm_hidden.to(DEVICE)

                        updated_action = (
                            residual_policy.predict_action_chunk(observation)
                            .squeeze(0)
                            .cpu()
                            .numpy()[0]
                        )
                        action = updated_action
                    
                    obs, _, done, _ = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break 
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            use_residual_suffix = "_residual" if use_residual_policy else ""
            video_path = (
                pathlib.Path(video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}{use_residual_suffix}.mp4"
            )
            fps = 30
            writer = imageio.get_writer(video_path, fps=fps)

            for image in frames:
                writer.append_data(image)
            writer.close()
            # import ipdb; ipdb.set_trace()

        # Log final results for the task
        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}")
        if total_episodes > 0:
            logging.info(f"Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}")

    logging.info("--- Evaluation finished ---")
    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    return {
        "execute_horizon": execute_horizon,
        "inference_delay": inference_delay,
        "task_suite_name": task_suite_name,
        "num_trials_per_task": num_trials_per_task,
        "use_residual_policy": use_residual_policy,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
    }


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval()
