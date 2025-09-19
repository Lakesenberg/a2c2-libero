import collections
import logging
import math
import pathlib
import os
import time
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.residualact.modeling_residualact import ResidualACTPolicy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

CHUNK_SIZE = 50
NUM_STEPS_WAIT = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def eval() -> None:
    base_policy_path: str = "k1000dai/smolvla_libero_scratch"
    residual_policy_path: str = "k1000dai/residualact_libero_small_200k"
    num_trials = 100
    seed = 7
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Load Policy ---
    base_policy = SmolVLAPolicy.from_pretrained(base_policy_path)
    base_policy.to(DEVICE)
    base_policy.eval()
    
    residual_policy = ResidualACTPolicy.from_pretrained(residual_policy_path)
    residual_policy.to(DEVICE)
    residual_policy.eval()

    base_policy_time_list, residual_policy_time_list = eval_inference_speed_using_libero(base_policy, residual_policy, use_residual_policy=True, num_trials=num_trials, seed=seed)
    print(base_policy_time_list)
    print(residual_policy_time_list)
    print(np.mean(base_policy_time_list))
    print(np.mean(residual_policy_time_list))
    

def eval_inference_speed_using_libero(base_policy: SmolVLAPolicy, 
                residual_policy:ResidualACTPolicy, 
                use_residual_policy: bool = True,
                num_trials: int = 10, 
                seed: int = 7,
                ) -> dict:
    
    task_suite_name = "libero_spatial"
    num_trials_per_task = 50
    execute_horizon = 1
    inference_delay = 0

    base_policy_time_list = []
    residual_policy_time_list = []
    
    # --- Load Policy ---
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
                        start_time = time.perf_counter()
                        new_action_chunk = base_policy.predict_action_chunk(observation)
                        end_time = time.perf_counter()
                        base_policy_time_list.append(end_time - start_time)
                        new_action_chunk = new_action_chunk.squeeze(0).cpu().numpy()
                        new_time_offsets = np.arange(new_action_chunk.shape[0], dtype=np.int64)

                        chunk_entries = [
                            {
                                "action": new_action_chunk[i],
                                "time_offset": int(new_time_offsets[i]),
                                "chunk": new_action_chunk,
                            }
                            for i in range(new_action_chunk.shape[0])
                        ]

                        available_prev = min(len(pending_actions), inference_delay)
                        exec_entries = [pending_actions.popleft() for _ in range(available_prev)]

                        start_new = min(inference_delay, execute_horizon)
                        exec_entries.extend(chunk_entries[start_new:execute_horizon])

                        action_plan = collections.deque(exec_entries)

                        if start_new > 0:
                            pending_actions.extend(chunk_entries[:start_new])
                        if execute_horizon < len(chunk_entries):
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

                    if use_residual_policy:
                        base_chunk_np = np.asarray(source_chunk, dtype=np.float32)[:CHUNK_SIZE]

                        observation["action"] = (
                            torch.from_numpy(action)
                            .to(torch.float32)
                            .to(DEVICE)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )  # Add batch and sequence dimensions
                        observation["base_action_chunk"] = (
                            torch.from_numpy(base_chunk_np)
                            .to(torch.float32)
                            .to(DEVICE)
                            .unsqueeze(0)
                        )
                        phase = 2 * math.pi * float(time_offset % CHUNK_SIZE) / max(CHUNK_SIZE - 1, 1)
                        observation["time_feature"] = (
                            torch.tensor([[math.sin(phase), math.cos(phase)]], dtype=torch.float32)
                            .to(DEVICE)
                        )
                        if getattr(base_policy, "vlm_hidden", None) is not None:
                            observation["vlm_hidden"] = base_policy.vlm_hidden.to(DEVICE)
                        start_time = time.perf_counter()
                        updated_action = residual_policy.predict_action_chunk(observation).squeeze(0).cpu().numpy()[0]
                        end_time = time.perf_counter()
                        residual_policy_time_list.append(end_time - start_time)
                        action = updated_action
                    
                    obs, _, done, _ = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break 
                    t += 1
                    if len(base_policy_time_list) == num_trials:
                        break
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            if len(base_policy_time_list) == num_trials:
                break
        if len(residual_policy_time_list) == num_trials:
            break

    return base_policy_time_list, residual_policy_time_list


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
