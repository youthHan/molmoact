# client.py

import argparse
import base64
import ast
import cv2
import numpy as np
import requests
import tqdm
from libero.libero import benchmark
from libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    get_libero_wrist_image, save_rollout_video
)
from robot_utils import get_image_resize_size, set_seed_everywhere

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000/generate_action"

# --- Client-Side Helper Functions ---

def normalize_gripper_action(action, binarize=True):
    """Changes gripper action from [0,1] to [-1,+1]."""
    if isinstance(action, list): action = np.array(action)
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize: action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):
    """Flips the sign of the gripper action."""
    if isinstance(action, list): action = np.array(action)
    action[..., -1] *= -1.0
    return action

# --- API Communication ---

def step_via_api(img, wrist_img, language_instruction, unnorm_key):
    """Sends observations to the server and gets an action in return."""
    # Encode images (which are RGB) to PNG bytes then to base64
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_buffer = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(img_buffer).decode('utf-8')
    
    wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
    _, wrist_buffer = cv2.imencode('.png', wrist_bgr)
    wrist_img_b64 = base64.b64encode(wrist_buffer).decode('utf-8')

    payload = {
        "img_b64": img_b64,
        "wrist_img_b64": wrist_img_b64,
        "language_instruction": language_instruction,
        "unnorm_key": unnorm_key,
    }

    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(f"Server error: {data['error']}")
    
    # Decode annotated image
    annotated_bytes = base64.b64decode(data["annotated_image_b64"])
    annotated_np_bgr = cv2.imdecode(np.frombuffer(annotated_bytes, np.uint8), cv2.IMREAD_COLOR)
    annotated_image = cv2.cvtColor(annotated_np_bgr, cv2.COLOR_BGR2RGB)
    
    action = np.array(data["action"]) if data["action"] is not None else None
    trace = np.array(data["trace"]) if data["trace"] is not None else None
    
    if action is None:
        raise ValueError("Server failed to return a valid action.")

    return action, annotated_image, trace

# --- Main Evaluation Loop ---

def eval_libero(args, task_suite_name, seed, num_trials_per_task, num_steps_wait):
    set_seed_everywhere(seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Task suite: {task_suite_name}")
    resize_size = get_image_resize_size()

    total_episodes, total_successes = 0, 0
    task_id = args.task_id
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "your-model-family", resolution=256)
    
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task), desc=f"Task {task_id}"):
        last_gripper_state = -1
        print(f"\nTask: {task_description}")
        obs = env.set_init_state(initial_states[episode_idx])
        
        replay_images = []
        unnorm_key_map = {"spatial": "libero_spatial_no_noops_modified", "object": "libero_object_no_noops_modified", "goal": "libero_goal_no_noops_modified", "10": "libero_10_no_noops_modified"}
        max_steps_map = {"spatial": 220, "object": 280, "goal": 300, "10": 520}
        unnorm_key = unnorm_key_map.get(args.task, "libero_10_no_noops_modified")
        max_steps = max_steps_map.get(args.task, 400)

        print(f"Starting episode {task_episodes+1}...")
        
        t, outer_done = 0, False
        while t < max_steps + num_steps_wait and not outer_done:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action("your-model-family"))
                t += 1
                continue

            img = get_libero_image(obs, resize_size)
            wrist_img = get_libero_wrist_image(obs, resize_size)
            
            try:
                action_matrix, annotated_image, traj = step_via_api(img, wrist_img, task_description, unnorm_key)
            except Exception as e:
                print(f"API call failed: {e}. Using dummy action.")
                action_matrix = np.zeros((1, 7), dtype=float)
                action_matrix[:, -1] = last_gripper_state
                annotated_image = img
                traj = None

            replay_images.append(annotated_image)

            action_num = 0
            for single_action in action_matrix:
                if isinstance(single_action, str): single_action = ast.literal_eval(single_action)
                single_action = normalize_gripper_action(single_action, binarize=True)
                single_action = invert_gripper_action(single_action)
                try:
                    obs, _, done, _ = env.step(single_action)
                    if done != env.done:
                        print(f"Env mismatch - Done: {done} | Env Done: {env.done}")
                except Exception as e:
                    print(f"Error during env step {e}")
                last_gripper_state = single_action[-1]
                
                visualize = get_libero_image(obs, resize_size)
                visualize_annotated = np.array(visualize.copy())
                if traj is not None and len(traj) > 0:
                    try:
                        for i in range(len(traj) - 1):
                            p1 = tuple(map(int, traj[i]))
                            p2 = tuple(map(int, traj[i + 1]))
                            cv2.line(visualize_annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Trajectory annotation failed: {e}")
                
                replay_images.append(visualize_annotated)
                action_num += 1
                
                t += 1
                if done:
                    outer_done = True
                    break
            
            if done:
                task_successes += 1
                break
        
        task_episodes += 1
        total_episodes += 1

        total_successes += task_successes
        save_rollout_video(replay_images, total_episodes, success=done, task_description=task_description, checkpoint="client-run", task=task_suite_name)
        print(f"Success: {done} | Episode {total_episodes} | Success Rate: {total_successes / total_episodes * 100:.1f}%")

# --- Main Entry Point ---

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["spatial", "object", "goal", "10", "90"])
    p.add_argument("--task_id", type=int, required=False, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    task_suite_name = f"libero_{args.task}"
    seed = 7
    num_trials_per_task = 50
    num_steps_wait = 10  
    
    if args.task_id is not None:
        eval_libero(args, task_suite_name, seed, num_trials_per_task, num_steps_wait)
    else:
        for task_id in range(10):
            print(f"\n{'='*50}\nRunning Task ID: {task_id}\n{'='*50}")
            args.task_id = task_id
            eval_libero(args, task_suite_name, seed, num_trials_per_task, num_steps_wait)

if __name__ == "__main__":
    main()