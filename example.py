from __future__ import annotations
import os
import sys
import logging
import gymnasium as gym
from datetime import datetime as dt

from gym_simplegrid.envs.simple_grid import SimpleGridEnv

# --- Configuration & Presets ---
MAPS: dict[str, list[str]] = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
    "corridor": [
        "00011000",
        "10010000",
        "00000001",
        "01000001",
    ]
}

def run_simulation_basic():

    options = {
        'start_loc': 0,      # Top-left
        'goal_loc': (2, 4)   # Specific coordinate
    }

    env = gym.make(
        'SimpleGrid-v0',
        obstacle_map=MAPS["corridor"],
        render_mode='human',
        render_fps=5,
        max_episode_steps=250
    )

    obs, info = env.reset(seed=42, options=options)
    print("Start position is (row, col):", env.unwrapped.start_xy)
    print("Goal position is (row, col):", env.unwrapped.goal_xy)
    print(f"Step: {info['step_count']} | Pos: {info['agent_xy']} | Reward: -")
    env.render() # Initial render

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {info['step_count']} | Pos: {info['agent_xy']} | Reward: {reward}")
        env.render()

        if terminated:
            print("Goal Reached!")
            break
        
        if truncated:
            print("Episode Truncated (Max Steps Reached).")
            break
    env.render()
    env.close()

def run_simulation_advanced():
    # Setup Logging
    folder_name = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"log/{folder_name}"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "simulation.log")),
            logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("------------- START SIMULATION -------------")

    # Define Environment Options
    # We can pass specific start/goal indices (int) or coordinates (tuple)
    options = {
        'start_loc': 0,      # Top-left
        'goal_loc': (2, 4)   # Specific coordinate
    }

    # Instantiate Environment
    # We pass the custom map, set a max_steps limit for truncation, and pick a render mode
    env = SimpleGridEnv(
        obstacle_map=MAPS["corridor"], 
        render_mode='human', 
        render_fps=10,
        max_steps=150
    )

    # Action-Perception Loop
    obs, info = env.reset(seed=42, options=options)
    env.render() # Initial render
    
    logger.info(f"Initial State: {obs} | Agent XY: {info['agent_xy']}")

    frames = []
    total_reward = 0.0
    
    while True:
        # Sample random action (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)
        action = env.action_space.sample()
        
        # Step the environment (Gymnasium 0.26+ returns 5 values)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Explicitly call render to update the UI
        # In 'human' mode, this updates the Matplotlib window
        # In 'rgb_array' mode, this returns the pixel data
        frame = env.render()
        if env.render_mode == 'rgb_array':
            frames.append(frame)

        logger.info(f"Step {info['step_count']} | Pos: {info['agent_xy']} | Last Reward: {reward}")

        # Check for episode end
        if terminated:
            logger.info(f"Goal Reached! Total Reward: {total_reward} at step {info['step_count']}")
            break
        
        if truncated:
            logger.warning(f"Episode Truncated (Max Steps Reached). Total Reward: {total_reward}")
            break

    # Post-Simulation Handling
    # NOTE: to get the video utils, run `pip install gymnasium[other]`
    if env.render_mode == 'rgb_array' and frames:
        from gymnasium.utils.save_video import save_video
        save_video(frames, log_dir, fps=env.metadata['render_fps'])
        logger.info(f"Video saved to {log_dir}")

    logger.info("------------- END SIMULATION -------------")
    
    # Clean up Matplotlib resources
    env.close()

if __name__ == '__main__':
    run_simulation_basic()
    # run_simulation_advanced()