# NOTE: to run this you must install additional dependencies

if __name__=='__main__':
    import logging, os, sys
    from gym_simplegrid.envs import SimpleGridEnv
    from datetime import datetime as dt
    import gymnasium as gym
    from gymnasium.utils.save_video import save_video

    # Folder name for the simulation
    FOLDER_NAME = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(f"log/{FOLDER_NAME}")

    # Logger to have feedback on the console and on a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("-------------START-------------")

    options ={
        'start_loc': 12,
        # goal_loc is not specified, so it will be randomly sampled
    }

    obstacle_map = [
        "10001000",
        "10010000",
        "00000001",
        "01000001",
    ]
    
    env = gym.make(
        'SimpleGrid-v0', 
        obstacle_map=obstacle_map, 
        render_mode='rgb_array_list'
    )

    obs, info = env.reset(seed=1, options=options)
    rew = env.unwrapped.reward
    done = env.unwrapped.done

    logger.info("Running action-perception loop...")
    
    with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
        f.write(f"step,x,y,reward,done,action\n")
        
        for t in range(500):
            #img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
            
            action = env.action_space.sample()
            f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")
            
            if done:
                logger.info(f"...agent is done at time step {t}")
                break
            
            obs, rew, done, _, info = env.step(action)
            
    env.close()
    if env.render_mode == 'rgb_array_list':
        frames = env.render()
        save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)
    logger.info("...done")
    logger.info("-------------END-------------")