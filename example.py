import imageio

# To run this example, you need tabulate or imageio.
# You can install them with pip:
# `pip install tabulate imageio`

def log_img(t, frame):
    imageio.imwrite(f'log/{FOLDER_NAME}/img/{t}.png', frame)

def create_gif(frames):
    imageio.mimsave(f'log/{FOLDER_NAME}/movie.gif', frames)

if __name__=='__main__':
    import logging, os, sys
    from gym_simplegrid.envs import SimpleGridEnv
    from datetime import datetime as dt
    import gymnasium as gym

    # Folder name for the simulation
    FOLDER_NAME = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(f"log/{FOLDER_NAME}/img")

    # Logger to have feedback on the console and on a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("-------------START-------------")

    options ={
        'start_loc': 2,
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
        render_mode='human'
    )

    #env = gym.make('SimpleGrid-8x8-v0', render_mode="human")

    obs, info = env.reset(seed=1, options=options)
    rew = env.unwrapped.reward
    done = env.unwrapped.done

    logger.info("Running action-perception loop...")
    
    with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
        f.write(f"step,x,y,reward,done,action\n")
        
        #frames = []

        for t in range(500):
            #img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
            #log_img(t, img)
            #frames.append(img)
            
            action = env.action_space.sample()
            f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")
            
            if done:
                logger.info(f"...agent is done at time step {t}")
                break
            
            obs, rew, done, _, info = env.step(action)
            
    env.close()
    #frames = env.render()
    #create_gif(frames)
    logger.info("...done")
    logger.info("-------------END-------------")