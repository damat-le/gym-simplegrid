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

    env = SimpleGridEnv(
        start_xy=(0, 5),
        goal_xy=(4, 7),
        obstacle_map="8x8",
        seed=42
    )
    obs, rew, done, _ = env.reset()

    logger.info("Running action-perception loop...")
    
    with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
        f.write(f"step,x,y,reward,done,action\n")

        frames = []

        for t in range(500):
            img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
            log_img(t, img)
            frames.append(img)

            action = env.action_space.sample()
            
            f.write(f"{t},{obs[0]},{obs[1]},{rew},{done},{action}\n")

            if done:
                logger.info(f"...agent is done at time step {t}")
                break

            obs, rew, done, _ = env.step(action)

    env.close()
    create_gif(frames)
    logger.info("...done")
    logger.info("-------------END-------------")