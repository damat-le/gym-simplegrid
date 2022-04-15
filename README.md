# Simple Gridworld Environment for OpenAI Gym

SimpleGrid is a super simple gridworld environment for OpenAI gym. It is easy to use and customise and it is intended to offer an environment for quick testing and prototyping different RL algorithms.

It is also efficient, lightweight and has few dependencies (gym, numpy, matplotlib). 

![](img/simplegrid.gif)

SimpleGrid involves navigating a grid from Start(S) (red tile) to Goal(G) (green tile) without colliding with any Wall(W) (black tiles) by walking over
the Empty(E) (white tiles) cells. The yellow circle denotes the agent's current position. 

Optionally, it is possible to introduce a noise in the environment that makes the agent move in a random direction that can be different than the desired one.


## Installation

To install SimpleGrid, you can either use pip

```bash
pip install gym-simplegrid
```

or you can clone the repository and run an editable installation

```bash
git clone https://github.com/damat-le/gym-simplegrid.git
cd gym-simplegrid
pip install -e .
```


## Citation

Please use this bibtex if you want to cite this repository in your publications:

```tex
@misc{gym_simplegrid,
  author = {Leo D'Amato},
  title = {Simple Gridworld Environment for OpenAI Gym},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/damat-le/gym-simplegrid}},
}
```

## Getting Started

Basic usage options:

```python
import gym 
import gym_simplegrid

# Load the default 8x8 map
env = gym.make('SimpleGrid-8x8-v0')

# Load the default 4x4 map
env = gym.make('SimpleGrid-4x4-v0')

# Load a random map
env = gym.make('SimpleGrid-v0')

# Load a custom map with multiple starting states
# At the beginning of each episode a new starting state will be sampled
my_desc = [
        "SEEEEEES",
        "EEESEEES",
        "WEEWEEEE",
        "EEEEEWEG",
    ]
env = gym.make('SimpleGrid-v0', desc=my_desc)

# Set custom rewards and introduce noise
# The agent will move in the intended direction with probability 1-p_noise
my_reward_map = {
        b'E': -1.0,
        b'S': -0.0,
        b'W': -5.0,
        b'G': 5.0,
    }
env = gym.make('SimpleGrid-8x8-v0', reward_map=my_reward_map, p_noise=.4)
```

Example with rendering:

```python
import gym 
import gym_simplegrid

env = gym.make('SimpleGrid-8x8-v0')
observation = env.reset()
T = 50
for _ in range(T):
    action = env.action_space.sample()
    env.render()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
```


## Environment Description

### Action Space

The action space is `gym.spaces.Discrete(4)`. An action is a `int` number and represents a direction according to the following scheme:

- 0: LEFT
- 1: DOWN
- 2: RIGHT
- 3: UP

### Observation Space

The observation is a value representing the agent's current position as
`current_row * ncols + current_col` (where both the row and col start at 0).
For example, the point in position `(2,3)` in a 4x5 map corresponds to state 13 (= 2 * 5 + 3).
The number of possible observations is dependent on the size of the map.
For example, the 4x4 map has 16 possible observations.

### Rewards

It is possible to customize the rewards for each state by passing a custom reward map through the argument `reward_map`.

The default reward schedule is:

- goal(G): +1
- wall(W): -1
- empty(E): 0
- start(S): 0

## Notes on rendering

The default frame rate is 5 FPS. It is possible to change it through `env.fps` after instantiating the environment.

To properly render the environment, remember that the point (x,y) in the desc matrix corresponds to the point (y,x) in the rendered matrix.
This is because the rendering code works in terms of width and height while the computation in the environment is done using x and y coordinates.
You don't have to worry about this unless you play with the environment's internals.

## Version History

v0: Initial versions release (1.0.0)