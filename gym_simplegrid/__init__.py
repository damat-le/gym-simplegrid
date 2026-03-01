from gymnasium.envs.registration import register

# --- Pre-defined Maps for Registration ---
# These are the "standard" maps previously hardcoded in the env.
MAP_4x4 = ["0000", "0101", "0001", "1000"]

MAP_8x8 = [
    "00000000",
    "00000000",
    "00010000",
    "00000100",
    "00010000",
    "01100010",
    "01001010",
    "00010000",
]

# --- Registration ---

# Generic registration: User must provide 'obstacle_map' in gym.make()
register(
    id='SimpleGrid-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
)

# Pre-set 4x4 Map
register(
    id='SimpleGrid-4x4-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': MAP_4x4},
)

# Pre-set 8x8 Map
register(
    id='SimpleGrid-8x8-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': MAP_8x8},
)