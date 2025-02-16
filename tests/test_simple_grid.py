from sys import path
path.append('.') # don't understand why this is necessary

import numpy as np
import pytest
from gym_simplegrid.envs import SimpleGridEnv

test_map = [
    "0010",
    "0010",
    "0110",
    "0000"
]

@pytest.fixture
def sgenv():
    """Fixture to create a SimpleGridEnv instance."""
    return SimpleGridEnv(
        obstacle_map=test_map,
        render_mode='ansi'
    )


# -----------------------------

expected_map = np.array([
    [0, 0, 1, 0], 
    [0, 0, 1, 0], 
    [0, 1, 1, 0], 
    [0, 0, 0, 0]
])

@pytest.mark.parametrize("obstacle_map, expected_map", [
    (test_map, expected_map)
])
def test_parse_obstacle_map(sgenv, obstacle_map, expected_map):
    assert (sgenv.parse_obstacle_map(obstacle_map) == expected_map).all()

# -----------------------------

@pytest.mark.parametrize("action, expected_state", [
    (0, (0,1)), #UP
    (1, (1,1)), #DOWN
    (2, (1,0)), #LEFT
    (3, (1,1))  #RIGHT
])
def test_step(sgenv, action, expected_state):
    sgenv.reset(seed=42, options={'start_loc': (1,1), 'goal_loc': (3,3)})
    sgenv.step(action)
    assert sgenv.agent_xy == expected_state

# -----------------------------

