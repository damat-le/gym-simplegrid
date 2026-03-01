import pytest
import gymnasium as gym
from gym_simplegrid.envs.simple_grid import SimpleGridEnv

@pytest.fixture
def env():
    """Fixture for a simple 3x3 environment."""
    obstacle_map = ["000", "010", "000"]
    return SimpleGridEnv(obstacle_map=obstacle_map, max_steps=10)

@pytest.fixture
def env_ansi_render_mode():
    """Fixture for environment with specific start and goal."""
    obstacle_map = ["000", "010", "000"]
    return SimpleGridEnv(obstacle_map=obstacle_map, max_steps=10, render_mode="ansi")

def test_env_reset(env):
    """Test reset returns correct observation and info."""
    obs, info = env.reset(seed=42)
    assert isinstance(obs, int)
    assert "agent_xy" in info
    assert env.step_count == 0

def test_env_reset_options(env):
    """Test reset with specific start and goal locations."""
    options = {"start_loc": (0, 0), "goal_loc": (2, 2)}
    obs, _ = env.reset(options=options)
    assert env.agent_xy == (0, 0)
    assert env.goal_xy == (2, 2)
    assert obs == env.model.to_index(0, 0)

def test_env_step_logic(env):
    """Test movement and termination."""
    env.reset(options={"start_loc": (0, 0), "goal_loc": (0, 1)})
    
    # Action 3: Right
    obs, reward, terminated, truncated, info = env.step(3)
    
    assert env.agent_xy == (0, 1)
    assert terminated is True
    assert reward == 1.0  # Goal reward

def test_env_step_penalty(env):
    """Test step penalty and invalid move penalty."""
    env.reset(options={"start_loc": (0, 0), "goal_loc": (2, 2)})
    
    # Move into wall (Action 1: Down from (0,1) into (1,1))
    env.agent_xy = (0, 1)
    _, reward, _, _, _ = env.step(1)
    assert reward == -1.0
    assert env.agent_xy == (0, 1) # Stays put

def test_env_truncation(env):
    """Test truncation when max_steps is reached."""
    env.reset(options={"start_loc": (0, 0), "goal_loc": (2, 2)})
    
    # Take 10 steps (max_steps is 10)
    for _ in range(9):
        _, _, _, truncated, _ = env.step(0)
        assert truncated is False
        
    _, _, _, truncated, _ = env.step(0)
    assert truncated is True

def test_render_ansi(env_ansi_render_mode):
    """Test ANSI rendering mode output."""
    env = env_ansi_render_mode
    env.reset(options={"start_loc": (0, 0), "goal_loc": (2, 2)})
    out = env.render()
    assert isinstance(out, str)
    assert "0,0,0,0.0,False,None" in out