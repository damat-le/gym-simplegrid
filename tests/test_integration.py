import gymnasium as gym
import pytest
import gym_simplegrid # Triggers registration

def test_gym_make():
    """Test that registered envs can be instantiated via gym.make."""
    env = gym.make("SimpleGrid-4x4-v0")
    assert env.unwrapped.model.nrow == 4
    env.close()

def test_custom_make():
    """Test gym.make with custom map."""
    custom_map = ["00", "01"]
    env = gym.make("SimpleGrid-v0", obstacle_map=custom_map)
    assert env.unwrapped.model.nrow == 2
    env.close()