from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

from ..parser import MapParser
from ..grid import GridModel
from ..renderer import GridRenderer


class SimpleGridEnv(gym.Env):
    """
    A pure state-machine implementation of the SimpleGrid environment.

    This version removes all rendering side-effects from the logic. 
    Rendering only occurs when the user explicitly calls `env.render()`.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    MOVES: dict[int, tuple[int, int]] = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def __init__(
        self,
        obstacle_map: list[str],
        render_mode: str | None = None,
        render_fps: int = 8,
        max_steps: int | None = None
    ):
        # Logic components
        grid_array = MapParser.parse(obstacle_map)
        self.model = GridModel(grid_array)
        self.max_steps = max_steps

        # Spaces
        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(self.model.nrow * self.model.ncol)

        # Rendering component (Only initialized if mode is set)
        self.render_mode = render_mode
        self.renderer = GridRenderer(self.model, render_mode, render_fps)

        # State
        self.agent_xy: tuple[int, int] = (0, 0)
        self.start_xy: tuple[int, int] = (0, 0)
        self.goal_xy: tuple[int, int] = (0, 0)
        self.step_count: int = 0
        self.last_reward: float = 0.0
        self.last_action: int | None = None

    def reset(
        self, 
        seed: int | None = None, 
        options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to an initial state. 
        
        The start and goal positions can be specified via options.
        If start_loc or goal_loc is not provided, they will be randomly sampled from valid positions.
        """
        super().reset(seed=seed)
        options = options or {}

        self.start_xy = self._parse_loc_option(options.get("start_loc"))
        self.goal_xy = self._parse_loc_option(options.get("goal_loc"))
        self._check_integrity()

        self.agent_xy = self.start_xy
        self.step_count = 0
        self.last_reward = 0.0
        self.last_action = None

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment based on the action.
        """

        self.last_action = action
        row, col = self.agent_xy
        
        # Compute target position based on action
        dr, dc = self.MOVES[action]
        target_xy = self.model.get_next_xy(row, col, dr, dc)
        is_valid = self.model.is_in_bounds(*target_xy) and self.model.is_free(*target_xy)
        
        # State Update
        self.agent_xy = target_xy if is_valid else self.agent_xy
        
        # Reward Calculation
        self.last_reward = self.get_reward(target_xy)
        
        # Step count update
        self.step_count += 1

        # Check Termination (Goal reached)
        terminated = self.agent_xy == self.goal_xy
        
        # Check Truncation (Step limit reached)
        truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps:
            # If we reached the goal on the exact last step, it is terminated, not truncated.
            if not terminated:
                truncated = True

        return self._get_obs(), self.last_reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray | str | None:
        """
        This method must be explicitly called by the user or a wrapper.
        """
        return self.renderer.render(
            agent_xy=self.agent_xy,
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            step_count=self.step_count,
            last_reward=self.last_reward,
            done=self.agent_xy == self.goal_xy,
            last_action=self.last_action
        )

    def close(self) -> None:
        self.renderer.close()

    def get_reward(
        self, 
        xy: tuple[int, int], 
    ) -> float:
        """
        Logic for reward calculation. Overload this to change behavior.
        """
        if not self._is_valid_xy(xy):
            return -1.0         # Penalty for invalid move
        if xy == self.goal_xy:
            return 1.0          # Reward for reaching the goal
        return -0.1             # Step penalty to encourage shorter paths

    def _get_obs(self) -> int:
        return self.model.to_index(*self.agent_xy)

    def _get_info(self) -> dict[str, Any]:
        return {"agent_xy": self.agent_xy, "step_count": self.step_count}
    
    def _is_valid_xy(self, xy: tuple[int, int]) -> bool:
        """
        Check if the given (row, col) coordinate is within bounds and not an obstacle.
        """
        return self.model.is_in_bounds(*xy) and self.model.is_free(*xy)

    def _parse_loc_option(self, loc: Any) -> tuple[int, int]:
        """
        Parse a location option which can be None, an integer index, or a (row, col) tuple.

        Returns a valid (row, col) coordinate. 
        If loc is None, it samples a random valid position.
        If loc is an integer, it converts it to (row, col).
        If loc is already a tuple, return it as it is.
        """
        if loc is None:
            return self.model.sample_valid_xy(self.np_random)
        return self.model.to_xy(loc) if isinstance(loc, int) else loc

    def _check_integrity(self) -> None:
        for name, pos in [("Start", self.start_xy), ("Goal", self.goal_xy)]:
            if not self._is_valid_xy(pos):
                raise ValueError(f"Invalid {name} position: {pos}")