from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

MAPS = {
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
}

class SimpleGridEnv(Env):
    """
    Simple Grid Environment

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when the agents reaches its goal.

    To initialise the grid, the user must decide where to put the walls on the grid. This can be done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000", 
     "0101", 
     "0001", 
     "1000"]

    Assume the environment is a grid of size (nrow, ncol). A state s of the environment is an elemente of gym.spaces.Discete(nrow*ncol), i.e. an integer between 0 and nrow * ncol - 1. Assume nrow=ncol=5 and s=10, to compute the (x,y) coordinates of s on the grid the following formula are used: x = s // ncol  and y = s % ncol.
     
    The user can also decide the starting and goal positions of the agent. This can be done by through the `options` dictionary in the `reset` method. The user can specify the starting and goal positions by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type int (s) or tuple (x,y) and represent the agent starting and goal positions respectively. 
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(self,     
        obstacle_map: str | list[str],
        render_mode: str | None = None,
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        agent_color: str
            Color of the agent. The available colors are: red, green, blue, purple, yellow, grey and black. Note that the goal cell will have the same color.
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. 
            An example of a 4x4 map is the following:
            ["0000",
             "0101", 
             "0001",
             "1000"]
        """

        # Env confinguration
        self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(n=self.nrow*self.ncol)

        # Rendering configuration
        self.fig = None

        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']

    def reset(
            self, 
            seed: int | None = None, 
            options: dict = dict()
        ) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        # parse options
        self.start_xy = self.parse_state_option('start_loc', options)
        self.goal_xy = self.parse_state_option('goal_loc', options)

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # Check integrity
        self.integrity_checks()

        #if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.get_info()
    
    def step(self, action: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space
        self.agent_action = action

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        self.n_iter += 1

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.reward, self.done, False, self.get_info()
    
    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "010", "011]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        elif isinstance(obstacle_map, str):
            map_str = MAPS[obstacle_map]
            map_str = np.asarray(map_str, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        else:
            raise ValueError(f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}.")
        
    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f'Allowed types for `{state_name}` are int or tuple.')
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(f'Key `{state_name}` not found in `options`. Random sampling a valid value for it:')
            logger.info(f'...`{state_name}` has value: {state}')
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.observation_space.sample()
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.observation_space.sample()
            pos_xy = self.to_xy(state)
        return pos_xy
    
    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."
        
    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == self.FREE
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return -1.0
        elif not self.is_free(x, y):
            return -1.0
        elif (x, y) == self.goal_xy:
            return 1.0
        else:
            return 0.0

    def get_obs(self) -> int:
        return self.to_s(*self.agent_xy)
    
    def get_info(self) -> dict:
        return {
            'agent_xy': self.agent_xy,
            'n_iter': self.n_iter,
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
        
        elif self.render_mode == "ansi":
            s = f"{self.n_iter},{self.agent_xy[0]},{self.agent_xy[1]},{self.reward},{self.done},{self.agent_action}\n"
            #print(s)
            return s

        elif self.render_mode == "rgb_array":
            self.render_frame()
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
        elif self.render_mode == "human":
            self.render_frame()
            plt.pause(1/self.fps)
            return None
        
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self):
        if self.fig is None:
            self.render_initial_frame()
            self.fig.canvas.mpl_connect('close_event', self.close)
        else:
            self.update_agent_patch()
        self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward}")
    
    def create_agent_patch(self):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        return mpl.patches.Circle(
            (self.agent_xy[1]+.5, self.agent_xy[0]+.5), 
            0.3, 
            facecolor='orange', 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        @NOTE: If agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) as center to the Circle patch.
        """
        self.agent_patch.center = (self.agent_xy[1]+.5, self.agent_xy[0]+.5)
        return None
    
    def render_initial_frame(self):
        """
        Render the initial frame.

        @NOTE: 0: free cell (white), 1: obstacle (black), 2: start (red), 3: goal (green)
        """
        data = self.obstacles.copy()
        data[self.start_xy] = 2
        data[self.goal_xy] = 3

        colors = ['white', 'black', 'red', 'green']
        bounds=[i-0.1 for i in [0, 1, 2, 3, 4]]

        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        #ax.grid(axis='both', color='#D3D3D3', linewidth=2) 
        ax.grid(axis='both', color='k', linewidth=1.3) 
        ax.set_xticks(np.arange(0, data.shape[1], 1))  # correct grid sizes
        ax.set_yticks(np.arange(0, data.shape[0], 1))
        ax.tick_params(
            bottom=False, 
            top=False, 
            left=False, 
            right=False, 
            labelbottom=False, 
            labelleft=False
        ) 

        # draw the grid
        ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            extent=[0, data.shape[1], data.shape[0], 0],
            interpolation='none'
        )

        # Create white holes on start and goal positions
        for pos in [self.start_xy, self.goal_xy]:
            wp = self.create_white_patch(*pos)
            ax.add_patch(wp)

        # Create agent patch in start position
        self.agent_patch = self.create_agent_patch()
        ax.add_patch(self.agent_patch)

        return None

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y+.5, x+.5), 
            0.4, 
            color='white', 
            fill=True, 
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        plt.close(self.fig)
        sys.exit()