from __future__ import annotations
import random
import numpy as np
from gym import spaces, Env
import gym_simplegrid.rendering as r
from gym_simplegrid.window import Window

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

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when all agents reach their goals.

    To initialise the grid, the user must decide where to put the walls on the grid. This can bee done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000", 
     "0101", 
     "0001", 
     "1000"]

    The user must also decide the number of agents and their starting and goal positions on the grid. This can be done by passing two lists of tuples, namely `starts_xy` and `goals_xy`, where each tuple is a pair of coordinates (x, y) representing the agent starting/goal position. 

    Currently, the user must also define the color of each agent. This can be done by passing a list of strings, where each string is a color name. The available color names are: red, green, blue, purple, yellow, grey and black. This requirement will be removed in the future and the color will be assigned automatically.

    The user can also decide whether the agents disappear when they reach their goal. This can be done by passing a boolean value to `disappear_on_goal`. If `disappear_on_goal` is True, the agent disappears when it reaches its goal. If `disappear_on_goal` is False, the agent remains on the grid after it reaches its goal. This feature is currently not implemented and will be added in future versions.
    """
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
        start_xy: tuple, 
        goal_xy: tuple, 
        agent_color: str = 'yellow', 
        seed: int | None = None   
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        start_xy: tuple
            Starting (x,y) position of the agent.
        goals_xy: tuple
            Goal (x,y) position of the agent.
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
        self.seed = seed        
        
        # Env confinguration
        self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
        self.nrow, self.ncol = self.obstacles.shape

        self.start_xy = start_xy
        self.goal_xy = goal_xy

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)

        # internal vars initialised in reset()
        self.curr_pos_xy: tuple = None
        self.curr_reward: float = 0.0
        self.done: bool = False
        self.info: dict = dict()

        # Rendering configuration
        self.window = None
        self.agent_color = agent_color
        self.tile_cache = {}
        self.fps = 10

    def set_seed(self, seed: int) -> None:
        """
        Seed the environment.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        np.random.seed(seed)
        random.seed(seed)

    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        
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
        return self.curr_pos_xy == self.goal_xy

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

    def move(self, action: int) -> None:
        """
        Move the agent according to the selected action.
        """
        #assert action in self.action_space

        # Get the current position of the agent
        row, col = self.curr_pos_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.curr_reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        if not self.done and self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.curr_pos_xy = (target_row, target_col)
            self.done = self.on_goal()
        
    def step(self, action: int):
        """
        Take a step in the environment.
        """
        self.move(action)
        return self.observation()
    
    def reset(self) -> tuple:
        """
        Reset the environment.

        By deafult, the agents are reset to the starting positions indicated during class initialisation. However, the user can also pass a dict of new starting positions.
        """
        # Set seed
        self.set_seed(self.seed)
        # Reset agent position
        self.curr_pos_xy = self.start_xy
        # Reset reward, done and info
        self.curr_reward = self.get_reward(*self.curr_pos_xy)
        self.done = self.on_goal()
        self.info = {}
        # Check integrity
        self.integrity_checks()

        return self.observation()
    
    def observation(self) -> tuple:
        return self.curr_pos_xy, self.curr_reward, self.done, self.info

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return -1.0
        elif not self.is_free(x, y):
            return -5.0
        elif (x, y) == self.goal_xy:
            return 1.0
        else:
            return 0.0

    def close(self):
        """
        Close the environment.
        """
        if self.window:
            self.window.close()
        return None

    def render(self, caption=None, mode='human'):
        """
        Render the environment.
        """
        if mode == "human":
            return self.render_gui(caption=caption)
        else:
            raise ValueError(f"Unsupported rendering mode {mode}")
    
    def render_gui(self, caption, tile_size=r.TILE_PIXELS, highlight_mask=None):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) to the grid.render method.

        tile_size: tile size in pixels
        """
        width = self.ncol
        height = self.nrow

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render grid with obstacles
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.obstacles[x,y] == self.OBSTACLE:
                    cell = r.Wall(color='black')
                else:
                    cell = None

                img = self.update_cell_in_img(img, x, y, cell, tile_size)

        # Render start
        x, y = self.start_xy
        cell = r.ColoredTile(color="red")
        img = self.update_cell_in_img(img, x, y, cell, tile_size)

        # Render goal
        x, y = self.goal_xy
        cell = r.ColoredTile(color="green")
        img = self.update_cell_in_img(img, x, y, cell, tile_size)

        # Render agent
        x, y = self.curr_pos_xy
        cell = r.Agent(color=self.agent_color)
        img = self.update_cell_in_img(img, x, y, cell, tile_size)

        if not self.window:
            self.window = Window('my_custom_env')
            self.window.show(block=False)
        self.window.show_img(img, caption, self.fps)

        return img
        
    def render_tile(
        self,
        obj: r.WorldObj,
        highlight=False,
        tile_size=r.TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        if not isinstance(obj, r.Agent):
            key = (None, highlight, tile_size)
            key = obj.encode() + key if obj else key

            if key in self.tile_cache:
                return self.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) + 255

        if obj != None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            r.highlight_img(img)

        # Draw the grid lines (top and left edges)
        r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (170, 170, 170))
        r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (170, 170, 170))

        # Downsample the image to perform supersampling/anti-aliasing
        img = r.downsample(img, subdivs)

        # Cache the rendered tile
        if not isinstance(obj, r.Agent):
            self.tile_cache[key] = img

        return img

    def update_cell_in_img(self, img, x, y, cell, tile_size):
        """
        Parameters
        ----------
        img : np.ndarray
            Image to update.
        x : int
            x-coordinate of the cell to update.
        y : int
            y-coordinate of the cell to update.
        cell : r.WorldObj
            New cell to render.
        tile_size : int
            Size of the cell in pixels.
        """
        tile_img = self.render_tile(cell, tile_size=tile_size)
        height_min = x * tile_size
        height_max = (x+1) * tile_size
        width_min = y * tile_size
        width_max = (y+1) * tile_size
        img[height_min:height_max, width_min:width_max, :] = tile_img
        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
    #     width = self.ncol
    #     height = self.nrow

    #     if vis_mask is None:
    #         vis_mask = np.ones((width, height), dtype=bool)

    #     array = np.zeros((width, height, 3), dtype='uint8')

    #     for i in range(width):
    #         for j in range(height):
    #             if vis_mask[i, j]:
    #                 v = self.get(i, j)

    #                 if v is None:
    #                     array[i, j, 0] = r.OBJECT_TO_IDX['empty']
    #                     array[i, j, 1] = 0
    #                     array[i, j, 2] = 0

    #                 else:
    #                     array[i, j, :] = v.encode()

    #     return array
        pass

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

    #     width, height, channels = array.shape
    #     assert channels == 3

    #     vis_mask = np.ones(shape=(width, height), dtype=bool)

    #     grid = SimpleGrid(width, height)
    #     for i in range(width):
    #         for j in range(height):
    #             type_idx, color_idx, state = array[i, j]
    #             v = WorldObj.decode(type_idx, color_idx, state)
    #             grid.set(i, j, v)
    #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

    #     return grid, vis_mask
        pass