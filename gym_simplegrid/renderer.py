from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import TYPE_CHECKING, Any
from .grid import GridModel

class GridRenderer:
    """
    Renderer for the SimpleGrid environment using Matplotlib.

    This class encapsulates all visual logic, keeping the environment 
    class clean of plotting code. It supports 'human', 'rgb_array', 
    and 'ansi' modes.

    Parameters
    ----------
    grid_model : GridModel
        The logical model of the grid to be rendered.
    render_mode : str, optional
        The mode to use for rendering ('human', 'rgb_array', 'ansi').
    render_fps : int, optional
        Frames per second for 'human' mode.
    """

    def __init__(
        self, 
        grid_model: GridModel, 
        render_mode: str | None = None, 
        render_fps: int = 8
    ):
        self.grid_model = grid_model
        self.render_mode = render_mode
        self.fps = render_fps

        # Visual Constants (Identical to original)
        self.CELL_COLORS = ['white', 'black', 'red', 'green']
        self.AGENT_COLOR = 'orange'
        self.AGENT_RADIUS = 0.3
        self.HOLE_RADIUS = 0.4
        
        # Internal state for Matplotlib
        self.fig: mpl.figure.Figure | None = None
        self.ax: mpl.axes.Axes | None = None
        self.agent_patch: mpl.patches.Circle | None = None
        self.img_handle: mpl.image.AxesImage | None = None

    def render(
        self, 
        agent_xy: tuple[int, int], 
        start_xy: tuple[int, int], 
        goal_xy: tuple[int, int],
        step_count: int,
        last_reward: float,
        done: bool,
        last_action: int | None
    ) -> np.ndarray | str | None:
        """
        Main render entry point.

        Parameters
        ----------
        agent_xy : tuple[int, int]
            Current (row, col) of the agent.
        start_xy : tuple[int, int]
            The starting (row, col) position.
        goal_xy : tuple[int, int]
            The goal (row, col) position.
        step_count : int
            Current iteration count.
        last_reward : float
            The reward received in the last step.
        done : bool
            Whether the episode is finished.
        last_action : int, optional
            The action that led to the current state.

        Returns
        -------
        np.ndarray | str | None
            RGB array if 'rgb_array', CSV string if 'ansi', else None.
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "ansi":
            return f"{step_count},{agent_xy[0]},{agent_xy[1]},{last_reward},{done},{last_action}\n"

        # Initialize or update the Matplotlib figure
        if self.fig is None:
            self._initial_render(agent_xy, start_xy, goal_xy)
        else:
            self._update_render(agent_xy)

        self.ax.set_title(f"Step: {step_count}, Reward: {last_reward}")

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.array(self.fig.canvas.renderer.buffer_rgba())

        elif self.render_mode == "human":
            plt.pause(1 / self.fps)
            return None
        
        return None

    def _initial_render(
        self, 
        agent_xy: tuple[int, int], 
        start_xy: tuple[int, int], 
        goal_xy: tuple[int, int]
    ) -> None:
        """Initialize the figure, grid, and patches."""
        plt.ion()
        self.fig, self.ax = plt.subplots(tight_layout=True)
        
        # Prepare the background data (0:free, 1:obstacle, 2:start, 3:goal)
        data = self.grid_model.grid.copy()
        data[start_xy] = 2
        data[goal_xy] = 3

        # Create discrete colormap
        cmap = mpl.colors.ListedColormap(self.CELL_COLORS)
        bounds = [i - 0.1 for i in range(len(self.CELL_COLORS) + 1)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Draw the static grid
        self.img_handle = self.ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            extent=[0, self.grid_model.ncol, self.grid_model.nrow, 0],
            interpolation='none'
        )

        # Formatting
        self.ax.grid(axis='both', color='k', linewidth=1.3)
        self.ax.set_xticks(np.arange(0, self.grid_model.ncol, 1))
        self.ax.set_yticks(np.arange(0, self.grid_model.nrow, 1))
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Create white "holes" for start and goal
        for pos in [start_xy, goal_xy]:
            hole = mpl.patches.Circle(
                (pos[1] + 0.5, pos[0] + 0.5), 
                self.HOLE_RADIUS, 
                color='white', 
                fill=True, 
                zorder=99
            )
            self.ax.add_patch(hole)

        # Create agent patch
        self.agent_patch = mpl.patches.Circle(
            (agent_xy[1] + 0.5, agent_xy[0] + 0.5), 
            self.AGENT_RADIUS, 
            facecolor=self.AGENT_COLOR, 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100
        )
        self.ax.add_patch(self.agent_patch)

        # Handle window close event
        self.fig.canvas.mpl_connect('close_event', lambda _: self.close())

    def _update_render(self, agent_xy: tuple[int, int]) -> None:
        """Efficiently update only the agent position."""
        if self.agent_patch:
            self.agent_patch.center = (agent_xy[1] + 0.5, agent_xy[0] + 0.5)

    def close(self) -> None:
        """Close the figure without exiting the process."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.agent_patch = None