from __future__ import annotations
import numpy as np


class GridModel:
    """
    Representation and logic engine of the grid world.

    This class handles coordinate transformations, boundary checks, 
    occupancy logic, and state transitions. It is independent of 
    rendering and Gymnasium-specific logic.

    Parameters
    ----------
    np_map : np.ndarray
        A 2D NumPy array of type `np.int8` where 0 is free and 1 is an obstacle.

    Attributes
    ----------
    nrow : int
        Number of rows in the grid.
    ncol : int
        Number of columns in the grid.
    grid : np.ndarray
        The underlying spatial data.
    """

    FREE: int = 0
    OBSTACLE: int = 1

    def __init__(self, np_map: np.ndarray):
        self.grid = np_map
        self.nrow, self.ncol = np_map.shape

    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a coordinate is within the grid boundaries.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        bool
            True if the coordinate is within bounds, False otherwise.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is walkable (not an obstacle).

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        bool
            True if the cell is free and in bounds, False otherwise.
        """
        if not self.is_in_bounds(row, col):
            return False
        return self.grid[row, col] == self.FREE

    def to_index(self, row: int, col: int) -> int:
        """
        Flatten (row, col) coordinates into a single integer state.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        int
            The flattened state index.
        """
        return row * self.ncol + col

    def to_xy(self, index: int) -> tuple[int, int]:
        """
        Convert a flattened state index back into (row, col) coordinates.

        Parameters
        ----------
        index : int
            Flattened state index.

        Returns
        -------
        tuple[int, int]
            A tuple containing (row, col).
        """
        return (index // self.ncol, index % self.ncol)

    def get_next_xy(
        self, 
        row: int, 
        col: int, 
        d_row: int, 
        d_col: int
    ) -> tuple[int, int]:
        """
        Calculate the next position given a move.

        NOTE: This method does not check for validity of the move. 
        It simply computes the target coordinates.

        Parameters
        ----------
        row : int
            Current row.
        col : int
            Current column.
        d_row : int
            Change in row (delta).
        d_col : int
            Change in column (delta).

        Returns
        -------
        tuple[int, int]
            The new (row, col) position.
        """
        target_row = row + d_row
        target_col = col + d_col
        return (target_row, target_col)

    def sample_valid_xy(self, np_random: np.random.Generator) -> tuple[int, int]:
        """
        Randomly sample a walkable coordinate from the grid.

        Parameters
        ----------
        np_random : np.random.Generator
            The random number generator to use.

        Returns
        -------
        tuple[int, int]
            A randomly selected (row, col) that is not an obstacle.
        """
        # Get indices of all free cells
        free_rows, free_cols = np.where(self.grid == self.FREE)

        if free_rows.size == 0:
            raise RuntimeError("The grid contains no free cells to sample from.")
        
        idx = np_random.integers(0, len(free_rows))
        
        return (int(free_rows[idx]), int(free_cols[idx]))