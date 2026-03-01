from __future__ import annotations
import numpy as np


class MapParser:
    """
    High-performance utility for parsing and validating grid-based maps.
    """

    @staticmethod
    def parse(obstacle_map: list[str]) -> np.ndarray:
        """
        Parse a list of strings into a 2D integer NumPy array using vectorized operations.


        Parameters
        ----------
        obstacle_map : list of str
            A list of strings where each string represents a row of the grid.
            '0' denotes a free cell and '1' denotes an obstacle.

        Returns
        -------
        grid : np.ndarray
            A 2D NumPy array of type `np.int8` containing 0s and 1s.

        Raises
        ------
        ValueError
            If the map is empty, rows have inconsistent lengths, or 
            contains characters other than '0' and '1'.
        """
        if not obstacle_map:
            raise ValueError("The provided obstacle map is empty.")

        # Validate consistency
        n_rows = len(obstacle_map)
        n_cols = len(obstacle_map[0])
        if not all(len(row) == n_cols for row in obstacle_map):
            raise ValueError("Inconsistent column lengths detected in obstacle_map.")

        # Vectorized conversion: 
        # Convert list of strings to a NumPy byte array (S format).
        # This is performed in optimized C code.
        try:
            # We use dtype='S' to get fixed-width byte strings
            # .view(np.uint8) gives us the ASCII values (e.g., '0' -> 48, '1' -> 49)
            raw_bytes = np.array(obstacle_map, dtype='S').view(np.uint8)
            
            # Reshape
            grid = raw_bytes.reshape(n_rows, n_cols)
            
            # Convert ASCII '0'/'1' (48/49) to integers 0/1 by subtracting 48
            # This is a vectorized subtraction (fast)
            grid = (grid - 48).astype(np.int8)

        except Exception as e:
            raise ValueError(f"Failed to parse map: {e}") from e

        # Vectorized validation
        # Check if any value is not 0 or 1 in one pass
        if not np.all((grid == 0) | (grid == 1)):
            raise ValueError(
                "Invalid characters detected. Only '0' and '1' are allowed."
            )

        return grid