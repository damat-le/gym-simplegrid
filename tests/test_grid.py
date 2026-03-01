import pytest
import numpy as np
from gym_simplegrid.grid import GridModel

@pytest.fixture
def grid_model():
    """Fixture for a 3x3 grid with one obstacle at (1,1)."""
    np_map = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    return GridModel(np_map)

def test_grid_dimensions(grid_model):
    """Test grid dimensions reporting."""
    assert grid_model.nrow == 3
    assert grid_model.ncol == 3

def test_coordinate_conversions(grid_model):
    """Test index to XY and vice-versa."""
    assert grid_model.to_index(1, 2) == 5
    assert grid_model.to_xy(5) == (1, 2)

@pytest.mark.parametrize("row, col, expected", [
    (0, 0, True),
    (1, 1, False),  # Obstacle
    (3, 0, False),  # Out of bounds
    (-1, 0, False), # Out of bounds
])
def test_is_free(grid_model, row, col, expected):
    """Test cell walkable check."""
    assert grid_model.is_free(row, col) == expected

def test_get_next_xy_valid(grid_model):
    """Test valid movement."""
    # From (0,0) go Right -> (0,1)
    assert grid_model.get_next_xy(0, 0, 0, 1) == (0, 1)

def test_sample_valid_xy(grid_model):
    """Test random valid sampling."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        pos = grid_model.sample_valid_xy(rng)
        assert grid_model.is_free(*pos)
