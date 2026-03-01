import pytest
import numpy as np
from gym_simplegrid.parser import MapParser

def test_parse_valid_map():
    """Test parsing a valid list of strings into a numpy array."""
    obstacle_map = ["001", "010"]
    expected = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int8)
    result = MapParser.parse(obstacle_map)
    assert np.array_equal(result, expected)
    assert result.dtype == np.int8

def test_parse_inconsistent_lengths():
    """Test that maps with varying row lengths raise ValueError."""
    obstacle_map = ["000", "01"]
    with pytest.raises(ValueError, match="Inconsistent column lengths"):
        MapParser.parse(obstacle_map)

def test_parse_invalid_characters():
    """Test that invalid characters raise ValueError."""
    obstacle_map = ["002", "010"]
    with pytest.raises(ValueError, match="Invalid characters detected"):
        MapParser.parse(obstacle_map)

def test_parse_empty_map():
    """Test that an empty map list raises ValueError."""
    with pytest.raises(ValueError, match="provided obstacle map is empty"):
        MapParser.parse([])
        