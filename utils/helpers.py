"""
Helper functions for the snake game
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_adjacent_positions(
    pos: Tuple[int, int], grid_size: int
) -> List[Tuple[int, int]]:
    """Get valid adjacent positions (up, down, left, right)"""
    x, y = pos
    adjacent = []

    # Up
    if y > 0:
        adjacent.append((x, y - 1))
    # Down
    if y < grid_size - 1:
        adjacent.append((x, y + 1))
    # Left
    if x > 0:
        adjacent.append((x - 1, y))
    # Right
    if x < grid_size - 1:
        adjacent.append((x + 1, y))

    return adjacent


def is_position_safe(
    pos: Tuple[int, int], snake_body: List[Tuple[int, int]], grid_size: int
) -> bool:
    """Check if position is safe (within bounds and not on snake)"""
    x, y = pos
    # Check bounds
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    # Check if on snake
    if pos in snake_body:
        return False
    return True


def save_json_data(data: Dict[Any, Any], filepath: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json_data(filepath: str) -> Dict[Any, Any]:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


def get_latest_model_info(model_dir: str) -> Dict[str, Any]:
    """Get information about the latest saved model"""
    if not os.path.exists(model_dir):
        return {}

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
    if not model_files:
        return {}

    # Sort by modification time (latest first)
    model_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True
    )
    latest_model = model_files[0]

    return load_json_data(os.path.join(model_dir, latest_model))
