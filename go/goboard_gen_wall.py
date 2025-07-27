"""
Go board wall/obstacle generation system.

This module provides functions to add various types of obstacles and walls to Go boards,
creating interesting and challenging board configurations for AI training and gameplay.

Converted from TypeScript obstacle generation system.
"""

import random

from gotypes import Point

# Board size constants (small to large)
BOARD_SIZES = [5, 7, 9, 13, 19]


def gen_wall(board_size: int) -> set[Point]:
    """
    Add various types of obstacles to a Go board.
    
    Args:
        board_obstacle: BoardObstacle instance to modify
        seed: Optional random seed for reproducible results
        
    Returns:
        Modified BoardObstacle with added obstacles
    """

    board = [ [ "." for _ in range(board_size) ] for _ in range(board_size) ]
    
    # Decide which obstacle types to add
    should_remove_corner = random.randint(0, 4) == 0
    should_remove_rows = not should_remove_corner and random.randint(0, 4) == 0
    should_add_center_break = (not should_remove_corner and 
                              not should_remove_rows and 
                              random.randint(0, 3) > 0)
    
    obstacle_type_count = (int(should_remove_corner) + 
                          int(should_remove_rows) + 
                          int(should_add_center_break))
    
    scale = get_scale(board_size)
    edge_dead_count = random.randint(1, int((scale + 2 - obstacle_type_count) * 1.5))
    
    # Apply obstacle generation in sequence
    if should_remove_corner:
        board = add_dead_corners(board)
    
    if should_add_center_break:
        board = add_center_break(board)
    
    board = randomize_rotation(board)
    
    if should_remove_rows:
        board = remove_rows(board)
    
    board = add_dead_nodes_to_edge(board, edge_dead_count)
    
    board = ensure_offline_nodes(board)
    
    points = set()
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == "#":
                points.add(Point(i + 1, j + 1))
    return points


def get_scale(size: int) -> int:
    """Get the scale index of the board based on its size."""
    return BOARD_SIZES.index(size)

def remove_rows(board: list[list[str]]) -> list[list[str]]:
    """Remove rows from the board by adding obstacles."""
    rows_to_remove = max(random.randint(-2, get_scale(len(board))), 1)
    
    for row_idx in range(rows_to_remove):
        for col in range(len(board)):
            board[row_idx][col] = "#"
    
    # Simulate rotation by applying to different edges
    return rotate_n_times(board, 3)


def add_dead_nodes_to_edge(board: list[list[str]], max_per_edge: int) -> list[list[str]]:
    """Add dead nodes to the edges of the board."""
    size = len(board)
    
    # Add obstacles to each edge (top, right, bottom, left)
    for _ in range(4):
        count = random.randint(0, max_per_edge)
        for _ in range(count):
          y = max(random.randint(-2, size - 1), 0)
          board[0][y] = "#"
    
        rotate_n_times(board, 1)
    return board


def add_dead_corners(board: list[list[str]]) -> list[list[str]]:
    """Add dead corners to the board."""
    scale = get_scale(len(board)) + 1
    
    # Add first corner
    board = add_dead_corner(board, scale)
    
    # 25% chance to add opposite corner
    if random.randint(0, 3) == 0:
        board = rotate_n_times(board, 2)
        board = add_dead_corner(board, scale - 2)
    
    return randomize_rotation(board)


def add_dead_corner(board: list[list[str]], size: int) -> list[list[str]]:
    """
    Add a dead corner pattern.
    """
    current_size = size
    
    for i in range(min(size, len(board))):
        if random.randint(0, 1) == 1:
            current_size -= 1
            
        for j in range(min(current_size, len(board))):
            if board[i][j] == ".":
                board[i][j] = "#"

    return board


def add_center_break(board: list[list[str]]) -> list[list[str]]:
    """Add a break in the center of the board."""
    size = len(board)
    max_offset = get_scale(size)
    
    x_index = random.randint(0, max_offset * 2) - max_offset + size // 2
    length = random.randint(1, int(size / 2.0 - 1))
    
    for i in range(length):
        board[x_index][i] = "#"
    
    return randomize_rotation(board)


def ensure_offline_nodes(board: list[list[str]]) -> list[list[str]]:
    """Ensure there's at least one obstacle on the board."""
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == "#":
                return board

    board[0][0] = "#"
    return board


def randomize_rotation(board: list[list[str]]) -> list[list[str]]:
    """Randomly rotate the obstacle pattern."""
    return rotate_n_times(board, random.randint(0, 3))

def rotate_n_times(board: list[list[str]], n: int) -> list[list[str]]:
    """Rotate the obstacle pattern N times (90 degrees each)."""
    for _ in range(n % 4):
        board = rotate_board_90_degrees(board)
    return board

def rotate_board_90_degrees(board: list[list[str]]) -> list[list[str]]:
    """Rotate the obstacle pattern 90 degrees clockwise."""
    new_board = [ [ "." for _ in range(len(board)) ] for _ in range(len(board)) ]
    for i in range(len(board)):
        for j in range(len(board)):
            new_board[j][len(board) - i - 1] = board[i][j]
    return new_board
