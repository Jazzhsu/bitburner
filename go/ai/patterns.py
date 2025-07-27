# Inspired by https://github.com/pasky/michi/blob/master/michi.py
from functools import lru_cache
import random
from typing import List, Optional
from goboard_fast import Board, Point, Player
from enum import Enum
from ai.utils import effective_liberties_of_new_move

class PointState(Enum):
  EMPTY = 0
  WHITE = 1
  BLACK = 2
  OFF_BOARD = 3

# 3x3 piece patterns; X,O are color pieces; x,o are any state except the opposite color piece;
# " " is off the edge of the board; "?" is any state (even off the board)
THREE_BY_THREE_PATTERNS = [
    [
        "XOX",  # hane pattern - enclosing hane
        "...",
        "???",
    ],
    [
        "XO.",  # hane pattern - non-cutting hane
        "...",
        "?.?",
    ],
    [
        "XO?",  # hane pattern - magari
        "X..",
        "o.?",
    ],
    [
        ".O.",  # generic pattern - katatsuke or diagonal attachment; similar to magari
        "X..",
        "...",
    ],
    [
        "XO?",  # cut1 pattern (kiri) - unprotected cut
        "O.x",
        "?x?",
    ],
    [
        "XO?",  # cut1 pattern (kiri) - peeped cut
        "O.X",
        "???",
    ],
    [
        "?X?",  # cut2 pattern (de)
        "O.O",
        "xxx",
    ],
    [
        "OX?",  # cut keima
        "x.O",
        "???",
    ],
    [
        "X.?",  # side pattern - chase
        "O.?",
        "   ",
    ],
    [
        "OX?",  # side pattern - block side cut
        "X.O",
        "   ",
    ],
    [
        "?X?",  # side pattern - block side connection
        "o.O",
        "   ",
    ],
    [
        "?XO",  # side pattern - sagari
        "o.o",
        "   ",
    ],
    [
        "?OX",  # side pattern - cut
        "X.O",
        "   ",
    ],
]


def get_neighborhood(board: Board, point: Point) -> List[List[PointState]]:
    """
    Gets the 3x3 neighborhood around the given point.
    
    Args:
        board: The Go board
        point: Center point to get neighborhood for
        
    Returns:
        3x3 grid of PointStates
    """
    neighborhood = []
    
    for row_offset in [-1, 0, 1]:
        row = []
        for col_offset in [-1, 0, 1]:
            neighbor_point = Point(point.row + row_offset, point.col + col_offset)
            
            if not board.is_on_grid(neighbor_point):
              row.append(PointState.OFF_BOARD)
            elif board._grid.get(neighbor_point) is None:
              row.append(PointState.EMPTY)
            elif board._grid.get(neighbor_point).color == Player.black:
              row.append(PointState.BLACK)
            else:
              row.append(PointState.WHITE)
                
        neighborhood.append(row)
    
    return neighborhood


def matches_pattern_symbol(symbol: str, point_state: PointState, current_player: PointState) -> bool:
    """
    Check if a point matches the given pattern symbol.
    
    Args:
        symbol: Pattern symbol ('X', 'O', 'x', 'o', '.', ' ', '?')
        point_state: PointState at the point
        current_player: The current player we're matching for
        
    Returns:
        True if the point matches the symbol
        
    Pattern symbols:
    - 'X': Must be current player's stone
    - 'O': Must be opponent's stone  
    - 'x': Must NOT be opponent's stone (current player or empty or off-board)
    - 'o': Must NOT be current player's stone (opponent or empty or off-board)
    - '.': Must be empty
    - ' ': Must be off the board
    - '?': Matches anything
    """
    opponent = PointState.BLACK if current_player == PointState.WHITE else PointState.WHITE
    
    if symbol == 'X':
        return point_state == current_player
    elif symbol == 'O':
        return point_state == opponent
    elif symbol == 'x':
        return point_state != opponent
    elif symbol == 'o':
        return point_state != current_player
    elif symbol == '.':
        return point_state == PointState.EMPTY
    elif symbol == ' ':
        return point_state == PointState.OFF_BOARD
    elif symbol == '?':
        return True
    else:
        return False


def check_pattern_match(neighborhood: List[List[PointState]], pattern: List[str], player: PointState) -> bool:
    """
    Check if a 3x3 neighborhood matches a pattern.
    
    Args:
        neighborhood: 3x3 grid of players
        pattern: 3x3 pattern strings
        player: Current player
        
    Returns:
        True if the neighborhood matches the pattern
    """
    for row in range(3):
        for col in range(3):
            pattern_symbol = pattern[row][col]
            state_at_point = neighborhood[row][col]
            
            if not matches_pattern_symbol(pattern_symbol, state_at_point, player):
                return False
                
    return True


def rotate_90_degrees(pattern: List[str]) -> List[str]:
    """Rotate a 3x3 pattern 90 degrees clockwise."""
    return [
        f"{pattern[2][0]}{pattern[1][0]}{pattern[0][0]}",
        f"{pattern[2][1]}{pattern[1][1]}{pattern[0][1]}",
        f"{pattern[2][2]}{pattern[1][2]}{pattern[0][2]}",
    ]


def vertical_mirror(pattern: List[str]) -> List[str]:
    """Mirror a pattern vertically."""
    return [pattern[2], pattern[1], pattern[0]]


def horizontal_mirror(pattern: List[str]) -> List[str]:
    """Mirror a pattern horizontally."""
    return [
        pattern[0][::-1],  # Reverse string
        pattern[1][::-1],
        pattern[2][::-1],
    ]

@lru_cache(maxsize=1)
def expand_all_patterns() -> List[List[str]]:
    """
    Generate all rotations and reflections of the base patterns.
    This should only be calculated once.
    
    Returns:
        List of all pattern variations
    """
    patterns = THREE_BY_THREE_PATTERNS.copy()
    
    # Add 90, 180, 270 degree rotations
    rotated_once = [rotate_90_degrees(p) for p in THREE_BY_THREE_PATTERNS]
    rotated_twice = [rotate_90_degrees(p) for p in rotated_once]
    rotated_thrice = [rotate_90_degrees(p) for p in rotated_twice]
    
    patterns.extend(rotated_once)
    patterns.extend(rotated_twice)
    patterns.extend(rotated_thrice)
    
    # Add vertical mirrors of all rotations
    mirrored_patterns = patterns.copy()
    mirrored_patterns.extend([vertical_mirror(p) for p in patterns])
    
    # Add horizontal mirrors of all previous patterns
    all_patterns = mirrored_patterns.copy()
    all_patterns.extend([horizontal_mirror(p) for p in mirrored_patterns])
    
    return all_patterns


def find_pattern_move(board: Board, player: Player, legal_moves: List[Point]) -> Optional[Point]:
    """
    Find a move that matches any of the 3x3 patterns.
    
    Args:
        board: The Go board
        player: Current player
        legal_moves: List of legal move points
        
    Returns:
        A point that matches a pattern, or None if no matches found
    """
        
    # This should only be calculated once.
    expanded_patterns = expand_all_patterns()

    matching_moves = []
    
    for move_point in legal_moves:
        neighborhood = get_neighborhood(board, move_point)
        
        # Check if this point matches any pattern
        for pattern in expanded_patterns:
            if not check_pattern_match(neighborhood, pattern, PointState.BLACK if player == Player.black else PointState.WHITE):
                continue
            
            liberties = effective_liberties_of_new_move(board, move_point, player)
            if len(liberties) > 1:
                matching_moves.append(move_point)
    
    if len(matching_moves) == 0:
        return None
    
    return random.choice(matching_moves)