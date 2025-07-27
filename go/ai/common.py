import random
from collections import defaultdict
from typing import Set
import copy

from goboard_fast import Board, Point, Player, GameState, GoString, Move
from agent.helpers import is_point_an_eye
from ai.patterns import find_pattern_move
from ai.utils import effective_liberties_of_new_move

def capture_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move, enemy_old_liberties, enemy_new_liberties """
  surrond_move = get_surrond_moves(game_state, player, available_moves)
  if surrond_move is None:
    return None

  move, _, enemy_new_liberties = surrond_move
  return move if enemy_new_liberties == 0 else None

def defend_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move """
  defend_move = get_defend_move(game_state, player, available_moves)
  return defend_move[0] if defend_move != None else None

def defend_capture_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move """
  defend_move = get_defend_move(game_state, player, available_moves)
  if defend_move == None:
    return None

  move, player_old_liberties, player_new_liberties = defend_move
  return move if player_old_liberties == 1 and player_new_liberties > 1 else None

def get_defend_move(game_state: GameState, player: Player, available_moves: list[Point]) -> tuple[Point, int, int] | None:
  """ return the best move, player_old_liberties, player_new_liberties """
  liberty_growth_moves = get_liberty_growth_moves(game_state, player, available_moves)

  # Moves that prevent the enemy from capturing and maximize liberty growth.
  potential_moves = set()
  max_liberty_growth = 0
  for move, old_liberties, new_liberties in liberty_growth_moves:
    if new_liberties > old_liberties and old_liberties <= 1:
      growth = new_liberties - old_liberties

      if growth == max_liberty_growth:
        potential_moves.add((move, old_liberties, new_liberties))
      elif growth > max_liberty_growth:
        max_liberty_growth = growth
        potential_moves = set([(move, old_liberties, new_liberties)])
  
  if len(potential_moves) == 0:
    return None

  return random.choice(list(potential_moves))
  
def growth_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move, player_old_liberties, player_new_liberties """
  liberty_growth_moves = get_liberty_growth_moves(game_state, player, available_moves)
  if len(liberty_growth_moves) == 0:
    return None
  
  max_growth = max(x[2] - x[1] for x in liberty_growth_moves)
  filtered_moves = [x for x in liberty_growth_moves if x[2] - x[1] == max_growth]
  return random.choice(filtered_moves)[0]
  

def eye_creation_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move, player_old_liberties, player_new_liberties """
  eye_creation_moves = get_eye_creation_moves(game_state, player, available_moves)
  if len(eye_creation_moves) == 0:
    return None
  
  return eye_creation_moves[0][0]

def eye_blocking_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move, player_old_liberties, player_new_liberties """
  """ If there is only one move that would create two eyes for the opponent, it should be blocked if possible """

  opponent = Player.black if player == Player.white else Player.white
  opponent_eye_moves = get_eye_creation_moves(game_state, opponent, available_moves, 5)
  two_eye_moves = [(move, creates_life) for move, creates_life in opponent_eye_moves if creates_life]
  one_eye_moves = [(move, creates_life) for move, creates_life in opponent_eye_moves if not creates_life]
  
  if len(two_eye_moves) == 1:
    return two_eye_moves[0][0]
  
  if len(two_eye_moves) == 0 and len(one_eye_moves) == 1:
    return one_eye_moves[0][0]
  
  return None
    
def pattern_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move, player_old_liberties, player_new_liberties """
  return find_pattern_move(game_state.board, player, available_moves)
    
def expansion_move(
  game_state: GameState,
  player: Player,
  available_moves: list[Point],
  expansion_moves: list[Point],
) -> Point | None:
  """ return the best move """
  if len(expansion_moves) > 0:
    return random.choice(expansion_moves)
  
  return None
  
def jump_move(
  game_state: GameState,
  player: Player,
  available_moves: list[Point],
  expansion_moves: list[Point],
) -> Point | None:
  """ return the best move """
  if len(expansion_moves) == 0:
    return None

  jump_moves = set()
  for move in expansion_moves:
    for (j_row, j_col) in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
      jump_neighbor = game_state.board._grid.get(Point(move.row + j_row, move.col + j_col))
      if jump_neighbor != None and jump_neighbor.color == player:
        jump_moves.add(move)

  return random.choice(list(jump_moves)) if len(jump_moves) > 0 else None

def corner_move(game_state: GameState, player: Player, available_moves: list[Point]) -> Point | None:
  """ return the best move """
  _, board = game_state.situation

  # (edge, placement), (top-left corner of 3x3 corner square, where to place the stone)
  edges = [
    (Point(1, 1), Point(3, 3)),
    (Point(1, board.num_rows - 2), Point(3, board.num_rows - 2)),
    (Point(board.num_rows - 2, 1), Point(board.num_rows - 2, 3)),
    (Point(board.num_rows - 2, board.num_cols - 2), Point(board.num_rows - 2, board.num_cols - 2)),
  ]

  for edge, placement in edges:
    count_non_wall = 0
    count_piece = 0
    for d_row in range(3):
      for d_col in range(3):
        p = Point(edge.row + d_row, edge.col + d_col)
        if board.is_wall(p):
          continue
        
        count_non_wall += 1
        if board._grid.get(p) is not None:
          count_piece += 1

    if count_non_wall >= 7 and count_piece == 0:
      return placement

  return None


def get_surrond_moves(game_state: GameState, player: Player, available_moves: list[Point]) -> tuple[Point, int, int] | None:
  """ return the best move, enemy_old_liberties, enemy_new_liberties """
  _, board = game_state.situation

  opponent = Player.black if player == Player.white else Player.white
  enemy_chain = [chain for _, chain in board._grid.items() if chain.color == opponent]

  if len(enemy_chain) == 0 or len(available_moves) == 0:
      return None

  potential_moves = set()
  for chain in enemy_chain:
    for liberty in chain.liberties:
      if liberty in available_moves:
        potential_moves.add(liberty)
  
  # (move, enemy_old_liberties, enemy_new_liberties)
  capture_move = set()
  atari_move = set()
  surrond_move = set()

  for move in potential_moves:
    # Get the liberties of the stone if it were to be played.
    new_liberties = effective_liberties_of_new_move(board, move, player)

    weakest_enemy_chain = weakest_adjacent_enemy_chain(board, move, player)

    enemy_chain_length = len(weakest_enemy_chain.stones)
    enemy_chain_liberties = len(weakest_enemy_chain.liberties)

    # Is enemy chain only have one liberty group
    # Group `enemy_chain_liberties` into groups if they are connected.
    group_sz = group_size(board, move, weakest_enemy_chain.liberties, set())

    # Do not suggest moves that do not capture anything and let your opponent immediately capture
    if len(new_liberties) <= 2 and enemy_chain_liberties > 2:
      continue

    # If the enemy chain liberties is 1, it is a capture move.
    if enemy_chain_liberties <= 1:
      capture_move.add((move, enemy_chain_liberties, enemy_chain_liberties - 1))

    # If the enemy chain liberties is 2, it is an atari move.
    # We should not put ourselves in atari (liberty >= 2) unless we can capture the enemy chain.
    elif enemy_chain_length == 2 and (len(new_liberties) >= 2 or (group_sz == 1 and enemy_chain_liberties > 3)):
      atari_move.add((move, enemy_chain_liberties, enemy_chain_liberties - 1))

    elif len(new_liberties) >= 2:
      surrond_move.add((move, enemy_chain_liberties, enemy_chain_liberties - 1))

  
  if len(capture_move) > 0:
    return list(capture_move)[0]

  if len(atari_move) > 0:
    return list(atari_move)[0]

  if len(surrond_move) > 0:
    return list(surrond_move)[0]

  return None

def get_expansion_move_array(game_state: GameState, available_moves: list[Point]) -> list[Point]:
  """ return the best move, player_old_liberties, player_new_liberties """
  _, board = game_state.situation
  empty_spaces = []
  for space in available_moves:
    empty_count = 0
    for neighbor in board.neighbors(space):
      if board._grid.get(neighbor) is None:
        empty_count += 1
    
    if empty_count == 4:
      empty_spaces.append(space)

  if len(empty_spaces) > 0:
    return empty_spaces

  return disputed_territory(game_state)

def get_eye_creation_moves(
  game_state: GameState,
  player: Player,
  available_moves: list[Point],
  max_liberties: int = 99,
) -> list[tuple[Point, int, int]]:
  """ return the best move, player_old_liberties, player_new_liberties """
  player, board = game_state.situation
  all_eyes = get_all_eyes(board, player)

  # Living groups are groups that have at least 2 eyes.
  living_groups = set([chain for chain, eyes in all_eyes.items() if len(eyes) >= 2])
  living_groups_count = len(living_groups)
  eye_count = sum(len(eyes) for eyes in all_eyes.values())

  point_to_analyze = set()
  for _, chain in board._grid.items():
    if chain.color != player:
      continue

    if len(chain.stones) == 1:
      continue

    if len(chain.liberties) > max_liberties:
      continue

    if chain in living_groups:
      continue

    for point in chain.liberties:
      if point not in available_moves:
        continue

      wall_count = board.get_wall_count(point)
      friend_count = 0
      empty_count = 0
      for neighbor in board.neighbors(point):
        if board._grid.get(neighbor) is None:
          empty_count += 1
        elif board._grid.get(neighbor).color == player:
          friend_count += 1

      if friend_count + wall_count >= 2 and empty_count >= 1:
        point_to_analyze.add(point)

  # tuple[Point, bool], true if the move creates a living group, false if it does not.
  moves = []

  # Evaluate the moves.
  for point in point_to_analyze:
    # Copy the board and place the stone for evaluation.
    evaluation_board = copy.deepcopy(board)
    evaluation_board.place_stone(player, point)

    # Get the new eyes and living groups.
    new_eyes = get_all_eyes(evaluation_board, player)
    new_living_groups = set([chain for chain, eyes in new_eyes.items() if len(eyes) >= 2])
    new_living_groups_count = len(new_living_groups)
    new_eye_count = sum(len(eyes) for eyes in new_eyes.values())
    
    if (new_living_groups_count > living_groups_count or
        (new_living_groups_count == living_groups_count and new_eye_count > eye_count)):
      moves.append((point, new_living_groups_count > living_groups_count))

  # Sort by if the move creates a living group.
  return sorted(moves, key=lambda x: x[1], reverse=True)

def get_liberty_growth_moves(
  game_state: GameState,
  player: Player,
  available_moves: list[Point],
) -> list[tuple[Point, int, int]]:
  """ return the best move, player_old_liberties, player_new_liberties """
  player, board = game_state.situation

  potential_moves = set()
  for _, chain in board._grid.items():
    if chain.color != player:
      continue

    for liberty in chain.liberties:
      if liberty in available_moves:
        potential_moves.add(liberty)

  if len(potential_moves) == 0:
    return []

  liberty_growth_moves = set()
  for move in potential_moves:
    new_liberties = len(effective_liberties_of_new_move(board, move, player))
    old_liberties = 0x7FFFFFFF
    for neighbor in board.neighbors(move):
      neighbor_chain = board._grid.get(neighbor)
      if neighbor_chain != None and neighbor_chain.color == player:
        old_liberties = min(old_liberties, len(neighbor_chain.liberties))

    if new_liberties > 1 and new_liberties >= old_liberties:
      liberty_growth_moves.add((move, old_liberties, new_liberties))

  # Sort by liberty growth.
  return liberty_growth_moves

def get_disputed_territory_moves(
  game_state: GameState,
  available_moves: list[Point],
) -> list[Point]:
  """ return list of moves that are disputed territory """
  _, board = game_state.situation
  filtered_moves = set()
  visited = set()

  for move in available_moves:
    if move in visited:
      continue

    types = set()
    flood_fill_find_neighbors(board, move, visited, types)

    if len(types) == 2:
      filtered_moves.add(move)
  
  return list(filtered_moves)

def flood_fill_find_neighbors(board: Board, point: Point, visited: Set[Point], types: Set[Player]):
  if point in visited:
    return

  if board._grid.get(point) != None:
    types.add(board._grid.get(point).color)
    return

  visited.add(point)

  for neighbor in board.neighbors(point):
    flood_fill_find_neighbors(board, neighbor, visited, types)


def group_size(board: Board, point: Point, liberties: Set[Point], visited: Set[Point]) -> int:
  if point in visited:
    return 0

  if point not in liberties:
    return 0

  visited.add(point)
  size = 1
  for neighbor in board.neighbors(point):
    if board._grid.get(neighbor) is None:
      size += group_size(board, neighbor, liberties, visited)

  return size

  
def weakest_adjacent_enemy_chain(board: Board, move: Point, player: Player) -> GoString:
  """ return the weakest adjacent enemy chain (chain.color != player) """
  weakest_chain = None
  for neighbor in board.neighbors(move):
    if board._grid.get(neighbor) is None:
      continue
    
    if board._grid.get(neighbor).color == player:
      continue
      
    chain = board._grid.get(neighbor)
    if weakest_chain is None or len(chain.stones) < len(weakest_chain.stones):
      weakest_chain = chain
  
  return weakest_chain


def disputed_territory(game_state: GameState) -> list[Point]:
  valid_moves = [ move.point for move in game_state.legal_moves(include_pass=False) if move.is_play ]
  player, board = game_state.situation

  # Do not play into two eyes.
  eyes = get_all_eyes(board, player)
  for _, eye_chains in eyes.items():
    if len(eye_chains) == 1:
      continue

    for eye_chain in eye_chains:
      for point in eye_chain.stones:
        if point in valid_moves:
          valid_moves.remove(point)

  opponent = Player.black if player == Player.white else Player.white
  opponent_eyes = get_all_potential_eyes(board, opponent)

  for empty_chain, neighboring_chains in opponent_eyes:
    attackable_points = set()

    for neighbor in neighboring_chains:
      liberties = neighbor.liberties

      # If the neighbor has more than 4 liberties, it is not worth attacking.
      if len(liberties) > 4:
        continue
      
      neighbor_s_neighbor = get_all_neighboring_chains(board, neighbor)
      # If the neighbor has no friendly neighbors, it is not worth attacking.
      if not any(n.color == player for n in neighbor_s_neighbor):
        continue

      # Filter out liberties that are not in the empty chain.
      liberty_to_analyze = [liberty for liberty in liberties if liberty in empty_chain.stones]

      # If the chain has liberties outside of the empty space that is being analyzed, it is not fully encircled.
      # and should not be attacked yet.
      if len(liberty_to_analyze) != len(liberties):
        continue

      attackable_points.update(liberties)

    for point in empty_chain.stones:
      if point not in attackable_points and point in valid_moves:
        valid_moves.remove(point)
    
  return list(valid_moves)





      


      



def get_all_eyes(board: Board, player: Player) -> dict[GoString, list[GoString]]:
  """ Returns a map of eyes for the given player. Key is the player chain,
      value is a list of empty chains that are encircled by the player chain. """
  eye_candidates = get_all_potential_eyes(board, player)
  eyes: dict[GoString, list[GoString]] = defaultdict(list)

  for candidate in eye_candidates:
    chain, neighboring_chains = candidate
    if len(neighboring_chains) == 0:
      continue

    if len(neighboring_chains) == 1:
      eyes[neighboring_chains[0]].append(chain)
      continue

    encircling_chain = find_neighboring_chains_that_fully_encircle_empty_space(board, chain, neighboring_chains)
    if encircling_chain is not None:
      eyes[encircling_chain].append(chain)
      continue

  return eyes

def get_all_potential_eyes(board: Board, player: Player, max_size: int = 11) -> list[tuple[GoString, list[GoString]]]:
  """ Returns a list of tuples, each containing an empty chain and a list of neighboring chains. """
  all_chains = get_all_chains(board)
  max_size = min(board.available_points() * 0.4, max_size)
  empty_chains = [chain for chain in all_chains if chain.color == None]

  eye_candidates = []
  for chain in empty_chains:
    if len(chain.stones) > max_size:
      continue

    neighboring_chains = get_all_neighboring_chains(board, chain)
    has_white_neighbor = any(neighbor.color == Player.white for neighbor in neighboring_chains)
    has_black_neighbor = any(neighbor.color == Player.black for neighbor in neighboring_chains)

    if (has_white_neighbor and not has_black_neighbor and player == Player.white) or \
       (not has_white_neighbor and has_black_neighbor and player == Player.black):
      eye_candidates.append((chain, neighboring_chains))

  return eye_candidates

def get_all_chains(board: Board) -> list[GoString]:
  chains = set()
  visited = set()
  for r in range(1, board.num_rows + 1):
    for c in range(1, board.num_cols + 1):
      point = Point(r, c)
      chain = board._grid.get(point)

      if chain is not None:
        chains.add(chain)
        continue

      if point in visited:
        continue
      
      stones = []
      flood_fill(board, point, stones, visited)
      chains.add(GoString(None, stones, []))

  return chains


def flood_fill(board: Board, point: Point, stones: list[Point], visited: set[Point]):
  if point in visited:
    return

  visited.add(point)
  stones.append(point)

  for neighbor in board.neighbors(point):
    if board._grid.get(neighbor) == None:
      flood_fill(board, neighbor, stones, visited)


def find_furthest_points_of_chain(chain: GoString) -> dict:
  """
  Determine the furthest that a chain extends in each of the cardinal directions.
  
  Args:
    chain: GoString containing the stones of a chain
    
  Returns:
    Dictionary with 'north', 'south', 'east', 'west' extents
  """
  if not chain.stones:
    return {'north': 0, 'south': 0, 'east': 0, 'west': 0}
  
  # Convert stones to list to get first stone
  stones_list: list[Point] = list(chain.stones)
  first_stone = stones_list[0]
  
  directions = {
    'north': first_stone.row,   # highest row number
    'south': first_stone.row,   # lowest row number  
    'east': first_stone.col,    # highest col number
    'west': first_stone.col,    # lowest col number
  }
  
  for stone in chain.stones:
    if stone.row > directions['north']:
      directions['north'] = stone.row
    if stone.row < directions['south']:
      directions['south'] = stone.row
    if stone.col > directions['east']:
      directions['east'] = stone.col
    if stone.col < directions['west']:
      directions['west'] = stone.col
      
  return directions

def get_player_neighbors(board: Board, chain: GoString) -> list[GoString]:
  """Get all neighbors of a chain that are not None."""
  neighbors = set()
  for stone in chain.stones:
    for neighbor in board.neighbors(stone):
      if board._grid.get(neighbor) != None:
        neighbors.add(board._grid.get(neighbor))
  return list(neighbors)

def get_all_neighboring_chains(board: Board, target_chain: GoString) -> list[GoString]:
  """Get all chains that neighbor the target chain."""
  return list(get_player_neighbors(board, target_chain))


def points_to_ignore(chain_list: list[GoString], index: int) -> set[Point]:
  """Remove a chain at the specified index and return the remaining chains."""
  points_to_ignore = set()
  for i, chain in enumerate(chain_list):
    if i == index:
      continue

    for stone in chain.stones:
      points_to_ignore.add(stone)

  return points_to_ignore


def find_neighboring_chains_that_fully_encircle_empty_space(
    board: Board,
    candidate_chain: GoString,
    neighbor_chain_list: list[GoString],
) -> GoString | None:
  """
  Find neighboring chains that fully encircle empty space.
  
  Args:
    board: The Go board
    candidate_chain: The chain we're checking for encirclement
    neighbor_chain_list: List of neighboring chains to evaluate
    all_chains: List of all chains on the board
    
  Returns:
    List of chains that fully encircle the candidate chain
  """
  board_max = board.num_rows  # Assuming square board
  candidate_spread = find_furthest_points_of_chain(candidate_chain)
  
  encircling_chains = []
  
  for index, neighbor_chain in enumerate(neighbor_chain_list):
    # If the chain does not go far enough to surround the eye in question, skip it
    neighbor_spread = find_furthest_points_of_chain(neighbor_chain)
    
    could_wrap_north = (
      neighbor_spread['north'] > candidate_spread['north'] or
      (candidate_spread['north'] == board_max and neighbor_spread['north'] == board_max)
    )
    could_wrap_east = (
      neighbor_spread['east'] > candidate_spread['east'] or
      (candidate_spread['east'] == board_max and neighbor_spread['east'] == board_max)
    )
    could_wrap_south = (
      neighbor_spread['south'] < candidate_spread['south'] or
      (candidate_spread['south'] == 1 and neighbor_spread['south'] == 1)
    )
    could_wrap_west = (
      neighbor_spread['west'] < candidate_spread['west'] or
      (candidate_spread['west'] == 1 and neighbor_spread['west'] == 1)
    )
    
    if not (could_wrap_north and could_wrap_east and could_wrap_south and could_wrap_west):
      continue
    
    # Get all points from other neighbor chains (excluding current one)
    points_ignore = points_to_ignore(neighbor_chain_list, index)

    # If only one neighboring chain remains, this chain fully encircles
    if is_in_chain(board, list(candidate_chain.stones)[0], points_ignore, neighbor_chain, set()):
      encircling_chains.append(neighbor_chain)
      
  # TODO: figure out why there are multiple encircling chains
  return encircling_chains[0] if len(encircling_chains) >= 1 else None


def is_in_chain(board: Board, point: Point, point_to_ignore: set[Point], chain: GoString, visited: set[Point]) -> bool:
  if point in visited:
    return True
  
  visited.add(point)

  stone_type = board._grid.get(point)
  empty = stone_type is None or point in point_to_ignore

  if not empty:
    return point in chain.stones

  in_chain = True
  for neighbor in board.neighbors(point):
    in_chain = in_chain and is_in_chain(board, neighbor, point_to_ignore, chain, visited)

  return in_chain
