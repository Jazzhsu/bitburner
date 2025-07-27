from typing import Set

from goboard_fast import Board, Point, Player


def effective_liberties_of_new_move(board: Board, move: Point, player: Player) -> Set[Point]:
  liberties = set()
  for neighbor in board.neighbors(move):
    if board._grid.get(neighbor) is None:
      liberties.add(neighbor)

    elif board._grid.get(neighbor).color == player:
      liberties.update(board._grid.get(neighbor).liberties)
  
  if move in liberties:
    liberties.remove(move)
  return liberties