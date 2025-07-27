import random

from agent import Agent
from goboard_fast import GameState, Point, Move
from ai.common import (
    get_surrond_moves,
    capture_move,
    defend_move,
    defend_capture_move,
    growth_move,
    eye_blocking_move,
    eye_creation_move,
    pattern_move,
    expansion_move,
    jump_move,
    corner_move,
    disputed_territory,
    get_disputed_territory_moves,
    get_expansion_move_array,
)

class IlluminatiAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

    def select_move(self, game_state: GameState) -> Move:
        player, board = game_state.situation
        available_moves = disputed_territory(game_state)
        contested_moves = get_disputed_territory_moves(game_state, available_moves)
        move = None

        endgame_available = game_state.last_move is None and len(contested_moves) == 0

        # prioritize captures
        move = capture_move(game_state, player, available_moves)
        if move != None:
            return Move(move)

        # defend captures
        move = defend_capture_move(game_state, player, available_moves)
        if move != None:
            return Move(move)

        # eye creation
        move = None if endgame_available else eye_creation_move(game_state, player, available_moves)
        if move != None:
            return Move(move)

        # surrond (if surronding resulted in <= 1 enemy liberties)
        surrond_move = get_surrond_moves(game_state, player, available_moves)
        if surrond_move != None and surrond_move[2] <= 1:
            return Move(surrond_move[0])

        # eye blocking
        move = None if endgame_available else eye_blocking_move(game_state, player, available_moves)
        if move != None:
            return Move(move)

        move = corner_move(game_state, player, available_moves)
        if move != None:
            return Move(move)

        grow_move = growth_move(game_state, player, available_moves)
        has_move = surrond_move != None or grow_move != None

        pat_move = pattern_move(game_state, player, available_moves)

        rng = random.random()
        use_pattern = rng > 0.25 or not has_move
        move = None if not use_pattern or endgame_available else pat_move
        if move != None:
            return Move(move)

        expansion_moves = get_expansion_move_array(game_state, available_moves)
        move = None if rng < 0.4 else jump_move(game_state, player, available_moves, expansion_moves)
        if move != None:
            return Move(move)

        if rng < 0.6 and surrond_move != None and surrond_move[2] <= 2:
            return Move(surrond_move[0])

        move_options = []
        if surrond_move != None:
            move_options.append(Move(surrond_move[0]))
        
        if grow_move != None:
            move_options.append(Move(grow_move))
        
        exp_move = expansion_move(game_state, player, available_moves, expansion_moves)
        if exp_move != None:
            move_options.append(Move(exp_move))

        if pat_move != None:
            move_options.append(Move(pat_move))
        
        if len(move_options) > 0:
            return random.choice(move_options)
        
        return Move.pass_turn()