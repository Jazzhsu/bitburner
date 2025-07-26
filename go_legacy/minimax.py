
import math
import copy
from go import GO, StoneType, opposite

class MinimaxAgent:
    def __init__(self, go: GO, player: StoneType):
        self._go = go
        self._player = player

    def get_action(self, depth: int) -> tuple[int, int] | None:
        """
        Calculates the best move for the agent using minimax with alpha-beta pruning.
        """
        best_move = None
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for i in range(self._go.size()):
            for j in range(self._go.size()):
                if self._go.is_valid_move(i, j, self._player):
                    next_go = copy.deepcopy(self._go)
                    next_go.move(i, j, self._player)
                    
                    score = self._minimax(next_go, depth - 1, alpha, beta, False, opposite(self._player))
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
                    
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break  # Beta cutoff

        return best_move

    def _minimax(self, go: GO, depth: int, alpha: float, beta: float, is_maximizing: bool, player: StoneType) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        """
        if depth == 0:
            return self._evaluate_state(go)

        if is_maximizing:
            best_score = -math.inf
            for i in range(go.size()):
                for j in range(go.size()):
                    if go.is_valid_move(i, j, player):
                        next_go = copy.deepcopy(go)
                        next_go.move(i, j, player)
                        
                        score = self._minimax(next_go, depth - 1, alpha, beta, False, opposite(player))
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        
                        if beta <= alpha:
                            return best_score
            return best_score
        else: # Minimizing
            best_score = math.inf
            for i in range(go.size()):
                for j in range(go.size()):
                    if go.is_valid_move(i, j, player):
                        next_go = copy.deepcopy(go)
                        next_go.move(i, j, player)

                        score = self._minimax(next_go, depth - 1, alpha, beta, True, opposite(player))
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        
                        if beta <= alpha:
                            return best_score
            return best_score

    def _get_dynamic_depth(self) -> int:
        """
        Adjusts search depth based on the number of empty intersections.
        """
        empty_intersections = 0
        for i in range(self._go.size()):
            for j in range(self._go.size()):
                if self._go._board[i][j] == StoneType.EMPTY:
                    empty_intersections += 1
        
        if empty_intersections > self._go.size() ** 2 * 0.7:
            return 4  # Early game
        elif empty_intersections > self._go.size() ** 2 * 0.3:
            return 5  # Mid game
        else:
            return 6  # Late game

    def _evaluate_state(self, go: GO) -> float:
        """
        Evaluates the board state from the perspective of the agent's player.
        The score is a combination of territory, liberties, group sizes, and captures.
        """
        eval_scores = go.eval()
        my_score = eval_scores[self._player]
        opponent_score = eval_scores[opposite(self._player)]
        total_score = eval_scores[StoneType.EMPTY]

        if total_score == 0:
            return 0.0

        score_ratio = (my_score - opponent_score) / total_score

        my_liberties, opponent_liberties = self._count_liberties(go)
        my_groups, opponent_groups = self._count_groups(go)

        # Combine metrics with weights
        liberty_weight = 0.2
        group_weight = 0.1

        final_score = (
            score_ratio +
            liberty_weight * (my_liberties - opponent_liberties) / (self._go.size() ** 2) +
            group_weight * (my_groups - opponent_groups) / (self._go.size() ** 2)
        )

        return final_score

    def _count_liberties(self, go: GO) -> tuple[int, int]:
        my_liberties = 0
        opponent_liberties = 0
        
        for root, libs in go._lib_map.items():
            idx = next(iter(go._group_map[root]))
            i, j = go._idx2ij(idx)
            stone_type = go._board[i][j]
            
            if stone_type == self._player:
                my_liberties += len(libs)
            else:
                opponent_liberties += len(libs)
                
        return my_liberties, opponent_liberties

    def _count_groups(self, go: GO) -> tuple[int, int]:
        my_groups = 0
        opponent_groups = 0
        
        for root, group in go._group_map.items():
            idx = next(iter(group))
            i, j = go._idx2ij(idx)
            stone_type = go._board[i][j]
            
            if stone_type == self._player:
                my_groups += 1
            else:
                opponent_groups += 1
                
        return my_groups, opponent_groups 