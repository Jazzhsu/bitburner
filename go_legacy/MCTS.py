import copy
import math
import numpy as np

import torch
from tqdm import tqdm

from go import GO, StoneType, opposite

class GOSimulator:
    def __init__(self, go: GO, turn: StoneType, pass_count: int = 0):
        self._go = copy.deepcopy(go)
        self._player = turn 
        self._pass_count = pass_count

        assert pass_count < 2

    def is_terminal(self) -> bool:
        return self._pass_count == 2

    def eval(self) -> dict:
        return self._go.eval()

    def action_to_pos(self, a: int) -> tuple[int, int]:
        if a == -1:
            return (-1, -1)
        return (a // self._go.size(), a % self._go.size())

    def serialize(self) -> str:
        return self._go.serialize(self._player)

    def torch_state(self) -> torch.Tensor:
        return self._go.get_nn_state(self._player)

    def get_action_size(self):
        return self._go.size() ** 2 + 1 # board size + pass

    def is_valid_move(self, action: int):
        if action == self.get_action_size() - 1:
            return True

        return self._go.is_valid_move(*self.action_to_pos(action), self._player)

    def move(self, action: int, check_valid: bool = True):
        assert self._pass_count < 2

        if action == self.get_action_size() - 1:
            self._pass_count += 1
        else:
            valid = self._go.move(*self.action_to_pos(action), self._player, check_valid)
            if valid:
                self._pass_count = 0

        self._player = opposite(self._player)
        # self._go.print_board()

    def player(self) -> StoneType:
        return self._player

class MCTS:
    """
    This class handles the MCTS search.
    It's adapted to use a neural network for policy and value predictions.
    """
    def __init__(self,  nnet, sims, cpuct, device):
        self.nnet = nnet
        self.sim_count = sims
        self.cpuct = cpuct
        self.device = device
        self.Qsa = {}  # Stores Q values for (state, action)
        self.Nsa = {}  # Stores visit counts for (state, action)
        self.Ns = {}   # Stores visit counts for state
        self.Ps = {}   # Stores initial policy (returned by neural net)

    def get_action_probs(self, go: GOSimulator, temp: int = 1, add_noise: bool = False):
        """
        Performs MCTS simulations and returns the resulting policy.
        """
        for _ in range(self.sim_count):
            self.search(copy.deepcopy(go))

        s = go.serialize()
        counts = [self.Nsa.get((s, a), 0) for a in range(go.get_action_size())]

        # Add Dirichlet noise for exploration if requested
        if add_noise:
            alpha = 0.3  # Dirichlet noise parameter
            noise = np.random.dirichlet([alpha] * go.get_action_size())
            # Only add noise to valid moves
            for a in range(go.get_action_size()):
                if go.is_valid_move(a):
                    counts[a] = 0.75 * counts[a] + 0.25 * noise[a] * max(counts)

        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        counts_sum = sum(counts)
        if counts_sum == 0:
            # If no counts, give uniform probability to valid moves
            probs = []
            valid_moves = [a for a in range(go.get_action_size()) if go.is_valid_move(a)]
            for a in range(go.get_action_size()):
                if a in valid_moves:
                    probs.append(1.0 / len(valid_moves))
                else:
                    probs.append(0.0)
            return probs
        
        probs = [x / float(counts_sum) for x in counts]
        return probs

    def search(self, go: GOSimulator, depth: int = 0, max_depth: int = 100):
        """
        Performs one MCTS simulation from the root node.
        Added depth limit to prevent infinite recursion.
        """
        # Prevent infinite recursion
        if depth > max_depth:
            # Return a neutral value for very deep states
            return 0.0
            
        s = go.serialize()

        if go.is_terminal():
            eval = go.eval()
            norm = eval[StoneType.EMPTY] / 2.
            if norm == 0:  # Prevent division by zero
                return 0.0
            return (eval[opposite(go.player())] - norm) / norm

        # --- SELECTION ---
        if s not in self.Ns:
            # --- EXPANSION & VALUE ---
            # Leaf node: get policy and value from the neural network
            try:
                state = go.torch_state()[None].float().to(self.device)
                policy, v = self.nnet(state)
                policy_probs = torch.exp(policy[0]).detach().cpu().numpy()  # Convert log probs to probs
                self.Ps[s] = policy_probs.tolist()
                self.Ns[s] = 1
                return -v.item()
            except Exception as e:
                # If neural network fails, return neutral value
                print(f"Neural network error at depth {depth}: {e}")
                return 0.0

        cur_best = -float('inf')
        best_act = -1
        valid_actions = []

        # Collect all valid actions first
        try:
            for a in range(go.get_action_size()):
                if go.is_valid_move(a):
                    valid_actions.append(a)
        except Exception as e:
            # If is_valid_move fails, force pass
            print(f"Valid move check failed at depth {depth}: {e}")
            valid_actions = [go.get_action_size() - 1]  # Pass action

        # If no valid actions, force pass
        if len(valid_actions) == 0:
            print(f"No valid actions at depth {depth}, forcing pass")
            valid_actions = [go.get_action_size() - 1]  # Pass action

        # Initialize best_act to first valid action
        best_act = valid_actions[0]

        # Select the action with the highest upper confidence bound (UCT)
        for a in valid_actions:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            
            if u > cur_best:
                cur_best = u
                best_act = a
        
        a = best_act
        
        # Store original state for comparison
        original_state = go.serialize()
        
        try:
            go.move(a, check_valid=False)
        except Exception as e:
            print(f"Move failed at depth {depth}: {e}")
            return 0.0
        
        # Check if state actually changed (prevent infinite loops)
        new_state = go.serialize()
        if new_state == original_state and depth > 5:
            print(f"State didn't change at depth {depth}, returning neutral value")
            return 0.0

        v = self.search(go, depth + 1, max_depth)

        # --- BACKPROPAGATION ---
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

