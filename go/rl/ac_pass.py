import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import encoders
import goboard_fast
from agent import Agent

__all__ = [
    'ACAgent',
    'load_passing_ac_agent',
]


class ACAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1.0

        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = torch.from_numpy(self.encoder.encode(game_state))

        actions, values = self.model(board_tensor[None].float())
        move_probs = actions[0].detach().cpu().numpy()
        estimated_value = values[0].detach().cpu().numpy()[0]
        self.last_state_value = float(estimated_value)

        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = np.power(move_probs, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            true_move = goboard_fast.Move.play(point)
            if not game_state.is_valid_move(true_move):
                true_move = goboard_fast.Move.pass_turn()
            if self.collector is not None:
                self.collector.record_decision(
                    state=board_tensor,
                    action=point_idx,
                    estimated_value=estimated_value
                )
            return true_move
        # No legal, non-self-destructive moves less.
        return goboard_fast.Move.pass_turn()

    def train(self, experience, lr=0.1, batch_size=128, epochs=1):
        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))

        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            policy_target[i][action] = experience.advantages[i]
            value_target[i] = reward

        policy_target = torch.from_numpy(policy_target).float()
        value_target = torch.from_numpy(value_target).float()
        board_tensor = torch.from_numpy(experience.states).float()

        policy_data = DataLoader(TensorDataset(board_tensor, policy_target, value_target), batch_size=batch_size, shuffle=True)

        opt = Adam(params=self.model.parameters(), lr=lr)
        self.model.train()

        for _ in range(epochs):
            for board_tensor, policy_target, value_target in policy_data:
                opt.zero_grad()
                actions, values = self.model(board_tensor)
                policy_loss = F.cross_entropy(actions, policy_target)
                value_loss = F.mse_loss(values, value_target.unsqueeze(1))
                loss = policy_loss + value_loss
                loss.backward()
                opt.step()
        self.model.eval()

