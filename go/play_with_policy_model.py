#!/usr/bin/env python3
"""
Script to play Go using a trained PolicyModel.

This script demonstrates:
1. Loading a trained PolicyModel
2. Using it to play against a human or other agents
3. Handling move selection with temperature and legal move masking
"""

import torch
import argparse
from typing import Optional

from models.policy_model import PolicyModel, create_medium_policy_model
from goboard_fast import GameState, Player, Move, Point
from ai.illuminati_agent import IlluminatiAgent


class PolicyAgent:
    """Agent that uses a trained PolicyModel to select moves."""
    
    def __init__(self, model_path: str, board_size: int, temperature: float = 1.0):
        """
        Initialize the PolicyAgent.
        
        Args:
            model_path: Path to the trained model file
            board_size: Size of the Go board
            temperature: Temperature for move selection (lower = more deterministic)
        """
        self.model = create_medium_policy_model(board_size)
        self.board_size = board_size
        self.temperature = temperature
        
        # Load trained weights
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Loaded PolicyModel from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            print("Using untrained model (random play)")
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using the PolicyModel."""
        current_player = game_state.next_player
        
        # Get move probabilities
        action_probs, best_action = self.model.predict_move(
            game_state, current_player, temperature=self.temperature
        )
        
        # Mask illegal moves
        masked_probs = self.model.mask_illegal_moves(action_probs, game_state)
        
        # Sample from the distribution (or take the best move)
        if self.temperature > 0:
            # Sample from the probability distribution
            action_idx = torch.multinomial(masked_probs, 1).item()
        else:
            # Take the best move (greedy)
            action_idx = torch.argmax(masked_probs).item()
        
        # Convert to move
        if action_idx >= self.model.num_positions:
            return Move.pass_turn()
        else:
            point = self.model.action_to_point(action_idx)
            return Move.play(point)
    
    def get_move_probabilities(self, game_state: GameState):
        """Get the full move probability distribution."""
        current_player = game_state.next_player
        action_probs, _ = self.model.predict_move(game_state, current_player, self.temperature)
        masked_probs = self.model.mask_illegal_moves(action_probs, game_state)
        return masked_probs


def print_board(game_state: GameState):
    """Print the current board state."""
    board = game_state.board
    
    # Header with column numbers
    header = "  "
    for col in range(1, board.num_cols + 1):
        header += f"{col:2}"
    print(header)
    
    # Board rows
    for row in range(1, board.num_rows + 1):
        line = f"{row:2}"
        for col in range(1, board.num_cols + 1):
            point = Point(row, col)
            stone = board.get(point)
            
            if board.is_wall(point):
                line += " #"
            elif stone == Player.black:
                line += " ●"
            elif stone == Player.white:
                line += " ○"
            else:
                line += " ·"
        line += f" {row}"
        print(line)
    
    # Footer with column numbers
    footer = "  "
    for col in range(1, board.num_cols + 1):
        footer += f"{col:2}"
    print(footer)
    
    # Next player info
    next_player = 'Black ●' if game_state.next_player == Player.black else 'White ○'
    print(f"Next player: {next_player}")


def get_human_move(game_state: GameState) -> Move:
    """Get a move from human input."""
    while True:
        try:
            move_input = input("\nEnter your move (e.g., '3,4' or 'pass'): ").strip().lower()
            
            if move_input in ['pass', 'p']:
                return Move.pass_turn()
            elif move_input in ['resign', 'r']:
                return Move.resign()
            else:
                # Parse coordinates
                parts = move_input.replace(' ', '').split(',')
                if len(parts) == 2:
                    row, col = int(parts[0]), int(parts[1])
                    point = Point(row, col)
                    move = Move.play(point)
                    
                    if game_state.is_valid_move(move):
                        return move
                    else:
                        print("❌ Invalid move! Try again.")
                else:
                    print("❌ Invalid format! Use 'row,col' (e.g., '3,4') or 'pass'")
        
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input! Use 'row,col' (e.g., '3,4') or 'pass'")
        except EOFError:
            return Move.resign()


def show_top_moves(agent: PolicyAgent, game_state: GameState, top_k: int = 5):
    """Show the top K moves suggested by the PolicyModel."""
    probs = agent.get_move_probabilities(game_state)
    
    # Get top moves
    sorted_indices = torch.argsort(probs, descending=True)
    
    print(f"\n🤖 Top {top_k} suggested moves:")
    for i in range(min(top_k, len(sorted_indices))):
        action_idx = sorted_indices[i].item()
        prob = probs[action_idx].item()
        
        if action_idx >= agent.model.num_positions:
            move_str = "Pass"
        else:
            point = agent.model.action_to_point(action_idx)
            move_str = f"{point.row},{point.col}"
        
        print(f"  {i+1}. {move_str} ({prob:.1%})")


def play_human_vs_policy(model_path: str, board_size: int, human_color: Player, temperature: float):
    """Play a game between human and PolicyModel."""
    print(f"🎮 Starting game: Human ({human_color}) vs PolicyModel")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Model temperature: {temperature}")
    print()
    
    # Create agents
    policy_agent = PolicyAgent(model_path, board_size, temperature)
    
    # Initialize game
    game_state = GameState.new_game(board_size)
    
    while not game_state.is_over():
        print_board(game_state)
        
        if game_state.next_player == human_color:
            # Human turn
            print(f"\n{human_color}'s turn (YOU)")
            show_top_moves(policy_agent, game_state)
            move = get_human_move(game_state)
        else:
            # Policy agent turn
            print(f"\n{game_state.next_player}'s turn (PolicyModel)")
            move = policy_agent.select_move(game_state)
            print(f"PolicyModel plays: {move}")
        
        if move.is_resign:
            print(f"\n{game_state.next_player} resigns!")
            break
        
        game_state = game_state.apply_move(move)
    
    # Game over
    print_board(game_state)
    winner = game_state.winner()
    
    if winner is None:
        print("\n🤝 Game ended in a draw!")
    else:
        print(f"\n🏆 Winner: {winner}")


def play_policy_vs_illuminati(model_path: str, board_size: int, num_games: int, temperature: float):
    """Play multiple games between PolicyModel and IlluminatiAgent."""
    print(f"🎯 Tournament: PolicyModel vs IlluminatiAgent")
    print(f"Games: {num_games}, Board size: {board_size}x{board_size}")
    print(f"PolicyModel temperature: {temperature}")
    print()
    
    # Create agents
    policy_agent = PolicyAgent(model_path, board_size, temperature)
    illuminati_agent = IlluminatiAgent()
    
    policy_wins = 0
    illuminati_wins = 0
    draws = 0
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}...", end=" ")
        
        # Alternate colors
        if game_num % 2 == 0:
            black_agent = policy_agent
            white_agent = illuminati_agent
            policy_color = Player.black
        else:
            black_agent = illuminati_agent
            white_agent = policy_agent
            policy_color = Player.white
        
        # Play the game
        game_state = GameState.new_game(board_size)
        
        while not game_state.is_over():
            if game_state.next_player == Player.black:
                move = black_agent.select_move(game_state)
            else:
                move = white_agent.select_move(game_state)
            
            if move.is_resign:
                break
            
            game_state = game_state.apply_move(move)
        
        # Check winner
        winner = game_state.winner()
        
        if winner is None:
            draws += 1
            print("Draw")
        elif winner == policy_color:
            policy_wins += 1
            print("PolicyModel wins")
        else:
            illuminati_wins += 1
            print("IlluminatiAgent wins")
    
    # Results
    print("\n📊 Tournament Results:")
    print(f"  PolicyModel: {policy_wins}/{num_games} ({policy_wins/num_games:.1%})")
    print(f"  IlluminatiAgent: {illuminati_wins}/{num_games} ({illuminati_wins/num_games:.1%})")
    print(f"  Draws: {draws}/{num_games} ({draws/num_games:.1%})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Play Go with a trained PolicyModel')
    parser.add_argument('--model', type=str, default='best_policy_model.pth',
                        help='Path to trained model file')
    parser.add_argument('--board_size', type=int, default=9,
                        help='Board size')
    parser.add_argument('--mode', type=str, default='human',
                        choices=['human', 'tournament'],
                        help='Game mode: human vs policy or policy vs illuminati tournament')
    parser.add_argument('--human_color', type=str, default='black',
                        choices=['black', 'white'],
                        help='Human player color (for human mode)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for move selection (0=greedy, >1=random)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games (for tournament mode)')
    
    args = parser.parse_args()
    
    # Convert human color
    human_color = Player.black if args.human_color == 'black' else Player.white
    
    if args.mode == 'human':
        try:
            play_human_vs_policy(args.model, args.board_size, human_color, args.temperature)
        except KeyboardInterrupt:
            print("\n\n👋 Game interrupted. Thanks for playing!")
    else:
        play_policy_vs_illuminati(args.model, args.board_size, args.games, args.temperature)


if __name__ == "__main__":
    main() 