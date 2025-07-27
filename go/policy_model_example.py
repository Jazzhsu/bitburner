#!/usr/bin/env python3
"""
Example usage of the PolicyModel with 4-channel one-hot input.

This script demonstrates:
1. Creating a PolicyModel
2. Encoding game states into 4-channel format
3. Getting move predictions with softmax probabilities
4. Converting between action indices and board points
5. Masking illegal moves
"""

import torch
from models.policy_model import PolicyModel, create_medium_policy_model
from goboard_fast import GameState, Player, Move, Point

def example_basic_usage():
    """Basic example of using the PolicyModel."""
    print("=== PolicyModel Basic Usage ===")
    
    # Create a 9x9 policy model
    board_size = 9
    model = create_medium_policy_model(board_size)
    
    print(f"Created PolicyModel for {board_size}x{board_size} board")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input channels: {model.num_input_channels}")
    print(f"Output actions: {model.num_actions} (board positions + pass)")
    print()
    
    # Create a game state
    game_state = GameState.new_game(board_size)
    current_player = Player.black
    
    # Get move prediction
    action_probs, best_action = model.predict_move(game_state, current_player)
    
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Best action index: {best_action}")
    print(f"Best action probability: {action_probs[best_action]:.4f}")
    
    # Convert action to point
    best_point = model.action_to_point(best_action)
    if best_point is None:
        print("Best move: Pass")
    else:
        print(f"Best move: {best_point}")
    
    print()

def example_4_channel_encoding():
    """Demonstrate the 4-channel encoding."""
    print("=== 4-Channel Encoding Example ===")
    
    model = PolicyModel(board_size=9)
    
    # Create a game state with some moves
    game_state = GameState.new_game(9)
    game_state = game_state.apply_move(Move.play(Point(3, 3)))  # Black
    game_state = game_state.apply_move(Move.play(Point(3, 7)))  # White
    game_state = game_state.apply_move(Move.play(Point(7, 3)))  # Black
    
    current_player = game_state.next_player  # White's turn
    
    # Encode the game state
    encoded = model.encode_game_state(game_state, current_player)
    
    print(f"Encoded state shape: {encoded.shape}")
    print("Channel meanings:")
    print("  0: Empty positions")
    print("  1: Wall positions") 
    print("  2: Self stones (current player)")
    print("  3: Opponent stones")
    print()
    
    # Show channel statistics
    for channel in range(4):
        count = torch.sum(encoded[channel]).item()
        print(f"Channel {channel}: {count} positions marked")
    
    print()

def example_legal_move_masking():
    """Demonstrate legal move masking."""
    print("=== Legal Move Masking Example ===")
    
    model = PolicyModel(board_size=5)  # Smaller board for clearer example
    game_state = GameState.new_game(5)
    current_player = Player.black
    
    # Get raw action probabilities
    action_probs, _ = model.predict_move(game_state, current_player)
    
    # Get legal actions
    legal_actions = model.get_legal_actions(game_state)
    
    print(f"Total possible actions: {model.num_actions}")
    print(f"Legal actions: {len(legal_actions)}")
    print(f"First few legal actions: {legal_actions[:10]}")
    
    # Apply legal move masking
    masked_probs = model.mask_illegal_moves(action_probs, game_state)
    
    print(f"Original probabilities sum: {action_probs.sum():.6f}")
    print(f"Masked probabilities sum: {masked_probs.sum():.6f}")
    
    # Show top legal moves
    sorted_indices = torch.argsort(masked_probs, descending=True)
    print("\nTop 5 legal moves:")
    for i in range(5):
        action_idx = sorted_indices[i].item()
        prob = masked_probs[action_idx].item()
        point = model.action_to_point(action_idx)
        move_str = "Pass" if point is None else str(point)
        print(f"  {i+1}. Action {action_idx} ({move_str}): {prob:.4f}")
    
    print()

def example_different_board_sizes():
    """Show the model works with different board sizes."""
    print("=== Different Board Sizes ===")
    
    board_sizes = [9, 13, 19]
    
    for board_size in board_sizes:
        model = PolicyModel(board_size)
        game_state = GameState.new_game(board_size)
        
        action_probs, best_action = model.predict_move(game_state, Player.black)
        
        print(f"Board {board_size}x{board_size}:")
        print(f"  Actions: {model.num_actions}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Best action: {best_action}")
    
    print()

def example_temperature_control():
    """Demonstrate temperature control for move selection."""
    print("=== Temperature Control Example ===")
    
    model = PolicyModel(board_size=9)
    game_state = GameState.new_game(9)
    current_player = Player.black
    
    temperatures = [0.1, 1.0, 2.0]
    
    for temp in temperatures:
        action_probs, best_action = model.predict_move(game_state, current_player, temperature=temp)
        
        # Calculate entropy (measure of randomness)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
        
        print(f"Temperature {temp}:")
        print(f"  Best action: {best_action}")
        print(f"  Max probability: {torch.max(action_probs):.4f}")
        print(f"  Entropy: {entropy:.2f}")
    
    print()

def main():
    """Run all examples."""
    print("PolicyModel Examples and Demonstrations")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_4_channel_encoding()
        example_legal_move_masking() 
        example_different_board_sizes()
        example_temperature_control()
        
        print("🎉 All examples completed successfully!")
        print()
        print("Key features of PolicyModel:")
        print("• 4-channel one-hot input (empty, wall, self, opponent)")
        print("• Softmax output over board_size² + 1 actions")
        print("• Residual CNN architecture for feature extraction")
        print("• Legal move masking for valid game play")
        print("• Temperature control for move randomness")
        print("• Works with any board size")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Note: Examples require PyTorch and complete Go setup")

if __name__ == "__main__":
    main() 