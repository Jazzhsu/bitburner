#!/usr/bin/env python3
"""
Test script to verify Black win data format and parsing.

This script helps debug the training data pipeline by:
1. Loading a sample Black win game
2. Parsing board states and moves
3. Verifying the 4-channel encoding
4. Testing the training data creation process
"""

import os
import json
from typing import Dict, Any
from train_policy_from_black_wins import load_black_win_games, parse_board_string, parse_move_string, encode_board_grid
from models.policy_model import create_medium_policy_model
from goboard_fast import Player

def test_data_loading():
    """Test loading Black win games from JSON files."""
    print("=== Testing Data Loading ===")
    
    black_wins_dir = "black_wins"
    
    if not os.path.exists(black_wins_dir):
        print(f"❌ Directory {black_wins_dir} does not exist!")
        print("Run: python illuminati_vs_illuminati.py --games 50 --workers 4")
        return False
    
    games = load_black_win_games(black_wins_dir)
    
    if not games:
        print("❌ No games found!")
        return False
    
    print(f"✅ Successfully loaded {len(games)} games")
    
    # Show sample game info
    sample_game = games[0]
    print(f"\nSample game info:")
    print(f"  Game ID: {sample_game.get('game_id', 'N/A')}")
    print(f"  Board size: {sample_game.get('board_size', 'N/A')}")
    print(f"  Total moves: {sample_game.get('total_moves', 'N/A')}")
    print(f"  Winner: {sample_game.get('winner', 'N/A')}")
    print(f"  Margin: {sample_game.get('margin', 'N/A')}")
    print(f"  Board states: {len(sample_game.get('board_states', []))}")
    print(f"  Moves: {len(sample_game.get('moves', []))}")
    
    return True

def test_board_parsing():
    """Test parsing board strings back to grid format."""
    print("\n=== Testing Board Parsing ===")
    
    games = load_black_win_games("black_wins")
    if not games:
        return False
    
    sample_game = games[0]
    board_size = sample_game['board_size']
    board_states = sample_game['board_states']
    
    if not board_states:
        print("❌ No board states found!")
        return False
    
    # Test parsing the first board state
    first_board_str = board_states[0]
    print(f"Original board string (first 200 chars):")
    print(first_board_str[:200] + "...")
    
    board_grid, next_player = parse_board_string(first_board_str, board_size)
    
    print(f"\nParsed board grid:")
    print(f"  Grid size: {len(board_grid)}x{len(board_grid[0]) if board_grid else 0}")
    print(f"  Next player: {next_player}")
    
    # Show the parsed board
    if board_grid:
        print(f"  Board representation:")
        for row in board_grid:
            row_str = ""
            for cell in row:
                if cell == 'empty':
                    row_str += "."
                elif cell == 'wall':
                    row_str += "#"
                elif cell == 'black':
                    row_str += "X"
                elif cell == 'white':
                    row_str += "O"
            print(f"    {row_str}")
    
    return True

def test_move_parsing():
    """Test parsing move strings."""
    print("\n=== Testing Move Parsing ===")
    
    games = load_black_win_games("black_wins")
    if not games:
        return False
    
    sample_game = games[0]
    moves = sample_game['moves']
    board_size = sample_game['board_size']
    
    if not moves:
        print("❌ No moves found!")
        return False
    
    print(f"Testing first 5 moves:")
    for i, move_str in enumerate(moves[:5]):
        parsed_move = parse_move_string(move_str, board_size)
        print(f"  {i+1}. '{move_str}' -> {parsed_move}")
    
    # Test specific format parsing
    print(f"\nTesting move format parsing:")
    test_moves = ["(r 3, c 4)", "Pass", "Resign", "(r 1, c 1)"]
    for test_move in test_moves:
        parsed = parse_move_string(test_move, board_size)
        print(f"  '{test_move}' -> {parsed}")
    
    return True

def test_encoding():
    """Test 4-channel board encoding."""
    print("\n=== Testing 4-Channel Encoding ===")
    
    games = load_black_win_games("black_wins")
    if not games:
        return False
    
    sample_game = games[0]
    board_size = sample_game['board_size']
    board_states = sample_game['board_states']
    
    if not board_states:
        return False
    
    # Parse a board state
    board_str = board_states[len(board_states)//2]  # Middle of game
    board_grid, next_player = parse_board_string(board_str, board_size)
    
    if not board_grid:
        print("❌ Failed to parse board!")
        return False
    
    # Encode to 4-channel format
    encoded = encode_board_grid(board_grid, board_size, next_player)
    
    print(f"Encoded tensor shape: {encoded.shape}")
    print(f"Current player: {next_player}")
    
    # Count elements in each channel
    for channel in range(4):
        count = int(encoded[channel].sum().item())
        channel_name = ['Empty', 'Wall', 'Self', 'Opponent'][channel]
        print(f"  Channel {channel} ({channel_name}): {count} positions")
    
    return True

def test_training_data_creation():
    """Test full training data creation pipeline."""
    print("\n=== Testing Training Data Creation (Black Only) ===")
    
    games = load_black_win_games("black_wins")
    if not games:
        return False
    
    # Use just one game for testing
    test_games = games[:1]
    board_size = test_games[0]['board_size']
    
    # Create model for the board size
    model = create_medium_policy_model(board_size)
    print(f"Created model for {board_size}x{board_size} board")
    
    # Import the training data creation function
    from train_policy_from_black_wins import create_training_data
    
    positions, actions = create_training_data(test_games, model)
    
    if positions:
        print(f"Position shape: {positions[0].shape}")
        print(f"Sample actions: {actions[:10]}")
        print(f"Action range: {min(actions)} to {max(actions)}")
        print(f"Expected max action: {model.num_actions - 1}")
        
        # Verify all positions are encoded from Black's perspective
        print(f"\nVerifying Black perspective encoding:")
        sample_pos = positions[0]
        print(f"  Channel 2 (Self/Black): {sample_pos[2].sum().item()} stones")
        print(f"  Channel 3 (Opponent/White): {sample_pos[3].sum().item()} stones")
        print("  ✅ All positions should be from Black's perspective")
    
    return len(positions) > 0

def show_sample_board_evolution():
    """Show how a board evolves through a game."""
    print("\n=== Sample Board Evolution ===")
    
    games = load_black_win_games("black_wins")
    if not games:
        return False
    
    sample_game = games[0]
    board_size = sample_game['board_size']
    board_states = sample_game['board_states']
    moves = sample_game['moves']
    
    # Show first, middle, and last positions
    positions_to_show = [0, len(board_states)//2, len(board_states)-1]
    
    for pos_idx in positions_to_show:
        if pos_idx >= len(board_states):
            continue
            
        print(f"\nPosition {pos_idx+1} (Move: {moves[pos_idx] if pos_idx < len(moves) else 'End'}):")
        
        board_grid, next_player = parse_board_string(board_states[pos_idx], board_size)
        
        if board_grid:
            for row in board_grid:
                row_str = ""
                for cell in row:
                    if cell == 'empty':
                        row_str += " ."
                    elif cell == 'wall':
                        row_str += " #"
                    elif cell == 'black':
                        row_str += " X"
                    elif cell == 'white':
                        row_str += " O"
                print(f"  {row_str}")
            print(f"  Next: {next_player}")
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Black Win Data Loading and Parsing")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Board Parsing", test_board_parsing),
        ("Move Parsing", test_move_parsing),
        ("4-Channel Encoding", test_encoding),
        ("Training Data Creation", test_training_data_creation),
        ("Board Evolution", show_sample_board_evolution),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print()
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results[test_name] = f"❌ ERROR: {e}"
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    
    all_passed = all("✅" in result for result in results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train the PolicyModel.")
        print("Run: python train_policy_from_black_wins.py --epochs 10 --batch_size 16")
    else:
        print("\n⚠️  Some tests failed. Check the data format and dependencies.")

if __name__ == "__main__":
    main() 