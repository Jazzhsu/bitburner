#!/usr/bin/env python3
"""
Interactive Go game against the Illuminati Agent.
Usage: python main_illum_agent.py [board_size]
"""

import sys
import argparse

from goboard_fast import GameState, Move, Point, Player
from ai.illuminati_agent import IlluminatiAgent
from scoring import compute_game_result

def print_board(game_state: GameState):
    """Print the current board state in a readable format."""
    board = game_state.board
    
    print("\n  ", end="")
    for col in range(1, board.num_cols + 1):
        print(f"{col:2}", end="")
    print()
    
    for row in range(1, board.num_rows + 1):
        print(f"{row:2}", end="")
        for col in range(1, board.num_cols + 1):
            point = Point(row, col)
            stone = board.get(point)
            
            if board.is_wall(point):
                print(" #", end="")    
            elif stone == Player.black:
                print(" ●", end="")
            elif stone == Player.white:
                print(" ○", end="")
            else:
                print(" ·", end="")
        print(f" {row}")
    
    print("  ", end="")
    for col in range(1, board.num_cols + 1):
        print(f"{col:2}", end="")
    print()
    
    print(f"\nNext player: {'Black ●' if game_state.next_player == Player.black else 'White ○'}")


def get_human_move(game_state: GameState) -> Move:
    """Get move input from human player."""
    while True:
        try:
            move_input = input("\nEnter your move (row col, 'pass', or 'resign'): ").strip().lower()
            
            if move_input == 'pass':
                return Move.pass_turn()
            elif move_input == 'resign':
                return Move.resign()
            
            # Parse coordinate input
            parts = move_input.split()
            if len(parts) != 2:
                print("Invalid input. Use 'row col' format (e.g., '4 4'), 'pass', or 'resign'")
                continue
            
            row, col = int(parts[0]), int(parts[1])
            point = Point(row, col)
            move = Move.play(point)
            
            if move in game_state.legal_moves():
                return move
            else:
                print("Invalid move. Try again.")
                
        except ValueError:
            print("Invalid input. Use numbers for coordinates.")
        except Exception as e:
            print(f"Error: {e}")


def print_game_result(game_state: GameState):
    """Print the final game result."""
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    
    if game_state.last_move and game_state.last_move.is_resign:
        winner = game_state.next_player  # The player who didn't resign
        print(f"Game ended by resignation.")
        print(f"Winner: {'Black ●' if winner == Player.black else 'White ○'}")
    else:
        result = compute_game_result(game_state)
        print(f"Final score: Black: {result.b}, White: {result.w}")
        if result.winner:
            print(f"Winner: {'Black ●' if result.winner == Player.black else 'White ○'}")
            print(f"Winning margin: {result.winning_margin}")
        else:
            print("Game is a tie!")


def main():
    """Main game loop."""
    parser = argparse.ArgumentParser(description="Play Go against the Illuminati Agent")
    parser.add_argument('board_size', type=int, nargs='?', default=9, 
                       help='Board size (default: 9, max: 19)')
    parser.add_argument('--human-color', choices=['black', 'white'], default='black',
                       help='Human player color (default: black)')
    
    args = parser.parse_args()
    
    # Validate board size
    if not 5 <= args.board_size <= 19:
        print("Error: Board size must be between 5 and 19")
        sys.exit(1)
    
    print(f"Starting {args.board_size}x{args.board_size} Go game")
    print(f"Human plays as {'Black ●' if args.human_color == 'black' else 'White ○'}")
    print("Enter moves as 'row col' (e.g., '4 4'), or 'pass'/'resign'")
    print("Coordinates are 1-indexed")
    
    # Initialize game
    game_state = GameState.new_game(args.board_size)
    ai_agent = IlluminatiAgent()
    
    human_player = Player.black if args.human_color == 'black' else Player.white
    ai_player = human_player.other
    
    print(f"\nAI Agent: Illuminati Agent")
    
    # Game loop
    move_count = 0
    while not game_state.is_over():
        print_board(game_state)
        
        if game_state.next_player == human_player:
            # Human turn
            print(f"\n--- Move {move_count + 1} ---")
            move = get_human_move(game_state)
            print(f"Human plays: {move}")
        else:
            # AI turn
            print(f"\n--- Move {move_count + 1} ---")
            print("AI is thinking...")
            move = ai_agent.select_move(game_state)
            print(f"AI plays: {move}")
        
        # Apply move
        game_state = game_state.apply_move(move)
        move_count += 1
        
        # Check for resignation
        if move.is_resign:
            break
    
    # Show final board and result
    print_board(game_state)
    print_game_result(game_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
        sys.exit(0)