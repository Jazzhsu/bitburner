#!/usr/bin/env python3
"""
Training script for PolicyModel using Black win games.

This script:
1. Loads game records from black_wins/ directory
2. Converts game states and moves to training data
3. Trains a PolicyModel using supervised learning
4. Saves the trained model
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import argparse
from typing import List, Tuple, Dict, Any

from models.policy_model import PolicyModel, create_medium_policy_model
from goboard_fast import GameState, Player, Move, Point


class GoGameDataset(Dataset):
    """Dataset for Go game positions and moves."""
    
    def __init__(self, positions: List[torch.Tensor], actions: List[int]):
        """
        Initialize dataset.
        
        Args:
            positions: List of 4-channel board encodings
            actions: List of action indices (targets)
        """
        self.positions = positions
        self.actions = actions
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.actions[idx]


def parse_board_string(board_str: str, board_size: int) -> Tuple[List[List[str]], Player]:
    """
    Parse a stringified board back to 2D representation.
    
    Args:
        board_str: String representation of board from JSON
        board_size: Size of the board
        
    Returns:
        Tuple of (board_grid, next_player)
    """
    lines = board_str.strip().split('\n')
    
    # Find the board lines (skip headers/footers)
    board_lines = []
    next_player = Player.black  # default
    
    for line in lines:
        # Skip empty lines and headers
        if not line.strip():
            continue

            # This is a board row
            # Extract just the symbols (●, ○, ·, #)
        symbols = []
        chars = line.strip()
        i = 0
        while i < len(chars) and len(symbols) < board_size:
            if chars[i] in ['●', '○', '·', '#', 'X', 'O', '.']:
                if chars[i] == '●' or chars[i] == 'X':
                    symbols.append('black')
                elif chars[i] == '○' or chars[i] == 'O':
                    symbols.append('white') 
                elif chars[i] == '#':
                    symbols.append('wall')
                else:  # '·' or '.'
                    symbols.append('empty')
            i += 1
        
        if len(symbols) == board_size:
            board_lines.append(symbols)
    
    return board_lines, next_player


def create_game_state_from_board(board_grid: List[List[str]], board_size: int, next_player: Player) -> GameState:
    """
    Create a GameState from parsed board grid.
    
    Args:
        board_grid: 2D list of strings ('black', 'white', 'empty', 'wall')
        board_size: Size of the board
        next_player: Player to move next
        
    Returns:
        GameState object
    """
    # Start with a new game
    game_state = GameState.new_game(board_size)
    
    # Apply moves to recreate the board state
    # This is a simplified approach - ideally we'd replay the actual move sequence
    moves = []
    
    for row in range(board_size):
        for col in range(board_size):
            point = Point(row + 1, col + 1)  # Convert to 1-indexed
            cell = board_grid[row][col]
            
            if cell == 'black':
                moves.append((Move.play(point), Player.black))
            elif cell == 'white':
                moves.append((Move.play(point), Player.white))
    
    # Apply moves alternating players (this is approximate)
    current_player = Player.black
    for move, intended_player in moves:
        if current_player == intended_player:
            game_state = game_state.apply_move(move)
            current_player = current_player.other
        # Skip moves that don't match the expected player (simplified)
    
    return game_state


def parse_move_string(move_str: str, board_size: int) -> Move:
    """
    Parse a move string to Move object.
    
    Args:
        move_str: String like "(r 3, c 4)" or "Pass" or "Resign"
        board_size: Board size for validation
        
    Returns:
        Move object
    """
    move_str = move_str.strip()
    
    if move_str.lower() == 'pass':
        return Move.pass_turn()
    elif move_str.lower() == 'resign':
        return Move.resign()
    elif move_str.startswith('(r ') and move_str.endswith(')'):
        # Extract coordinates from "(r 3, c 4)"
        try:
            # Remove parentheses and split by comma
            inner = move_str[1:-1]  # Remove "(" and ")"
            parts = inner.split(', ')
            
            # Parse "r 3" and "c 4"
            row_part = parts[0].strip()  # "r 3"
            col_part = parts[1].strip()  # "c 4"
            
            if row_part.startswith('r ') and col_part.startswith('c '):
                row = int(row_part[2:])  # Extract number after "r "
                col = int(col_part[2:])  # Extract number after "c "
                return Move.play(Point(row, col))
            else:
                return Move.pass_turn()  # Fallback
        except:
            return Move.pass_turn()  # Fallback
    else:
        return Move.pass_turn()  # Fallback


def load_black_win_games(black_wins_dir: str) -> List[Dict[str, Any]]:
    """
    Load all Black win games from JSON files.
    
    Args:
        black_wins_dir: Directory containing JSON game files
        
    Returns:
        List of game dictionaries
    """
    games = []
    
    if not os.path.exists(black_wins_dir):
        print(f"Directory {black_wins_dir} does not exist!")
        return games
    
    json_files = [f for f in os.listdir(black_wins_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {black_wins_dir}")
    
    for json_file in json_files:
        filepath = os.path.join(black_wins_dir, json_file)
        try:
            with open(filepath, 'r') as f:
                game_data = json.load(f)
                games.append(game_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Successfully loaded {len(games)} games")
    return games


def create_training_data(games: List[Dict[str, Any]], model: PolicyModel) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Convert game data to training data from BLACK'S PERSPECTIVE ONLY.
    
    This function filters to only include positions where Black is to move,
    ensuring consistent channel encoding (Black=Self, White=Opponent).
    
    Args:
        games: List of game dictionaries
        model: PolicyModel for encoding states
        
    Returns:
        Tuple of (encoded_positions, target_actions) - all from Black's perspective
    """
    positions = []
    actions = []
    total_positions = 0
    black_positions = 0
    
    for game_idx, game in enumerate(games):
        board_size = game['board_size']
        board_states = game['board_states']
        moves = game['moves']
        
        print(f"Processing game {game_idx + 1}/{len(games)} (board_size: {board_size}, moves: {len(moves)})")
        
        # Skip if wrong board Size
        if board_size != model.board_size:
            continue
        
        # Process each position-move pair (BLACK PERSPECTIVE ONLY)
        for i in range(len(moves) - 1):  # Skip last move (no next move to learn)
            try:
                # Parse the board state
                board_grid, next_player = parse_board_string(board_states[i], board_size)
                
                if len(board_grid) != board_size:
                    print(f"board_size: {board_size}, Model board size: {board_grid}")
                    continue  # Skip malformed boards
                
                total_positions += 1
                
                # FILTER: Only include positions where Black is to move
                if next_player != Player.black:
                    continue
                
                black_positions += 1
                
                # Parse the next move
                next_move = parse_move_string(moves[i], board_size)
                
                
                # Encode the position using the simplified board grid
                # Always from Black's perspective (Black=Self, White=Opponent)
                encoded_pos = encode_board_grid(board_grid, board_size, Player.black)
                
                # Convert move to action index
                if next_move.is_pass:
                    action_idx = model.num_positions  # Pass action
                elif next_move.is_play:
                    action_idx = model.point_to_action(next_move.point)
                else:
                    continue  # Skip resign moves
                
                positions.append(encoded_pos)
                actions.append(action_idx)
                
            except Exception as e:
                print(f"Error processing position {i} in game {game_idx}: {e}")
                continue
    
    print(f"Position statistics:")
    print(f"  Total positions processed: {total_positions}")
    print(f"  Black-to-move positions: {black_positions} ({black_positions/total_positions*100:.1f}%)")
    print(f"  White-to-move positions: {total_positions - black_positions} (filtered out)")
    print(f"  Final training examples: {len(positions)} (Black perspective only)")
    return positions, actions


def encode_board_grid(board_grid: List[List[str]], board_size: int, current_player: Player) -> torch.Tensor:
    """
    Encode a board grid into 4-channel format.
    
    Args:
        board_grid: 2D list of strings
        board_size: Size of the board
        current_player: Current player to move
        
    Returns:
        4-channel encoded tensor
    """
    encoded = torch.zeros(4, board_size, board_size, dtype=torch.float32)
    
    for row in range(board_size):
        for col in range(board_size):
            cell = board_grid[row][col]
            
            if cell == 'empty':
                encoded[0, row, col] = 1.0  # Empty channel
            elif cell == 'wall':
                encoded[1, row, col] = 1.0  # Wall channel
            elif cell == 'black':
                if current_player == Player.black:
                    encoded[2, row, col] = 1.0  # Self channel
                else:
                    encoded[3, row, col] = 1.0  # Opponent channel
            elif cell == 'white':
                if current_player == Player.white:
                    encoded[2, row, col] = 1.0  # Self channel
                else:
                    encoded[3, row, col] = 1.0  # Opponent channel
    
    return encoded


def train_model(model: PolicyModel, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, learning_rate: float, device: torch.device) -> None:
    """
    Train the PolicyModel.
    
    Args:
        model: PolicyModel to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (positions, actions) in enumerate(train_loader):
            positions, actions = positions.to(device), actions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            action_probs = model(positions)
            loss = criterion(action_probs, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(action_probs.data, 1)
            train_total += actions.size(0)
            train_correct += (predicted == actions).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for positions, actions in val_loader:
                positions, actions = positions.to(device), actions.to(device)
                
                action_probs = model(positions)
                loss = criterion(action_probs, actions)
                
                val_loss += loss.item()
                _, predicted = torch.max(action_probs.data, 1)
                val_total += actions.size(0)
                val_correct += (predicted == actions).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_policy_model.pth')
            print(f'  💾 Saved best model (val_loss: {val_loss:.4f})')
        
        scheduler.step()
        print()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PolicyModel from Black win games')
    parser.add_argument('--black_wins_dir', type=str, default='black_wins',
                        help='Directory containing Black win JSON files')
    parser.add_argument('--board_size', type=int, default=9,
                        help='Board size to train on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Model size')
    
    args = parser.parse_args()
    
    print("🚀 Starting PolicyModel training from Black win games")
    print("📋 Training from BLACK'S PERSPECTIVE ONLY")
    print("   Channel encoding: Empty=0, Wall=1, Black=Self=2, White=Opponent=3")
    print(f"Settings: board_size={args.board_size}, batch_size={args.batch_size}, "
          f"epochs={args.epochs}, lr={args.lr}")
    print()
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model_size == 'small':
        from models.policy_model import create_small_policy_model
        model = create_small_policy_model(args.board_size)
    elif args.model_size == 'large':
        from models.policy_model import create_large_policy_model
        model = create_large_policy_model(args.board_size)
    else:
        model = create_medium_policy_model(args.board_size)
    
    print(f"Created {args.model_size} PolicyModel with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load games
    games = load_black_win_games(args.black_wins_dir)
    if not games:
        print("❌ No games found! Run illuminati_vs_illuminati.py first to generate training data.")
        return
    
    # Create training data
    positions, actions = create_training_data(games, model)
    if not positions:
        print("❌ No training data created! Check board size and game format.")
        return
    
    # Split data
    num_train = int(len(positions) * (1 - args.val_split))
    train_positions = positions[:num_train]
    train_actions = actions[:num_train]
    val_positions = positions[num_train:]
    val_actions = actions[num_train:]
    
    print(f"Training examples: {len(train_positions)}")
    print(f"Validation examples: {len(val_positions)}")
    
    # Create datasets and loaders
    train_dataset = GoGameDataset(train_positions, train_actions)
    val_dataset = GoGameDataset(val_positions, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Train model
    print("\n🎯 Starting training...")
    start_time = datetime.now()
    
    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    
    end_time = datetime.now()
    print(f"✅ Training completed in {end_time - start_time}")
    
    # Save final model
    torch.save(model.state_dict(), f'policy_model_final_{args.board_size}x{args.board_size}.pth')
    print(f"💾 Saved final model to policy_model_final_{args.board_size}x{args.board_size}.pth")


if __name__ == "__main__":
    main() 