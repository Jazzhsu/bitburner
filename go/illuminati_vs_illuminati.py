#!/usr/bin/env python3
"""
IlluminatiAgent vs IlluminatiAgent Tournament (Multiprocessed)

This script runs tournaments between two IlluminatiAgent instances to analyze:
- Win rate variance due to randomization
- Average game length and patterns
- Color advantage (black vs white)
- Performance consistency over multiple games

Enhanced with:
- Multiprocessing for faster tournaments
- Detailed game recording with board states and moves
- Automatic saving of Black wins (rare and interesting) to disk

The IlluminatiAgent has randomized elements, so self-play helps understand
its strategic stability and decision-making patterns.
"""

import time
import random
import argparse
import multiprocessing as mp
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from goboard_fast import GameState, Player, Point, Move
from ai.illuminati_agent import IlluminatiAgent
from scoring import compute_game_result


class GameRecord:
    """Complete record of a game including all moves and board states."""
    
    def __init__(self, game_id: str, board_size: int):
        self.game_id = game_id
        self.board_size = board_size
        self.moves = []
        self.board_states = []
        self.move_times = []
        self.winner = None
        self.margin = 0
        self.total_time = 0
        self.total_moves = 0
        self.start_time = None
        self.end_time = None
    
    def add_move(self, move: Move, game_state: GameState, move_time: float):
        """Add a move and the resulting board state."""
        self.moves.append(str(move))
        self.board_states.append(self.stringify_board(game_state))
        self.move_times.append(move_time)
        self.total_moves += 1
    
    def stringify_board(self, game_state: GameState) -> str:
        """Convert board state to string representation (like main_illum_agent.py)."""
        board = game_state.board
        lines = []
        
        # Board rows
        for row in range(1, board.num_rows + 1):
            line = ""
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                stone = board.get(point)
                
                if board.is_wall(point):
                    line += "#"
                elif stone == Player.black:
                    line += "O"
                elif stone == Player.white:
                    line += "X"
                else:
                    line += "."
            lines.append(line)
        
        return "\n".join(lines)
    
    def finalize(self, result, total_time: float):
        """Finalize the game record with result."""
        self.winner = result.winner
        self.margin = result.winning_margin
        self.total_time = total_time
        self.end_time = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert game record to dictionary for JSON serialization."""
        return {
            'game_id': self.game_id,
            'board_size': self.board_size,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_time': self.total_time,
            'total_moves': self.total_moves,
            'winner': str(self.winner) if self.winner else None,
            'margin': self.margin,
            'moves': self.moves,
            'move_times': self.move_times,
            'board_states': self.board_states,
            'final_result': f"{self.winner} wins by {self.margin}" if self.winner else "Draw"
        }


class GameStats:
    """Track statistics for a single game (lightweight version for workers)."""
    
    def __init__(self):
        self.winner = None
        self.margin = 0
        self.total_time = 0
        self.total_moves = 0
        self.black_agent = "IlluminatiAgent"
        self.white_agent = "IlluminatiAgent"


class TournamentStats:
    """Track overall tournament statistics."""
    
    def __init__(self):
        self.games_played = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0
        self.total_moves = 0
        self.total_time = 0
        self.game_lengths = []
        self.winning_margins = []
        self.black_margins = []
        self.white_margins = []
        self.black_wins_saved = 0
    
    def add_game(self, game_stats: GameStats):
        """Add a completed game to the statistics."""
        self.games_played += 1
        self.total_moves += game_stats.total_moves
        self.total_time += game_stats.total_time
        self.game_lengths.append(game_stats.total_moves)
        self.winning_margins.append(abs(game_stats.margin))
        
        if game_stats.winner == Player.black:
            self.black_wins += 1
            self.black_margins.append(game_stats.margin)
        elif game_stats.winner == Player.white:
            self.white_wins += 1
            self.white_margins.append(abs(game_stats.margin))
        else:
            self.draws += 1
    
    def print_summary(self):
        """Print comprehensive tournament summary."""
        print("=" * 60)
        print("TOURNAMENT SUMMARY")
        print("=" * 60)
        
        # Basic results
        print(f"Games played: {self.games_played}")
        print(f"Black wins: {self.black_wins} ({self.black_wins/self.games_played*100:.1f}%)")
        print(f"White wins: {self.white_wins} ({self.white_wins/self.games_played*100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws/self.games_played*100:.1f}%)")
        print(f"Black wins saved to disk: {self.black_wins_saved}")
        print()
        
        # Game length statistics
        if self.game_lengths:
            avg_length = sum(self.game_lengths) / len(self.game_lengths)
            min_length = min(self.game_lengths)
            max_length = max(self.game_lengths)
            print(f"Average game length: {avg_length:.1f} moves")
            print(f"Shortest game: {min_length} moves")
            print(f"Longest game: {max_length} moves")
            print()
        
        # Timing statistics
        if self.games_played > 0:
            avg_time = self.total_time / self.games_played
            avg_move_time = self.total_time / self.total_moves if self.total_moves > 0 else 0
            print(f"Average game time: {avg_time:.2f} seconds")
            print(f"Average time per move: {avg_move_time:.3f} seconds")
            print()
        
        # Winning margin statistics
        if self.winning_margins:
            avg_margin = sum(self.winning_margins) / len(self.winning_margins)
            min_margin = min(self.winning_margins)
            max_margin = max(self.winning_margins)
            print(f"Average winning margin: {avg_margin:.1f} points")
            print(f"Smallest margin: {min_margin:.1f} points")
            print(f"Largest margin: {max_margin:.1f} points")
            print()
        
        # Color-specific margin analysis
        if self.black_margins:
            avg_black_margin = sum(self.black_margins) / len(self.black_margins)
            print(f"Average Black winning margin: {avg_black_margin:.1f} points")
        
        if self.white_margins:
            avg_white_margin = sum(self.white_margins) / len(self.white_margins)
            print(f"Average White winning margin: {avg_white_margin:.1f} points")
        
        print()
        
        # Performance consistency analysis
        if len(self.game_lengths) > 1:
            import statistics
            length_stddev = statistics.stdev(self.game_lengths)
            print(f"Game length consistency (std dev): {length_stddev:.1f} moves")
            
            if self.winning_margins:
                margin_stddev = statistics.stdev(self.winning_margins)
                print(f"Winning margin consistency (std dev): {margin_stddev:.1f} points")


def save_black_win_game(game_record: GameRecord, output_dir: str = "black_wins"):
    """Save a Black win game to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"black_win_{game_record.game_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(game_record.to_dict(), f, indent=2)
        print(f"  💾 Saved Black win game: {filename}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to save game: {e}")
        return False


def simulate_game_worker(args) -> Tuple[GameStats, Optional[GameRecord]]:
    """Worker function for multiprocessing - simulates a single game."""
    game_id, board_size, record_games, worker_seed = args
    
    # Set unique seed for this worker
    random.seed(worker_seed)
    
    # Create fresh agents
    black_agent = IlluminatiAgent()
    white_agent = IlluminatiAgent()
    
    game_state = GameState.new_game(board_size)
    agents = {Player.black: black_agent, Player.white: white_agent}
    
    # Initialize game recording
    game_record = GameRecord(game_id, board_size) if record_games else None
    if game_record:
        game_record.start_time = datetime.now().isoformat()
        game_record.board_states.append(game_record.stringify_board(game_state))  # Initial state
    
    stats = GameStats()
    start_time = time.time()
    max_moves = 400
    last_moves = []  # Track last few moves for pass detection
    
    while not game_state.is_over() and stats.total_moves < max_moves:
        current_player = game_state.next_player
        agent = agents[current_player]
        
        move_start = time.time()
        move = agent.select_move(game_state)
        move_time = time.time() - move_start
        
        game_state = game_state.apply_move(move)
        stats.total_moves += 1
        last_moves.append(move)
        
        # Keep only last 2 moves for pass detection
        if len(last_moves) > 2:
            last_moves.pop(0)
        
        # Record move and resulting board state
        if game_record:
            game_record.add_move(move, game_state, move_time)
        
        # Check for consecutive passes (game end)
        if (len(last_moves) >= 2 and 
            all(hasattr(m, 'is_pass') and m.is_pass for m in last_moves[-2:])):
            break
    
    stats.total_time = time.time() - start_time
    result = compute_game_result(game_state)
    stats.winner = result.winner
    stats.margin = result.winning_margin
    
    # Finalize game record
    if game_record:
        game_record.finalize(result, stats.total_time)
    
    return stats, game_record


def run_tournament_mp(num_games: int = 50, board_size: int = 9, 
                     num_workers: int = None, record_games: bool = True,
                     seed: int = None, show_progress: bool = True) -> TournamentStats:
    """Run a multiprocessed tournament between IlluminatiAgent instances."""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), max(1, num_games // 4))
    
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    tournament_stats = TournamentStats()
    
    print(f"Running {num_games} game tournament between IlluminatiAgent instances")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Workers: {num_workers}")
    print(f"Recording games: {'Yes' if record_games else 'No'}")
    print("=" * 60)
    
    # Prepare worker arguments
    worker_args = []
    for game_num in range(num_games):
        game_id = f"game_{game_num:04d}"
        worker_seed = random.randint(0, 2**32 - 1) if seed is None else seed + game_num
        worker_args.append((game_id, board_size, record_games, worker_seed))
    
    # Run games in parallel
    start_time = time.time()
    
    with mp.Pool(num_workers) as pool:
        if show_progress:
            print("Starting multiprocessed games...")
            
        results = pool.map(simulate_game_worker, worker_args)
        
        # Process results
        for i, (game_stats, game_record) in enumerate(results):
            tournament_stats.add_game(game_stats)
            
            # Save Black wins to disk
            if game_record and game_stats.winner == Player.black:
                if save_black_win_game(game_record):
                    tournament_stats.black_wins_saved += 1
            
            if show_progress and (i + 1) % max(1, num_games // 10) == 0:
                progress = (i + 1) / num_games * 100
                current_black_rate = tournament_stats.black_wins / (i + 1) * 100
                print(f"  Progress: {i + 1}/{num_games} ({progress:.0f}%) - Black win rate: {current_black_rate:.1f}%")
    
    total_time = time.time() - start_time
    print(f"\nTournament completed in {total_time:.2f} seconds")
    print(f"Average time per game: {total_time/num_games:.3f} seconds")
    
    return tournament_stats


def analyze_randomness_mp(num_games: int = 100, board_size: int = 9, num_workers: int = None):
    """Analyze how randomization affects game outcomes using multiprocessing."""
    print("RANDOMNESS ANALYSIS (Multiprocessed)")
    print("=" * 40)
    
    # Run with different seeds to see variance
    win_rates = []
    
    for seed in range(5):  # Test 5 different seeds
        print(f"\nTesting with seed {seed}...")
        stats = run_tournament_mp(num_games, board_size, num_workers, 
                                 record_games=False, seed=seed, show_progress=False)
        
        black_rate = stats.black_wins / stats.games_played * 100
        win_rates.append(black_rate)
        print(f"Seed {seed}: Black win rate = {black_rate:.1f}%")
    
    # Calculate variance in win rates
    if len(win_rates) > 1:
        import statistics
        mean_rate = statistics.mean(win_rates)
        stddev_rate = statistics.stdev(win_rates)
        
        print(f"\nRandomness Analysis:")
        print(f"Mean Black win rate: {mean_rate:.1f}%")
        print(f"Standard deviation: {stddev_rate:.1f}%")
        print(f"Min win rate: {min(win_rates):.1f}%")
        print(f"Max win rate: {max(win_rates):.1f}%")
        print(f"Range: {max(win_rates) - min(win_rates):.1f}%")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="IlluminatiAgent vs IlluminatiAgent Tournament (Multiprocessed)")
    parser.add_argument('--games', '-g', type=int, default=50,
                       help='Number of games to play (default: 50)')
    parser.add_argument('--board-size', '-b', type=int, default=9,
                       help='Board size (default: 9)')
    parser.add_argument('--workers', '-w', type=int,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--no-record', action='store_true',
                       help='Disable game recording (faster)')
    parser.add_argument('--seed', '-s', type=int,
                       help='Random seed for reproducible results')
    parser.add_argument('--analyze-randomness', '-r', action='store_true',
                       help='Analyze how randomization affects outcomes')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick tournament (20 games, no recording)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.games = 20
        args.no_record = True
        print("Quick tournament mode (no recording)")
    
    if args.analyze_randomness:
        analyze_randomness_mp(args.games, args.board_size, args.workers)
        return
    
    # Run main tournament
    tournament_stats = run_tournament_mp(
        num_games=args.games,
        board_size=args.board_size,
        num_workers=args.workers,
        record_games=not args.no_record,
        seed=args.seed,
        show_progress=True
    )
    
    # Print results
    tournament_stats.print_summary()
    
    # Additional insights
    print("INSIGHTS:")
    black_win_rate = tournament_stats.black_wins / tournament_stats.games_played * 100
    
    if abs(black_win_rate - 50) > 10:
        if black_win_rate > 60:
            print("• Black shows significant advantage - possible first-move advantage")
        elif black_win_rate < 40:
            print("• White shows significant advantage - possible komi effect or defensive advantage")
    else:
        print("• Win rates are relatively balanced between colors")
    
    if tournament_stats.draws > tournament_stats.games_played * 0.1:
        print(f"• High draw rate ({tournament_stats.draws/tournament_stats.games_played*100:.1f}%) suggests conservative play")
    
    if tournament_stats.game_lengths:
        avg_length = sum(tournament_stats.game_lengths) / len(tournament_stats.game_lengths)
        if avg_length < 50:
            print("• Games are relatively short - aggressive/tactical play")
        elif avg_length > 150:
            print("• Games are relatively long - strategic/territorial play")
    
    if tournament_stats.black_wins_saved > 0:
        print(f"• {tournament_stats.black_wins_saved} Black win games saved to 'black_wins/' directory")
        print("  These rare games can be analyzed to understand Black's winning patterns")
    
    print(f"\nTo run more tests:")
    print(f"python illuminati_vs_illuminati.py --games 200 --workers 8")
    print(f"python illuminati_vs_illuminati.py --analyze-randomness --games 100")
    print(f"python illuminati_vs_illuminati.py --quick  # Fast test with no recording")


if __name__ == "__main__":
    main() 