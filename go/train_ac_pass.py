import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np
import torch

import scoring
import rl
from rl.experience import ExperienceCollector, combine_experience, load_experience
from rl import ac_pass
from goboard_fast import GameState, Player, Point
from encoders import alphago
from models.ac_model import ACModel
from ai.illuminati_agent import IlluminatiAgent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def load_agent(filename, board_size):
    model_dict = torch.load(filename)
    model = ACModel(alphago.AlphaGoEncoder((board_size, board_size)))
    model.load_state_dict(model_dict)
    encoder = alphago.AlphaGoEncoder((board_size, board_size))
    return ac_pass.ACAgent(model, encoder, DEVICE)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            if board.is_wall(Point(row=row, col=col)):
                line.append('#')
            else:
                line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    num_moves = 0
    while not game.is_over():
        agents[game.next_player]
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
        num_moves += 1

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, num_games, experience_filename):

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename, board_size)
    reference_agent = IlluminatiAgent()

    collector1 = ExperienceCollector()

    black_player = agent1
    white_player = reference_agent
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)

        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == Player.black:
            print('RL Agent wins.')
            collector1.complete_episode(reward=1)
        else:
            print('Reference Agent wins.')
            collector1.complete_episode(reward=-1)

    experience = combine_experience([collector1])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def generate_experience(learning_agent, exp_file,
                        num_games, board_size, num_workers):
    print(f"Generating experience for {learning_agent} with {num_games} games and {num_workers} workers")
    experience_files = []
    workers = []
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                games_per_worker,
                filename,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = load_experience(expf)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = load_experience(expf)
        combined_buffer = combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size, board_size):
    learning_agent = load_agent(learning_agent, board_size)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = load_experience(expf)
    learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)
    torch.save(learning_agent.model.state_dict(), output_file)


def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size, board_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    print(f"Training {learning_agent} on {experience_file} with {lr} learning rate and {batch_size} batch size")
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size,
            board_size
        )
    )
    worker.start()
    worker.join()


def play_games(args):
    agent1_fname, num_games, board_size = args

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname, board_size)
    reference_agent = IlluminatiAgent()

    wins, losses = 0, 0
    black_player = agent1
    white_player = reference_agent
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == Player.black:
            print('RL Agent wins')
            wins += 1
        else:
            print('Reference Agent wins')
            losses += 1
        print('RL Agent record: %d/%d' % (wins, wins + losses))
    return wins, losses


def evaluate(learning_agent,
             num_games, num_workers, board_size):
    print(f"Evaluating {learning_agent} on {num_games} games with {num_workers} workers")
    games_per_worker = num_games // num_workers
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent,
            games_per_worker,
            board_size,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args[0])

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent')
    parser.add_argument('--games-per-batch', '-g', type=int, default=1000)
    parser.add_argument('--work-dir', '-d')
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l')

    args = parser.parse_args()

    if args.agent is None:
        model = ACModel(alphago.AlphaGoEncoder((args.board_size, args.board_size)))
        torch.save(model.state_dict(), "model.pt")
        args.agent = "model.pt"

    logf = open(args.log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting from %s at %s\n' % (
        args.agent, datetime.datetime.now()))

    learning_agent = args.agent
    experience_file = os.path.join(args.work_dir, 'exp_temp.hdf5')
    tmp_agent = os.path.join(args.work_dir, 'agent_temp.hdf5')
    working_agent = os.path.join(args.work_dir, 'agent_cur.hdf5')
    best_agent = os.path.join(args.work_dir, 'agent_best.hdf5')
    highest_win_count = 0
    total_games = 0
    while True:
        logf.write('Total games so far %d\n' % (total_games,))
        generate_experience(
            learning_agent,
            experience_file,
            num_games=args.games_per_batch,
            board_size=args.board_size,
            num_workers=args.num_workers)
        train_on_experience(
            learning_agent, tmp_agent, experience_file,
            lr=args.lr, batch_size=args.bs, board_size=args.board_size)
        total_games += args.games_per_batch
        wins = evaluate(
            learning_agent,
            num_games=480,
            num_workers=args.num_workers,
            board_size=args.board_size)
        print('Won %d / 480 games (%.3f)' % (
            wins, float(wins) / 480.0))
        logf.write('Won %d / 480 games (%.3f)\n' % (
            wins, float(wins) / 480.0))
        shutil.copy(tmp_agent, working_agent)
        learning_agent = working_agent
        if wins > highest_win_count:
            highest_win_count = wins
            shutil.copy(tmp_agent, best_agent)
            logf.write('New best agent is %s\n' % best_agent)
        logf.flush()


if __name__ == '__main__':
    main()