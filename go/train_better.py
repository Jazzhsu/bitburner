"""
Moderate training improvement for better Go play.
Balanced approach: 200 iterations with enhanced parameters.
Should take ~2-3 hours but produce much stronger play.
"""
from collections import deque
import multiprocessing as mp

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from go import GO, StoneType, opposite
from MCTS import MCTS, GOSimulator
from model import GoNNet

BOARD_SIZE = 7
NUM_ITERATIONS = 200  # 🔧 Moderate increase (2x original)
NUM_SELF_PLAY_GAMES = 40  # Slightly fewer games but better quality
MCTS_SIMULATIONS = 75   # 🔧 50% more simulations
TRAINING_EPOCHS = 12    # 🔧 More training per iteration
BATCH_SIZE = 64
LEARNING_RATE = 1.5e-4  # 🔧 Slightly higher learning rate
CPUCT = 1.8  # 🔧 Better exploration balance
DEVICE = "mps"

def self_play_worker(nnet_state_dict):
    """Worker function for improved self-play."""
    nnet = GoNNet(BOARD_SIZE)
    nnet.load_state_dict(nnet_state_dict)
    
    episode_examples = execute_episode(nnet)
    return episode_examples

def execute_episode(nnet):
    """Enhanced self-play with better game quality."""
    train_examples = []
    go = GOSimulator(go=GO(BOARD_SIZE), turn=StoneType.WHITE)

    move_counter = 0
    consecutive_passes = 0

    while True:
        # 🔧 Adaptive MCTS simulations
        if move_counter < 5:
            sims = MCTS_SIMULATIONS - 15  # Faster early game
        elif move_counter < 15:
            sims = MCTS_SIMULATIONS  # Standard mid-game
        else:
            sims = MCTS_SIMULATIONS + 15  # More careful endgame

        mcts = MCTS(nnet, sims, CPUCT, device='cpu')

        # 🔧 Better temperature schedule
        if move_counter < 6:
            temp = 1.1  # High exploration early
            add_noise = True
        elif move_counter < 18:
            temp = 0.8  # Moderate exploration mid-game  
            add_noise = True
        else:
            temp = 0.2  # Focused endgame
            add_noise = False

        policy = mcts.get_action_probs(go, temp=temp, add_noise=add_noise)
        
        # 🔧 Strong pass prevention in early game
        training_policy = policy.copy()
        if move_counter < 8:
            valid_board_moves = [a for a in range(go.get_action_size() - 1) 
                               if go.is_valid_move(a)]
            if len(valid_board_moves) > 0:
                # Strong pass reduction
                training_policy[-1] *= 0.02  # 98% reduction
                total_prob = sum(training_policy)
                if total_prob > 0:
                    training_policy = [p / total_prob for p in training_policy]
        
        train_examples.append([go.torch_state(), go.player(), training_policy])
        
        # Choose action
        action = np.random.choice(len(training_policy), p=training_policy)
        
        if action == go.get_action_size() - 1:
            consecutive_passes += 1
        else:
            consecutive_passes = 0
        
        go.move(action)
        move_counter += 1
        
        # 🔧 Better game termination
        min_moves = max(8, BOARD_SIZE * 1.8)  # Slightly longer games
        max_moves = (BOARD_SIZE ** 2) * 1.3
        
        game_should_end = (
            (go.is_terminal() and move_counter >= min_moves) or 
            move_counter >= max_moves or
            consecutive_passes >= 2
        )
        
        if game_should_end:
            eval = go.eval()
            examples = []
            
            for board_state, player, policy in train_examples:
                value = 0
                if go.is_terminal():
                    norm = eval[StoneType.EMPTY] / 2.
                    if norm > 0:
                        value = (eval[player] - norm) / norm
                    else:
                        total_stones = eval[StoneType.WHITE] + eval[StoneType.BLACK]
                        if total_stones > 0:
                            value = (eval[player] / total_stones) * 2 - 1
                        else:
                            value = 0
                
                # 🔧 Quality-based value adjustment
                if move_counter < min_moves:
                    value *= 0.15  # Penalty for too short
                elif move_counter > min_moves + 8:
                    value *= 1.1   # Slight bonus for longer strategic games
                
                examples.append((board_state, policy, value))

            return examples

def train():
    """Moderate training improvement."""
    print("⚡ STARTING BETTER TRAINING (200 iterations)")
    
    nnet = GoNNet(BOARD_SIZE).to(DEVICE)
    
    # 🔧 Better optimizer
    optimizer = optim.Adam(nnet.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
    
    # 🔧 Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.85)
    
    training_data = deque(maxlen=25 * NUM_SELF_PLAY_GAMES)

    for i in range(NUM_ITERATIONS):
        print(f"--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        
        # Self-Play Phase
        print("Starting better self-play...")
        nnet.eval()

        cpu_nnet_state_dict = nnet.to('cpu').state_dict()
        
        num_cores = max(1, 6)
        with mp.Pool(num_cores) as pool:
            tasks = [cpu_nnet_state_dict] * NUM_SELF_PLAY_GAMES
            for result in tqdm(pool.imap_unordered(self_play_worker, tasks), 
                             total=len(tasks)):
                training_data.extend(result)

        nnet.to(DEVICE)
        
        # Training Phase
        print("Starting enhanced training...")
        nnet.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        for epoch in range(TRAINING_EPOCHS):
            epoch_batch_count = len(training_data) // BATCH_SIZE
            
            for _ in range(epoch_batch_count):
                sample_ids = np.random.randint(len(training_data), size=BATCH_SIZE)
                boards, policies, values = list(zip(*[training_data[i] for i in sample_ids]))
                
                # Convert to tensors
                boards = torch.stack(boards).float().to(DEVICE)
                target_policies = torch.FloatTensor(np.array(policies)).to(DEVICE)
                target_values = torch.FloatTensor(np.array(values).astype(np.float64)).to(DEVICE)

                # Predict
                out_log_policies, out_values = nnet(boards)
                
                # Loss calculation with light regularization
                policy_loss = -torch.mean(torch.sum(target_policies * out_log_policies, dim=1))
                value_loss = torch.mean((target_values - out_values.view(-1))**2)
                
                # Light L2 regularization
                l2_reg = sum(p.pow(2.0).sum() for p in nnet.parameters())
                total_loss = policy_loss + value_loss + 5e-6 * l2_reg
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                batch_count += 1
                
                # Backpropagate
                optimizer.zero_grad()
                total_loss.backward()
                
                # Light gradient clipping
                torch.nn.utils.clip_grad_norm_(nnet.parameters(), max_norm=0.8)
                
                optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        if batch_count > 0:
            avg_policy_loss = total_policy_loss / batch_count
            avg_value_loss = total_value_loss / batch_count
            current_lr = scheduler.get_last_lr()[0]
            print(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, LR: {current_lr:.6f}")
        
        print(f"Iteration {i+1} complete. Training data size: {len(training_data)}")
        
        # Save checkpoint every 50 iterations
        if (i + 1) % 50 == 0:
            checkpoint_name = f'better_go_model_{i+1}.pth'
            torch.save(nnet.state_dict(), checkpoint_name)
            print(f"💾 Checkpoint saved: {checkpoint_name}")
        
        print()

    # Save final model
    torch.save(nnet.state_dict(), 'better_go_model.pth')
    print("🎉 BETTER TRAINING COMPLETE! Model saved as 'better_go_model.pth'")
    print("🎮 Try: Update main.py to load 'better_go_model.pth' for improved play!")

if __name__ == "__main__":
    train() 