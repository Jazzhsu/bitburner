from collections import deque
import multiprocessing as mp

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from go import GO, StoneType, opposite
from MCTS import MCTS, GOSimulator
from model import GoNNet

BOARD_SIZE = 5
NUM_ITERATIONS = 100
NUM_SELF_PLAY_GAMES = 50
MCTS_SIMULATIONS = 50
TRAINING_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
CPUCT = 2.4 # controls explore-exploit ratio factor. higher = more explore
DEVICE = "mps"

def self_play_worker(nnet_state_dict):
    """
    A worker function that plays one full game of self-play.
    It creates its own instances of the game and network.
    """
    # Each worker needs its own instance of the game and network
    nnet = GoNNet(BOARD_SIZE)
    nnet.load_state_dict(nnet_state_dict)
    
    # Run a single episode
    episode_examples = execute_episode(nnet)
    return episode_examples

def execute_episode(nnet):
    """
    Executes one full game of self-play, returning training examples.
    An episode is a list of (board, policy, value) tuples.
    """
    train_examples = []
    go = GOSimulator(go=GO(BOARD_SIZE), turn=StoneType.WHITE)

    move_counter = 0
    consecutive_passes = 0  # Track consecutive passes separately

    while True:

        mcts = MCTS(nnet, MCTS_SIMULATIONS, CPUCT, device='cpu')

        # Add exploration noise early in the game
        add_noise = move_counter < 10  # Add noise for first 10 moves
        policy = mcts.get_action_probs(go, temp=1, add_noise=add_noise)
        
        # FIXED: Modify policy to discourage early passing BEFORE saving to training data
        training_policy = policy.copy()
        if move_counter < 5:
            # Check if there are valid board moves
            valid_board_moves = [a for a in range(go.get_action_size() - 1) if go.is_valid_move(a)]
            if len(valid_board_moves) > 0:
                # Reduce pass probability significantly in early game
                training_policy[-1] *= 0.1  # Reduce pass probability by 90%
                # Renormalize the policy
                total_prob = sum(training_policy)
                if total_prob > 0:
                    training_policy = [p / total_prob for p in training_policy]
        
        # Save the MODIFIED policy for training (consistent with actual play)
        train_examples.append([go.torch_state(), go.player(), training_policy])
        
        # Choose an action based on the MODIFIED policy
        action = np.random.choice(len(training_policy), p=training_policy)
        
        # Track consecutive passes
        if action == go.get_action_size() - 1:  # Pass move
            consecutive_passes += 1
        else:
            consecutive_passes = 0
        
        # Apply the action
        go.move(action)

        move_counter += 1
        
        # Check for game end - require at least 5 moves before allowing termination by passing
        game_should_end = (go.is_terminal() and move_counter >= 5) or move_counter >= (BOARD_SIZE ** 2) * 1.5
        
        if game_should_end:
            # Assign the final game result to all examples from this game
            eval = go.eval()
            examples = []
            
            for board_state, player, policy in train_examples:
                value = 0
                if go.is_terminal():
                    norm = eval[StoneType.EMPTY] / 2.
                    if norm > 0:  # Prevent division by zero
                        value = (eval[player] - norm) / norm
                    else:
                        # If no empty spaces, just use stone difference
                        total_stones = eval[StoneType.WHITE] + eval[StoneType.BLACK]
                        if total_stones > 0:
                            value = (eval[player] / total_stones) * 2 - 1
                        else:
                            value = 0
                
                # Penalize very short games to encourage longer, more meaningful games
                if move_counter < 8:
                    value *= 0.5  # Reduce the magnitude of the reward for short games
                
                examples.append((board_state, policy, value))

            return examples

def train():
    """Main training loop."""
    nnet = GoNNet(BOARD_SIZE).to(DEVICE)
    
    # FORCE fresh start - don't load any existing biased model
    print("🔥 Starting fresh training (no existing model loaded)")
    
    optimizer = optim.Adam(nnet.parameters(), lr=LEARNING_RATE)
    
    # Store examples from recent iterations
    training_data = deque(maxlen=20 * NUM_SELF_PLAY_GAMES)

    for i in range(NUM_ITERATIONS):
        print(f"--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        
        # 1. Self-Play Phase
        print("Starting self-play phase...")
        nnet.eval()  # Set network to evaluation mode for self-play

        # Move network to CPU and get its state for workers
        cpu_nnet_state_dict = nnet.to('cpu').state_dict()
        
        # Create a pool of worker processes
        # Use slightly fewer cores than available to leave resources for the OS

        num_cores = max(1, 8)
        with mp.Pool(num_cores) as pool:
            # Each worker will call self_play_worker with a copy of the network state
            tasks = [cpu_nnet_state_dict] * NUM_SELF_PLAY_GAMES
            for result in tqdm(pool.imap_unordered(self_play_worker, tasks), total=len(tasks)):
                training_data.extend(result)

        # for k in range(50):
        #     training_data.extend(self_play_worker(cpu_nnet_state_dict))

        nnet.to(DEVICE)
        
        # 2. Training Phase
        print("Starting training phase...")
        nnet.train() # Set network to training mode
        
        # Track some statistics
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        for epoch in range(TRAINING_EPOCHS):
            epoch_batch_count = len(training_data) // BATCH_SIZE
            
            for _ in range(epoch_batch_count):
                # Sample a random batch from the training data
                sample_ids = np.random.randint(len(training_data), size=BATCH_SIZE)
                boards, policies, values = list(zip(*[training_data[i] for i in sample_ids]))
                
                # Convert to tensors
                boards = torch.stack(boards).float().to(DEVICE)
                target_policies = torch.FloatTensor(np.array(policies)).to(DEVICE)
                target_values = torch.FloatTensor(np.array(values).astype(np.float64)).to(DEVICE)

                # Predict
                out_log_policies, out_values = nnet(boards)
                
                # Calculate loss
                policy_loss = -torch.mean(torch.sum(target_policies * out_log_policies, dim=1))
                value_loss = torch.mean((target_values - out_values.view(-1))**2)
                total_loss = policy_loss + value_loss
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                batch_count += 1
                
                # Backpropagate and update weights
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        # Print training statistics
        if batch_count > 0:
            avg_policy_loss = total_policy_loss / batch_count
            avg_value_loss = total_value_loss / batch_count
            print(f"Average Policy Loss: {avg_policy_loss:.4f}, Average Value Loss: {avg_value_loss:.4f}")
        
        print(f"Iteration {i+1} complete. Training data size: {len(training_data)}\n")

    # Save the trained model
    torch.save(nnet.state_dict(), 'best_go_model.pth')
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
