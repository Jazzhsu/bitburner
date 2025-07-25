import torch
import numpy as np

from go import GO, StoneType, opposite
from model import GoNNet
from MCTS import MCTS, GOSimulator

def main():
    B_SIZE = 5
    go = GO(B_SIZE)
    turn = StoneType.WHITE
    
    # Load the clean model (no pass bias)
    state_dict = torch.load("clean_go_model.pth", weights_only=False)
    model = GoNNet(B_SIZE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("🎮 Go Game - You are WHITE, AI is BLACK")
    print("🔹 Enter moves as 'i j' (e.g., '2 3')")
    print("🔹 Enter blank line to pass")
    print("=" * 50)

    while True:
        go.print_board()
        print(go.eval())
        
        if turn == StoneType.BLACK:
            print("🤖 AI thinking...")
            
            # 🔧 FIXED: Use MCTS for proper AI play (not raw network!)
            go_sim = GOSimulator(go=go, turn=turn)
            # 🚀 IMPROVED: Better MCTS parameters for stronger play
            mcts = MCTS(model, sims=100, cpuct=1.4, device='cpu')  # More simulations, better exploration
            policy = mcts.get_action_probs(go_sim, temp=0.5)  # Higher temp for less greedy play
            
            # Choose action based on MCTS policy
            action = np.random.choice(len(policy), p=policy)

            if action == B_SIZE ** 2:
                print("🤖 (black) pass")
            else:
                i, j = action // B_SIZE, action % B_SIZE
                print(f"🤖 (black) plays at ({i}, {j})")

                valid = go.move(i, j, turn)
                if not valid:
                    print("❌ AI made invalid move, passing instead")
                    turn = opposite(turn)
                    continue

            turn = opposite(turn)
            continue

        # Human player
        inp = input(f"👤 ({turn.name.lower()}): ") 
        if inp.strip() == "":
            print("👤 (white) pass")
            turn = opposite(turn)
            continue

        try:
            parts = inp.split()
            if len(parts) != 2:
                print("❌ Use format 'i j' (e.g., '2 3')")
                continue
            i, j = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print("❌ Invalid input. Use 'i j' format")
            continue

        valid = go.move(i, j, turn)
        if valid:
            turn = opposite(turn)
        else:
            print("❌ INVALID MOVE")


if __name__ == "__main__":
    main()
