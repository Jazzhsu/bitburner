
import time
from go import GO, StoneType, opposite
from minimax import MinimaxAgent

def main():
    B_SIZE = 5
    go = GO(B_SIZE)
    turn = StoneType.WHITE
    
    print("🎮 Go Game - You are WHITE, AI is BLACK")
    print("🔹 Enter moves as 'i j' (e.g., '2 3')")
    print("🔹 Enter blank line to pass")
    print("=" * 50)

    while True:
        go.print_board()
        print(go.eval())
        
        if turn == StoneType.BLACK:
            print("🤖 AI (BLACK) thinking...")
            
            agent = MinimaxAgent(go, turn)
            
            best_move = None
            max_depth = agent._get_dynamic_depth()
            
            for depth in range(1, max_depth + 1):
                print(f"  ... Searching at depth {depth}")
                start_time = time.time()
                move = agent.get_action(depth)
                end_time = time.time()
                
                if move:
                    best_move = move
                
                print(f"  ... Depth {depth} search took {end_time - start_time:.2f} seconds")

                # If time is running out, use the best move found so far
                if end_time - start_time > 5.0 and depth < max_depth:
                    print("  ... Time limit reached, using best move from current depth")
                    break
            
            if best_move:
                i, j = best_move
                print(f"🤖 (black) plays at ({i}, {j})")
                valid = go.move(i, j, turn)
                if not valid:
                    print("❌ AI made invalid move!")
                    print("🤖 (black) pass")
            else:
                print("🤖 (black) pass")

            turn = opposite(turn)
            continue

        # Human player (WHITE)
        inp = input(f"👤 ({turn.name.lower()}): ") 
        if inp.strip() == "":
            print("👤 (white) pass")
            turn = opposite(turn)
            continue

        try:
            parts = inp.strip().split()
            if len(parts) != 2:
                print("❌ Invalid input. Use 'i j' format (e.g., '2 3')")
                continue
                
            i, j = int(parts[0]), int(parts[1])
        except ValueError:
            print("❌ Invalid numbers. Use 'i j' format (e.g., '2 3')")
            continue

        valid = go.move(i, j, turn)
        if valid:
            turn = opposite(turn)
        else:
            print("❌ INVALID MOVE - try again")

if __name__ == "__main__":
    main() 