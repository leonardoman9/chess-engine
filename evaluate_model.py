import chess
import chess.engine
import torch
import numpy as np
from test_model import ChessNet, ChessBoard, ChessAgent
import time

def evaluate_model_against_stockfish(model_path, num_games=10, time_limit=1.0):
    # Load the model
    model = ChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")  # Make sure Stockfish is installed
    
    # Initialize results tracking
    results = []
    total_time = 0
    
    print(f"\nStarting evaluation against Stockfish ({num_games} games)")
    print("Time limit per move:", time_limit, "seconds")
    
    for game in range(num_games):
        print(f"\nGame {game + 1}/{num_games}")
        board = ChessBoard()
        agent = ChessAgent(model)
        
        # Alternate who plays first
        model_plays_white = game % 2 == 0
        
        while not board.is_game_over():
            if (model_plays_white and board.board.turn) or \
               (not model_plays_white and not board.board.turn):
                # Model's turn
                start_time = time.time()
                move = agent.select_move(board, temperature=0.1)
                move_time = time.time() - start_time
                total_time += move_time
                
                if move:
                    print(f"Model plays: {move} (Time: {move_time:.2f}s)")
                    board.make_move(move)
                else:
                    print("Model has no legal moves!")
                    break
            else:
                # Stockfish's turn
                start_time = time.time()
                result = engine.play(board.board, chess.engine.Limit(time=time_limit))
                move_time = time.time() - start_time
                total_time += move_time
                
                if result.move:
                    print(f"Stockfish plays: {result.move.uci()} (Time: {move_time:.2f}s)")
                    board.make_move(result.move.uci())
                else:
                    print("Stockfish has no legal moves!")
                    break
        
        # Record result
        result = board.get_result()
        if model_plays_white:
            results.append(result)
        else:
            results.append(-result)
            
        print(f"Game {game + 1} result: {'Model wins' if results[-1] > 0 else 'Stockfish wins' if results[-1] < 0 else 'Draw'}")
    
    # Calculate statistics
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    draws = sum(1 for r in results if r == 0)
    win_rate = wins / num_games
    avg_time = total_time / (num_games * 2)  # Average time per move
    
    print("\nEvaluation Results:")
    print(f"Games played: {num_games}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average time per move: {avg_time:.2f}s")
    
    # Estimate ELO rating
    # Stockfish's ELO is around 3500, so we can estimate our model's ELO
    # based on the win rate against Stockfish
    stockfish_elo = 3500
    if win_rate > 0:
        # Simple ELO estimation based on win rate
        # This is a rough approximation
        model_elo = stockfish_elo - (1 - win_rate) * 700
        print(f"Estimated ELO rating: {model_elo:.0f}")
    
    engine.quit()
    return results

if __name__ == "__main__":
    model_path = "content/models/checkpoint_10.pt"  # Adjust this to your model's path
    results = evaluate_model_against_stockfish(model_path, num_games=10, time_limit=1.0)