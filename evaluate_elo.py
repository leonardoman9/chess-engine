import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import sys
from PyQt5.QtWidgets import QApplication
from gui import ChessGUI
import os

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_value = nn.Linear(512, 1)
        self.fc_policy = nn.Linear(512, 4672)  # All possible moves

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = torch.tanh(self.fc_value(x))
        policy = F.log_softmax(self.fc_policy(x), dim=1)

        return value, policy

class EloEvaluator:
    def __init__(self, model_path, stockfish_path, time_control=1.0, num_games=10, stockfish_elo=3000, save_pgn=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Configure Stockfish based on ELO
        if stockfish_elo < 1320:
            # For ratings below 1320, use Skill Level instead of UCI_Elo
            skill_level = max(0, min(20, int((stockfish_elo - 100) / 61)))  # Map 100-1320 to 0-20
            self.engine.configure({"UCI_LimitStrength": False, "Skill Level": skill_level})
            print(f"Using Skill Level {skill_level} (approximately {stockfish_elo} ELO)")
        else:
            # For ratings 1320 and above, use UCI_Elo
            stockfish_elo = max(1320, min(3190, stockfish_elo))
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
            print(f"Using UCI_Elo {stockfish_elo}")
        
        self.time_control = time_control
        self.num_games = num_games
        self.stockfish_elo = stockfish_elo
        self.model_elo = 1000  # Start from 1000 ELO
        self.save_pgn = save_pgn
        self.results = []  # Initialize results list
        self.use_gui = False  # Initialize use_gui flag
        self.model_path = model_path  # Store model path for saving results
        
        # Create directory for PGN files if saving is enabled
        if save_pgn:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.pgn_dir = Path(f"data/eval_{date_str}_{stockfish_elo}_{int(self.model_elo)}")
            self.pgn_dir.mkdir(parents=True, exist_ok=True)
            print(f"PGN files will be saved in: {self.pgn_dir}")

    def calculate_elo(self, wins, losses, draws, k_factor=32):
        """Calculate ELO rating using the standard formula"""
        total_games = wins + losses + draws
        if total_games == 0:
            return self.model_elo
        
        # Calculate actual score (1 for win, 0.5 for draw, 0 for loss)
        actual_score = (wins + 0.5 * draws) / total_games
        
        # Calculate expected score using the ELO formula
        # Note: We use (model_elo - stockfish_elo) because we want the model's expected score
        expected_score = 1 / (1 + 10 ** ((self.model_elo - self.stockfish_elo) / 400))
        
        # Calculate new ELO
        elo_change = k_factor * (actual_score - expected_score)
        self.model_elo += elo_change
        
        return self.model_elo

    def play_game(self, model_plays_white):
        """Play a single game against Stockfish"""
        board = chess.Board()
        game = chess.pgn.Game()
        
        # Set up game headers
        game.headers["Event"] = "Model vs Stockfish Evaluation"
        game.headers["Site"] = "Local Evaluation"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(len(self.results) + 1)
        game.headers["White"] = "Model" if model_plays_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if model_plays_white else "Model"
        game.headers["Model_ELO"] = str(int(self.model_elo))
        game.headers["Stockfish_ELO"] = str(self.stockfish_elo)
        game.headers["Stockfish_Strength"] = f"Skill Level {int((self.stockfish_elo - 100) / 61)}" if self.stockfish_elo < 1320 else f"UCI_Elo {self.stockfish_elo}"
        
        game_result = None
        move_number = 1
        node = game
        
        print("\nStarting new game!")
        print(f"Model plays as {'White' if model_plays_white else 'Black'}")
        print("\nInitial position:")
        print(board)
        
        if self.use_gui:
            # Update the GUI labels to match who is actually playing
            if model_plays_white:
                self.gui.white_label.setText("White: Model")
                self.gui.black_label.setText("Black: Stockfish")
            else:
                self.gui.white_label.setText("White: Stockfish")
                self.gui.black_label.setText("Black: Model")
            self.gui.update_board(board)
        
        while not board.is_game_over():
            if (model_plays_white and board.turn) or (not model_plays_white and not board.turn):
                # Model's turn
                state = self.get_board_state(board)
                move = self.select_move(state, board)
                board.push(chess.Move.from_uci(move))
                node = node.add_variation(chess.Move.from_uci(move))
                print(f"\nMove {move_number}: Model plays {move}")
            else:
                # Stockfish's turn
                result = self.engine.play(board, chess.engine.Limit(time=self.time_control))
                board.push(result.move)
                node = node.add_variation(result.move)
                print(f"\nMove {move_number}: Stockfish plays {result.move.uci()}")
            
            print("\nCurrent position:")
            print(board)
            
            if self.use_gui:
                self.gui.update_board(board)
                self.gui.update_status(f"Game {len(self.results) + 1}/{self.num_games} - Move {move_number}")
                self.app.processEvents()  # Process GUI events
            
            move_number += 1
        
        # Determine game result
        if board.is_checkmate():
            # If it's checkmate and it's White's turn, Black won
            # If it's checkmate and it's Black's turn, White won
            game_result = -1 if board.turn else 1
            game.headers["Result"] = "0-1" if board.turn else "1-0"
        else:
            game_result = 0
            game.headers["Result"] = "1/2-1/2"
            
        # Adjust result based on which color the model played
        if not model_plays_white:
            game_result = -game_result
            
        # Print game result
        if game_result == 1:
            print("\nGame Over: Model wins!")
            if self.use_gui:
                self.gui.show_game_over("Model wins!")
        elif game_result == -1:
            print("\nGame Over: Stockfish wins!")
            if self.use_gui:
                self.gui.show_game_over("Stockfish wins!")
        else:
            print("\nGame Over: Draw!")
            if self.use_gui:
                self.gui.show_game_over("Draw!")
        
        # Save PGN if enabled
        if self.save_pgn:
            pgn_file = self.pgn_dir / f"game_{len(self.results) + 1}.pgn"
            with open(pgn_file, "w") as f:
                print(game, file=f, end="\n\n")
            
        return game_result

    def get_board_state(self, board):
        """Convert board to neural network input format"""
        state = np.zeros((13, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                state[12][square // 8][square % 8] = 1
            else:
                piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)
                state[piece_idx][square // 8][square % 8] = 1
        return state

    def select_move(self, state, board):
        """Select the best move using the model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy = self.model(state_tensor)
        
        # Get legal moves from the current board state
        legal_moves = [move.uci() for move in board.legal_moves]
        
        if not legal_moves:
            return None
            
        # Get move probabilities
        move_probs = torch.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = self._move_to_index(move)
            if move_idx < policy.shape[1]:
                move_probs[i] = policy[0][move_idx]
        
        # Apply softmax to get probabilities
        move_probs = torch.softmax(move_probs, dim=0)
        
        # Select move with highest probability
        move_idx = torch.argmax(move_probs).item()
        selected_move = legal_moves[move_idx]
        
        # Verify the move is legal before returning
        try:
            move = chess.Move.from_uci(selected_move)
            if move in board.legal_moves:
                return selected_move
            else:
                # If move is illegal, select a random legal move
                return legal_moves[np.random.randint(len(legal_moves))]
        except ValueError:
            # If move is invalid, select a random legal move
            return legal_moves[np.random.randint(len(legal_moves))]

    def _move_to_index(self, move_uci):
        """Convert UCI move to policy index"""
        try:
            from_square = chess.parse_square(move_uci[:2])
            to_square = chess.parse_square(move_uci[2:4])
            return from_square * 64 + to_square
        except ValueError:
            return 0

    def evaluate(self):
        """Run the evaluation"""
        print(f"Starting ELO evaluation against Stockfish (ELO: {self.stockfish_elo})")
        print(f"Number of games: {self.num_games}")
        print(f"Time control: {self.time_control} seconds per move")
        
        wins = 0
        losses = 0
        draws = 0
        
        for game in tqdm(range(self.num_games)):
            print(f"\n{'='*50}")
            print(f"Game {game + 1} of {self.num_games}")
            print(f"{'='*50}")
            
            # Alternate between playing as White and Black
            model_plays_white = game % 2 == 0
            result = self.play_game(model_plays_white)
            
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
                
            self.results.append({
                'game': game + 1,
                'model_plays_white': model_plays_white,
                'result': result
            })
            
            # Calculate current ELO after each game
            current_elo = self.calculate_elo(wins, losses, draws)
            
            # Print progress
            print(f"\nProgress after {game + 1} games:")
            print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"Current ELO: {current_elo:.1f}")
            
            if self.use_gui:
                self.gui.update_status(f"Game {game + 1}/{self.num_games} - ELO: {current_elo:.1f}")
                self.app.processEvents()
            
            # Ask if user wants to continue
            if game < self.num_games - 1:
                input("\nPress Enter to continue to next game...")
        
        final_elo = self.calculate_elo(wins, losses, draws)
        
        print("\nEvaluation completed!")
        print(f"Final results:")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Final ELO: {final_elo:.1f}")
        
        if self.use_gui:
            self.gui.update_status(f"Evaluation completed! Final ELO: {final_elo:.1f}")
            self.app.processEvents()
        
        return final_elo

    def evaluate_with_gui(self):
        """Run the evaluation with GUI visualization"""
        self.use_gui = True
        self.app = QApplication(sys.argv)
        self.gui = ChessGUI(self.model, human_plays_white=False, evaluation_mode=True)
        self.gui.show()
        
        # Run the evaluation
        final_elo = self.evaluate()
        
        # Keep the GUI window open until closed
        self.app.exec_()

def main():
    parser = argparse.ArgumentParser(description='Evaluate chess model against Stockfish')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--stockfish', type=str, required=True, help='Path to Stockfish executable')
    parser.add_argument('--time', type=float, default=1.0, help='Time control in seconds per move')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--stockfish-elo', type=int, default=3000, help='Stockfish ELO rating (100-3190)')
    parser.add_argument('--gui', action='store_true', help='Show GUI during evaluation')
    parser.add_argument('--save', action='store_true', help='Save games as PGN files')
    
    args = parser.parse_args()
    
    evaluator = EloEvaluator(
        args.model,
        args.stockfish,
        time_control=args.time,
        num_games=args.games,
        stockfish_elo=args.stockfish_elo,
        save_pgn=args.save
    )
    
    if args.gui:
        evaluator.evaluate_with_gui()
    else:
        evaluator.evaluate()

if __name__ == "__main__":
    main() 