import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import sys
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

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def get_board_state(self):
        """Convert board to neural network input format"""
        state = np.zeros((13, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                state[12][square // 8][square % 8] = 1
            else:
                piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)
                state[piece_idx][square // 8][square % 8] = 1
        return state

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def make_move(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except ValueError:
            return False

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        if not self.is_game_over():
            return 0.0
        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

class ChessAgent:
    def __init__(self, model):
        self.model = model
        self.board = ChessBoard()

    def select_move(self, temperature=1.0):
        state = self.board.get_board_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            value, policy = self.model(state_tensor)

        policy = policy / temperature
        legal_moves = self.board.get_legal_moves()
        
        if not legal_moves:
            return None

        move_probs = torch.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = self._move_to_index(move)
            if move_idx < policy.shape[1]:
                move_probs[i] = policy[0][move_idx]

        move_probs = F.softmax(move_probs, dim=0)
        move_idx = torch.multinomial(move_probs, 1).item()
        return legal_moves[move_idx]

    def _move_to_index(self, move_uci):
        try:
            from_square = chess.parse_square(move_uci[:2])
            to_square = chess.parse_square(move_uci[2:4])
            return from_square * 64 + to_square
        except ValueError:
            return 0

def print_board(board):
    print(board)
    print()

def train_model(model_path=None, num_games=1000, temperature=1.0):
    # Initialize model
    model = ChessNet()
    if model_path and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for game in range(num_games):
    agent = ChessAgent(model)
        board = agent.board
        game_states = []
        game_policies = []
        
        # Play a game
    while not board.is_game_over():
            state = board.get_board_state()
            move = agent.select_move(temperature)
            if move is None:
                        break
                
            game_states.append(state)
            policy = torch.zeros(4672)
            move_idx = agent._move_to_index(move)
            policy[move_idx] = 1
            game_policies.append(policy)
            
            board.make_move(move)
    
        # Get game result
    result = board.get_result()
        
        # Convert states and policies to tensors
        states = torch.FloatTensor(np.array(game_states))
        policies = torch.stack(game_policies)
        values = torch.full((len(game_states),), result)
        
        # Compute loss and update model
        optimizer.zero_grad()
        predicted_values, predicted_policies = model(states)
        value_loss = F.mse_loss(predicted_values.squeeze(), values)
        policy_loss = -torch.mean(torch.sum(policies * predicted_policies, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        
        if (game + 1) % 100 == 0:
            print(f"Game {game + 1}/{num_games}, Loss: {loss.item():.4f}")
            # Save model checkpoint
            torch.save(model.state_dict(), f"models/checkpoint_{game + 1}.pt")

def main():
    if len(sys.argv) < 2:
        print("Usage: python training.py <num_games> [model_path]")
        sys.exit(1)

    num_games = int(sys.argv[1])
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    train_model(model_path, num_games)

if __name__ == "__main__":
    main()