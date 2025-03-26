import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QSize, QMimeData
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QDrag

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

class ChessSquare(QLabel):
    def __init__(self, square, parent=None):
        super().__init__(parent)
        self.square = square
        self.setFixedSize(60, 60)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid black;")
        self.setAcceptDrops(True)
        self.piece = None
        self.is_selected = False
        font = self.font()
        font.setPointSize(40)
        self.setFont(font)  # Make pieces bigger

    def set_piece(self, piece):
        self.piece = piece
        if piece:
            self.setText(piece)
        else:
            self.setText("")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.piece:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(f"{self.square}")
            drag.setMimeData(mime)
            drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        from_square = int(event.mimeData().text())
        to_square = self.square
        # Find the main window (ChessGUI) by traversing up the widget hierarchy
        parent = self.parent()
        while parent and not isinstance(parent, ChessGUI):
            parent = parent.parent()
        if parent:
            parent.make_move(chess.Move(from_square, to_square).uci())

class ChessBoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.squares = {}
        self.selected_square = None
        self.init_board()
        self.setFixedSize(480, 480)

    def init_board(self):
        layout = QVBoxLayout()
        for row in range(8):
            row_layout = QHBoxLayout()
            for col in range(8):
                square = ChessSquare(chess.square(col, 7-row))
                square.setStyleSheet(f"background-color: {'#b58863' if (row + col) % 2 == 0 else '#f0d9b5'}")
                self.squares[chess.square(col, 7-row)] = square
                row_layout.addWidget(square)
            layout.addLayout(row_layout)
        self.setLayout(layout)

    def update_board(self, board):
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                self.squares[square].set_piece(self.get_piece_symbol(piece))
            else:
                self.squares[square].set_piece(None)

    def get_piece_symbol(self, piece):
        symbols = {
            'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
            'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
        }
        return symbols.get(piece.symbol(), '')

class ChessGUI(QMainWindow):
    def __init__(self, model_path, human_plays_white=True):
        super().__init__()
        self.model = ChessNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.board = ChessBoard()
        self.agent = ChessAgent(self.model)
        self.human_plays_white = human_plays_white
        
        self.init_ui()
        self.update_board()
        
        # If human plays black, make model's first move
        if not human_plays_white:
            self.make_model_move()

    def init_ui(self):
        self.setWindowTitle('Chess Game')
        self.setFixedSize(600, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create chess board widget
        self.board_widget = ChessBoardWidget()
        layout.addWidget(self.board_widget)

        # Create status label
        self.status_label = QLabel(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Create new game button
        new_game_button = QPushButton("New Game")
        new_game_button.clicked.connect(self.new_game)
        layout.addWidget(new_game_button)

    def update_board(self):
        self.board_widget.update_board(self.board.board)
        if self.board.is_game_over():
            result = self.board.get_result()
            if result == 1:
                QMessageBox.information(self, "Game Over", "White wins!")
            elif result == -1:
                QMessageBox.information(self, "Game Over", "Black wins!")
            else:
                QMessageBox.information(self, "Game Over", "Draw!")

    def make_model_move(self):
        # Update agent's board state before making its move
        self.agent.board.board = self.board.board.copy()
        move = self.agent.select_move(temperature=0.1)
        if move:
            self.board.make_move(move)
            self.status_label.setText(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
            self.update_board()

    def make_move(self, move_uci):
        if self.board.make_move(move_uci):
            self.update_board()
            if not self.board.is_game_over():
                # Model's turn
                self.status_label.setText("Model's turn")
                QApplication.processEvents()  # Update UI
                self.make_model_move()
            return True
        return False

    def new_game(self):
        self.board = ChessBoard()
        self.agent.board = ChessBoard()  # Reset agent's board too
        self.update_board()
        self.status_label.setText(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
        # If human plays black, make model's first move
        if not self.human_plays_white:
            self.make_model_move()

def print_board(board):
    print("\n")
    print(board.board)

def play_terminal_mode(model_path, human_plays_white=True):
    # Load the model
    model = ChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize game
    board = ChessBoard()
    agent = ChessAgent(model)
    
    print("\nStarting new game!")
    print("You are playing as", "White" if human_plays_white else "Black")
    
    while not board.is_game_over():
        print_board(board)
        
        if (human_plays_white and board.board.turn) or \
           (not human_plays_white and not board.board.turn):
            # Human's turn
            while True:
                try:
                    move = input("\nEnter your move (UCI format, e.g., 'e2e4'): ")
                    if board.make_move(move):
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Invalid move format! Use UCI format (e.g., 'e2e4')")
        else:
            # Model's turn
            # Update agent's board state
            agent.board.board = board.board.copy()
            move = agent.select_move(temperature=0.1)  # Lower temperature for more deterministic play
            print(f"\nModel plays: {move}")
            board.make_move(move)
    
    # Game over
    print_board(board)
    result = board.get_result()
    if result == 1:
        print("\nWhite wins!")
    elif result == -1:
        print("\nBlack wins!")
    else:
        print("\nDraw!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Play chess against the trained model')
    parser.add_argument('--mode', choices=['gui', 'terminal'], default='gui',
                      help='Choose the interface mode (default: gui)')
    parser.add_argument('--model', default='content/models/checkpoint_10.pt',
                      help='Path to the model checkpoint (default: content/models/checkpoint_10.pt)')
    parser.add_argument('--color', choices=['white', 'black'], default='white',
                      help='Choose your color (default: white)')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        app = QApplication(sys.argv)
        gui = ChessGUI(args.model, human_plays_white=(args.color == 'white'))
        gui.show()
        sys.exit(app.exec_())
    else:
        human_plays_white = args.color == 'white'
        play_terminal_mode(args.model, human_plays_white)

if __name__ == "__main__":
    main()