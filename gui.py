import chess
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QTextEdit
from PyQt5.QtCore import Qt, QSize, QMimeData
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QDrag

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
        self.setFont(font)

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
        # Find the main window by traversing up the widget hierarchy
        parent = self.parent()
        while parent and not isinstance(parent, ChessGUI):
            parent = parent.parent()
        if parent:
            parent.make_move(chess.Move(from_square, to_square).uci())

class ChessBoardWidget(QWidget):
    def __init__(self, board, human_plays_white, parent=None):
        super().__init__(parent)
        self.board = board
        self.human_plays_white = human_plays_white
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
        self.board = board
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

    def get_square_at(self, pos):
        for square, label in self.squares.items():
            if label.geometry().contains(self.mapFromGlobal(pos)):
                return square
        return None

    def highlight_square(self, square):
        for square, label in self.squares.items():
            if square == square:
                label.setStyleSheet("border: 2px solid red;")
            else:
                label.setStyleSheet("border: 1px solid black;")

class ChessGUI(QMainWindow):
    def __init__(self, model, human_plays_white=True, evaluation_mode=False):
        super().__init__()
        self.model = model
        self.board = chess.Board()
        self.human_plays_white = human_plays_white
        self.evaluation_mode = evaluation_mode
        
        self.init_ui()
        self.update_board()
        
        # If human plays black and not in evaluation mode, make model's first move
        if not human_plays_white and not evaluation_mode:
            self.make_model_move()

    def init_ui(self):
        self.setWindowTitle('Chess Game')
        self.setFixedSize(800, 600)  # Increased width to accommodate move history

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create player labels
        player_layout = QHBoxLayout()
        self.white_label = QLabel("White: Model" if self.human_plays_white else "White: Stockfish")
        self.black_label = QLabel("Black: Stockfish" if self.human_plays_white else "Black: Model")
        self.white_label.setAlignment(Qt.AlignCenter)
        self.black_label.setAlignment(Qt.AlignCenter)
        player_layout.addWidget(self.white_label)
        player_layout.addWidget(self.black_label)
        layout.addLayout(player_layout)

        # Create horizontal layout for board and move history
        board_history_layout = QHBoxLayout()

        # Create chess board widget
        self.board_widget = ChessBoardWidget(self.board, self.human_plays_white)
        board_history_layout.addWidget(self.board_widget)

        # Create move history widget
        move_history_widget = QWidget()
        move_history_layout = QVBoxLayout(move_history_widget)
        
        # Add move history label
        history_label = QLabel("Move History")
        history_label.setAlignment(Qt.AlignCenter)
        move_history_layout.addWidget(history_label)
        
        # Add move history display using QTextEdit
        self.move_history = QTextEdit()
        self.move_history.setReadOnly(True)
        self.move_history.setFixedWidth(250)  # Set fixed width
        self.move_history.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid black;
                font-size: 14px;
                padding: 10px;
            }
        """)
        move_history_layout.addWidget(self.move_history)
        
        board_history_layout.addWidget(move_history_widget)
        layout.addLayout(board_history_layout)

        # Create status label
        self.status_label = QLabel(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Create new game button
        self.new_game_button = QPushButton("New Game")
        self.new_game_button.clicked.connect(self.new_game)
        layout.addWidget(self.new_game_button)
        
        # Disable interaction in evaluation mode
        if self.evaluation_mode:
            self.new_game_button.setEnabled(False)
            self.board_widget.setEnabled(False)

    def update_board(self, board=None):
        """Update the board display with a new position"""
        if board is not None:
            self.board = board
        self.board_widget.update_board(self.board)
        
        # Update move history
        self.update_move_history()
        
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                QMessageBox.information(self, "Game Over", "White wins!")
            elif result == "0-1":
                QMessageBox.information(self, "Game Over", "Black wins!")
            else:
                QMessageBox.information(self, "Game Over", "Draw!")

    def update_move_history(self):
        """Update the move history display with algebraic notation"""
        history = []
        temp_board = chess.Board()  # Create a temporary board to get correct SAN
        
        for i, move in enumerate(self.board.move_stack):
            try:
                if i % 2 == 0:  # White's move
                    move_text = f"{i//2 + 1}. {temp_board.san(move)}"
                    history.append(move_text)
                    print(move_text, end=" ")  # Print white's move
                else:  # Black's move
                    move_text = temp_board.san(move)
                    history.append(f"{move_text}\n")
                    print(move_text)  # Print black's move and newline
                temp_board.push(move)
            except AssertionError:
                # If SAN conversion fails, use UCI notation
                if i % 2 == 0:  # White's move
                    move_text = f"{i//2 + 1}. {move.uci()}"
                    history.append(move_text)
                    print(move_text, end=" ")  # Print white's move
                else:  # Black's move
                    move_text = move.uci()
                    history.append(f"{move_text}\n")
                    print(move_text)  # Print black's move and newline
                temp_board.push(move)
        
        # Format the history text
        history_text = " ".join(history)
        self.move_history.setText(history_text)
        # Scroll to bottom
        self.move_history.verticalScrollBar().setValue(
            self.move_history.verticalScrollBar().maximum()
        )

    def update_status(self, message):
        """Update the status label with a message"""
        self.status_label.setText(message)

    def show_game_over(self, message):
        """Show game over dialog"""
        QMessageBox.information(self, "Game Over", message)

    def make_model_move(self):
        # Model's turn
        self.status_label.setText("Model's turn")
        QApplication.processEvents()  # Update UI
        
        # Get model's move
        state = self.get_board_state()
        move = self.select_move(state)
        
        if move:
            self.board.push(chess.Move.from_uci(move))
            self.status_label.setText(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
            self.update_board()

    def make_move(self, move_uci):
        if self.evaluation_mode or self.board.is_game_over():
            return False
            
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                if not self.board.is_game_over():
                    self.make_model_move()
                return True
        except ValueError:
            pass
        return False

    def new_game(self):
        if self.evaluation_mode:
            return
        self.board = chess.Board()
        self.update_board()
        self.status_label.setText(f"Your turn ({'White' if self.human_plays_white else 'Black'})")
        if not self.human_plays_white:
            self.make_model_move()

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

    def select_move(self, state):
        """Select the best move using the model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value, policy = self.model(state_tensor)
        
        # Get legal moves
        legal_moves = [move.uci() for move in self.board.legal_moves]
        
        if not legal_moves:
            return None
            
        # Get move probabilities
        move_probs = torch.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = self._move_to_index(move)
            if move_idx < policy.shape[1]:
                move_probs[i] = policy[0][move_idx]
        
        move_probs = torch.softmax(move_probs, dim=0)
        move_idx = torch.argmax(move_probs).item()
        return legal_moves[move_idx]

    def _move_to_index(self, move_uci):
        """Convert UCI move to policy index"""
        try:
            from_square = chess.parse_square(move_uci[:2])
            to_square = chess.parse_square(move_uci[2:4])
            return from_square * 64 + to_square
        except ValueError:
            return 0 