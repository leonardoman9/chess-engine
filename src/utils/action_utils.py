"""
Action utilities for chess DQN
Handles conversion between UCI moves and action indices, plus action masking
"""

import chess
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class ChessActionSpace:
    """
    Manages the action space for chess DQN
    Converts between UCI moves and action indices with efficient action masking
    """
    
    def __init__(self, action_size: int = 4672):
        """
        Initialize chess action space
        
        Args:
            action_size: Total number of possible actions (default: 64*64 + special moves)
        """
        self.action_size = action_size
        
        # Create mappings between UCI moves and action indices
        self._create_action_mappings()
    
    def _create_action_mappings(self):
        """Create bidirectional mappings between UCI moves and indices"""
        self.move_to_index = {}
        self.index_to_move = {}
        
        index = 0
        
        # Standard moves: from_square * 64 + to_square
        for from_square in range(64):
            for to_square in range(64):
                if from_square != to_square:  # No null moves
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
                    self.move_to_index[move_uci] = index
                    self.index_to_move[index] = move_uci
                    index += 1
        
        # Promotion moves: from_square + to_square + promotion_piece
        promotion_pieces = ['q', 'r', 'b', 'n']  # queen, rook, bishop, knight
        
        for from_square in range(48, 56):  # White pawn promotion (rank 7 to 8)
            for to_square in range(56, 64):  # Rank 8
                if abs((from_square % 8) - (to_square % 8)) <= 1:  # Valid pawn move
                    for piece in promotion_pieces:
                        move_uci = chess.square_name(from_square) + chess.square_name(to_square) + piece
                        if index < self.action_size:
                            self.move_to_index[move_uci] = index
                            self.index_to_move[index] = move_uci
                            index += 1
        
        for from_square in range(8, 16):  # Black pawn promotion (rank 2 to 1)
            for to_square in range(0, 8):  # Rank 1
                if abs((from_square % 8) - (to_square % 8)) <= 1:  # Valid pawn move
                    for piece in promotion_pieces:
                        move_uci = chess.square_name(from_square) + chess.square_name(to_square) + piece
                        if index < self.action_size:
                            self.move_to_index[move_uci] = index
                            self.index_to_move[index] = move_uci
                            index += 1
        
        print(f"Created action mappings for {index} moves out of {self.action_size} possible actions")
    
    def move_to_action(self, move: chess.Move) -> int:
        """
        Convert chess move to action index
        
        Args:
            move: Chess move object
            
        Returns:
            Action index (int)
        """
        move_uci = move.uci()
        return self.move_to_index.get(move_uci, 0)  # Default to 0 if not found
    
    def action_to_move(self, action: int) -> Optional[str]:
        """
        Convert action index to UCI move string
        
        Args:
            action: Action index
            
        Returns:
            UCI move string or None if invalid
        """
        return self.index_to_move.get(action, None)
    
    def get_legal_actions_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Create a boolean mask for legal actions
        
        Args:
            board: Current chess board state
            
        Returns:
            Boolean tensor [action_size] where True = legal action
        """
        mask = torch.zeros(self.action_size, dtype=torch.bool)
        
        for move in board.legal_moves:
            action_idx = self.move_to_action(move)
            if action_idx < self.action_size:
                mask[action_idx] = True
        
        return mask
    
    def get_legal_actions(self, board: chess.Board) -> List[int]:
        """
        Get list of legal action indices
        
        Args:
            board: Current chess board state
            
        Returns:
            List of legal action indices
        """
        legal_actions = []
        for move in board.legal_moves:
            action_idx = self.move_to_action(move)
            if action_idx < self.action_size:
                legal_actions.append(action_idx)
        
        return legal_actions
    
    def sample_random_legal_action(self, board: chess.Board) -> int:
        """
        Sample a random legal action
        
        Args:
            board: Current chess board state
            
        Returns:
            Random legal action index
        """
        legal_actions = self.get_legal_actions(board)
        if not legal_actions:
            return 0  # Fallback
        return np.random.choice(legal_actions)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert chess board to neural network input tensor
    
    Args:
        board: Chess board state
    
    Returns:
        Tensor of shape [15, 8, 8] representing the board
    """
    # Initialize tensor: 12 pieces + 1 metadata + 2 control channels
    tensor = torch.zeros(15, 8, 8, dtype=torch.float32)
    
    # Piece type mapping
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1, 
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Fill piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = square // 8
            col = square % 8
            
            # Get channel based on piece type and color
            channel = piece_to_channel[piece.piece_type]
            if not piece.color:  # Black pieces
                channel += 6
                
            tensor[channel, row, col] = 1.0
    
    # Metadata channel (whose turn, castling rights, etc.)
    if board.turn:  # White to move
        tensor[12, :, :] = 1.0
    
    # Additional metadata can be encoded in specific positions
    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[12, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[12, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[12, 7, 0] = 1.0

    # Control maps: squares attacked by white (channel 13) and black (channel 14)
    white_control = tensor[13]
    black_control = tensor[14]
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_control[square // 8, square % 8] = 1.0
        if board.is_attacked_by(chess.BLACK, square):
            black_control[square // 8, square % 8] = 1.0
    
    return tensor


def augment_board_tensor(tensor: torch.Tensor, augmentation: str) -> torch.Tensor:
    """
    Apply data augmentation to board tensor
    
    Args:
        tensor: Input board tensor [13, 8, 8]
        augmentation: Type of augmentation ('flip_h', 'flip_v', 'rotate_90', etc.)
        
    Returns:
        Augmented tensor
    """
    if augmentation == 'flip_h':
        return torch.flip(tensor, dims=[2])  # Horizontal flip
    elif augmentation == 'flip_v':
        return torch.flip(tensor, dims=[1])  # Vertical flip
    elif augmentation == 'rotate_90':
        return torch.rot90(tensor, k=1, dims=[1, 2])
    elif augmentation == 'rotate_180':
        return torch.rot90(tensor, k=2, dims=[1, 2])
    elif augmentation == 'rotate_270':
        return torch.rot90(tensor, k=3, dims=[1, 2])
    else:
        return tensor


class BatchActionMasker:
    """
    Efficient batch action masking for training
    """
    
    def __init__(self, action_space: ChessActionSpace):
        self.action_space = action_space
    
    def get_batch_masks(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Get action masks for a batch of boards
        
        Args:
            boards: List of chess board states
            
        Returns:
            Batch of action masks [batch_size, action_size]
        """
        batch_size = len(boards)
        masks = torch.zeros(batch_size, self.action_space.action_size, dtype=torch.bool)
        
        for i, board in enumerate(boards):
            masks[i] = self.action_space.get_legal_actions_mask(board)
        
        return masks
    
    def apply_masks(self, q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Apply action masks to Q-values
        
        Args:
            q_values: Q-values tensor [batch_size, action_size]
            masks: Action masks [batch_size, action_size]
            
        Returns:
            Masked Q-values
        """
        return q_values.masked_fill(~masks, -1e9)


# Global action space instance
global_action_space = ChessActionSpace()


def get_action_space() -> ChessActionSpace:
    """Get the global action space instance"""
    return global_action_space


def move_to_action(move: chess.Move) -> int:
    """Convert chess move to action index using global action space"""
    return global_action_space.move_to_action(move)


def action_to_move(action: int, board: chess.Board = None) -> Optional[chess.Move]:
    """
    Convert action index to chess move using global action space
    
    Args:
        action: Action index
        board: Chess board (used to validate the move)
        
    Returns:
        Chess move object or None if invalid
    """
    move_uci = global_action_space.action_to_move(action)
    if move_uci is None:
        return None
    
    try:
        move = chess.Move.from_uci(move_uci)
        # If board is provided, check if move is legal
        if board is not None and move not in board.legal_moves:
            return None
        return move
    except:
        return None


def get_legal_actions_mask(board: chess.Board) -> torch.Tensor:
    """Get legal actions mask using global action space"""
    return global_action_space.get_legal_actions_mask(board)


def get_legal_actions(board: chess.Board) -> List[int]:
    """Get legal actions list using global action space"""
    return global_action_space.get_legal_actions(board)


if __name__ == "__main__":
    # Test the action space
    action_space = ChessActionSpace()
    
    # Test with initial position
    board = chess.Board()
    
    print(f"Total action space size: {action_space.action_size}")
    print(f"Legal moves in starting position: {len(list(board.legal_moves))}")
    
    # Test move conversion
    for i, move in enumerate(list(board.legal_moves)[:5]):
        action_idx = action_space.move_to_action(move)
        recovered_move = action_space.action_to_move(action_idx)
        print(f"Move: {move.uci()}, Action: {action_idx}, Recovered: {recovered_move}")
    
    # Test legal action mask
    mask = action_space.get_legal_actions_mask(board)
    print(f"Legal actions mask sum: {mask.sum().item()}")
    
    # Test board to tensor conversion
    tensor = board_to_tensor(board)
    print(f"Board tensor shape: {tensor.shape}")
    print(f"Non-zero elements: {torch.nonzero(tensor).shape[0]}")
    
    # Test augmentation
    augmented = augment_board_tensor(tensor, 'flip_h')
    print(f"Augmented tensor shape: {augmented.shape}")
    print(f"Tensors equal after flip: {torch.equal(tensor, augmented)}")
