"""
AlphaZero-style board representation for chess
Implements the input representation used in the AlphaZero paper
"""

import chess
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque


class AlphaZeroChessRepresentation:
    """
    AlphaZero-style board representation for chess
    
    Input features (119 planes total):
    - 6 planes for P1 pieces (current player)
    - 6 planes for P2 pieces (opponent)  
    - 2 planes for repetitions
    - 1 plane for color
    - 1 plane for total move count
    - 2 planes for castling rights (P1 kingside/queenside)
    - 2 planes for castling rights (P2 kingside/queenside)
    - 1 plane for no-progress count
    
    All repeated for T=8 time steps = 8 * 14 + 7 = 119 planes
    """
    
    def __init__(self, history_length: int = 8):
        self.history_length = history_length
        self.board_size = 8
        
        # Piece type mapping (excluding king for now, will be separate)
        self.piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        # Initialize history buffer
        self.position_history: Deque[chess.Board] = deque(maxlen=history_length)
        
    def reset_history(self):
        """Reset the position history"""
        self.position_history.clear()
        
    def add_position(self, board: chess.Board):
        """Add a position to the history"""
        # Store a copy of the board
        self.position_history.append(board.copy())
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to AlphaZero-style tensor representation
        
        Args:
            board: Current chess board position
            
        Returns:
            Tensor of shape (119, 8, 8) representing the position
        """
        # Add current position to history
        self.add_position(board)
        
        # Calculate total planes needed
        planes_per_position = 14  # 6 + 6 + 2 planes per position
        constant_planes = 7       # color, move count, castling (4), no-progress
        total_planes = planes_per_position * self.history_length + constant_planes
        
        # Initialize tensor
        tensor = torch.zeros(total_planes, self.board_size, self.board_size)
        plane_idx = 0
        
        # Process each position in history (most recent first)
        positions_to_process = list(self.position_history)
        
        # Pad with empty positions if we don't have full history
        while len(positions_to_process) < self.history_length:
            positions_to_process.insert(0, chess.Board())  # Empty board
            
        # Take only the most recent history_length positions
        positions_to_process = positions_to_process[-self.history_length:]
        
        for pos_idx, historical_board in enumerate(positions_to_process):
            # Determine perspective (always from current player's view)
            current_player = board.turn
            
            # P1 pieces (current player's pieces)
            for piece_type in self.piece_types:
                piece_plane = torch.zeros(self.board_size, self.board_size)
                piece_squares = historical_board.pieces(piece_type, current_player)
                
                for square in piece_squares:
                    row, col = divmod(square, 8)
                    if current_player == chess.WHITE:
                        piece_plane[7-row, col] = 1.0  # White perspective
                    else:
                        piece_plane[row, 7-col] = 1.0  # Black perspective (flipped)
                        
                tensor[plane_idx] = piece_plane
                plane_idx += 1
            
            # P2 pieces (opponent's pieces)  
            opponent = not current_player
            for piece_type in self.piece_types:
                piece_plane = torch.zeros(self.board_size, self.board_size)
                piece_squares = historical_board.pieces(piece_type, opponent)
                
                for square in piece_squares:
                    row, col = divmod(square, 8)
                    if current_player == chess.WHITE:
                        piece_plane[7-row, col] = 1.0  # White perspective
                    else:
                        piece_plane[row, 7-col] = 1.0  # Black perspective (flipped)
                        
                tensor[plane_idx] = piece_plane
                plane_idx += 1
            
            # Repetition planes (2 planes)
            # For simplicity, we'll implement basic repetition detection
            repetition_count = self._count_repetitions(historical_board)
            
            # First repetition plane (1 repetition)
            if repetition_count >= 1:
                tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
            plane_idx += 1
            
            # Second repetition plane (2+ repetitions)  
            if repetition_count >= 2:
                tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
            plane_idx += 1
        
        # Constant planes (same for all time steps)
        
        # Color plane (1 if white to move, 0 if black)
        color_value = 1.0 if board.turn == chess.WHITE else 0.0
        tensor[plane_idx] = torch.full((self.board_size, self.board_size), color_value)
        plane_idx += 1
        
        # Total move count plane
        move_count = board.fullmove_number
        move_count_normalized = min(move_count / 100.0, 1.0)  # Normalize to [0,1]
        tensor[plane_idx] = torch.full((self.board_size, self.board_size), move_count_normalized)
        plane_idx += 1
        
        # Castling rights planes (4 planes total)
        # P1 (current player) kingside castling
        if board.has_kingside_castling_rights(board.turn):
            tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
        plane_idx += 1
        
        # P1 (current player) queenside castling  
        if board.has_queenside_castling_rights(board.turn):
            tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
        plane_idx += 1
        
        # P2 (opponent) kingside castling
        if board.has_kingside_castling_rights(not board.turn):
            tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
        plane_idx += 1
        
        # P2 (opponent) queenside castling
        if board.has_queenside_castling_rights(not board.turn):
            tensor[plane_idx] = torch.ones(self.board_size, self.board_size)
        plane_idx += 1
        
        # No-progress count plane (50-move rule)
        no_progress_count = board.halfmove_clock
        no_progress_normalized = min(no_progress_count / 50.0, 1.0)  # Normalize to [0,1]
        tensor[plane_idx] = torch.full((self.board_size, self.board_size), no_progress_normalized)
        plane_idx += 1
        
        return tensor
    
    def _count_repetitions(self, board: chess.Board) -> int:
        """
        Count how many times the current position has occurred
        
        Args:
            board: Chess board to check
            
        Returns:
            Number of repetitions (0, 1, 2+)
        """
        if len(self.position_history) < 2:
            return 0
            
        current_fen = board.fen().split(' ')[0]  # Only board position, not move info
        count = 0
        
        for historical_board in self.position_history:
            historical_fen = historical_board.fen().split(' ')[0]
            if current_fen == historical_fen:
                count += 1
                
        return max(0, count - 1)  # Subtract 1 because current position is included


# Global instance for backward compatibility
alphazero_representation = AlphaZeroChessRepresentation()


def board_to_alphazero_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert chess board to AlphaZero-style tensor (backward compatibility function)
    
    Args:
        board: Chess board position
        
    Returns:
        Tensor of shape (119, 8, 8)
    """
    return alphazero_representation.board_to_tensor(board)


def reset_board_history():
    """Reset the global board history"""
    alphazero_representation.reset_history()
