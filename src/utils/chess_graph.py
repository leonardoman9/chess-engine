"""
Chess Board to Graph Conversion for GNN-based Chess Engine.

This module converts chess board states into graph representations suitable for
Graph Neural Networks (GNNs). The graph captures piece relationships, attacks,
defenses, and spatial information that CNNs might miss.

Graph Structure:
- Nodes: Chess pieces + empty squares (64 nodes total)
- Edges: Attacks, defenses, spatial proximity, piece mobility
- Node Features: Piece type, color, position, value, mobility
- Edge Features: Relationship type, distance, strength
"""

import chess
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class EdgeType(Enum):
    """Types of edges in the chess graph."""
    ATTACK = 0      # Piece A attacks piece B
    DEFEND = 1      # Piece A defends piece B  
    SPATIAL = 2     # Spatial proximity (adjacent squares)
    MOBILITY = 3    # Piece can move to square
    PIN = 4         # Piece A pins piece B
    FORK = 5        # Piece A forks pieces B and C
    DISCOVERY = 6   # Moving piece A discovers attack from piece B

@dataclass
class ChessGraphConfig:
    """Configuration for chess graph construction."""
    include_empty_squares: bool = True
    max_spatial_distance: int = 2  # Include edges up to 2 squares away
    include_mobility_edges: bool = True
    include_tactical_edges: bool = True  # pins, forks, discoveries
    normalize_features: bool = True
    edge_types: List[EdgeType] = None
    
    def __post_init__(self):
        if self.edge_types is None:
            self.edge_types = [EdgeType.ATTACK, EdgeType.DEFEND, EdgeType.SPATIAL, EdgeType.MOBILITY]

class ChessGraphConverter:
    """Converts chess board states to graph representations."""
    
    def __init__(self, config: ChessGraphConfig = None):
        self.config = config or ChessGraphConfig()
        
        # Piece values for features
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        
        # Pre-compute spatial distances
        self._precompute_spatial_distances()
    
    def _precompute_spatial_distances(self):
        """Pre-compute distances between all squares."""
        self.spatial_distances = {}
        for sq1 in range(64):
            for sq2 in range(64):
                rank1, file1 = chess.square_rank(sq1), chess.square_file(sq1)
                rank2, file2 = chess.square_rank(sq2), chess.square_file(sq2)
                distance = max(abs(rank1 - rank2), abs(file1 - file2))
                self.spatial_distances[(sq1, sq2)] = distance
    
    def board_to_graph(self, board: chess.Board) -> Dict:
        """
        Convert a chess board to graph representation.
        
        Args:
            board: chess.Board object
            
        Returns:
            Dictionary containing:
            - node_features: [64, num_node_features] tensor
            - edge_index: [2, num_edges] tensor (source, target indices)
            - edge_features: [num_edges, num_edge_features] tensor
            - edge_types: [num_edges] tensor (edge type indices)
        """
        # Extract node features
        node_features = self._extract_node_features(board)
        
        # Extract edges
        edge_data = self._extract_edges(board)
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_data['edge_index'], dtype=torch.long).t(),
            'edge_features': torch.tensor(edge_data['edge_features'], dtype=torch.float32),
            'edge_types': torch.tensor(edge_data['edge_types'], dtype=torch.long),
            'num_nodes': 64
        }
    
    def _extract_node_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract enhanced node features for each square.
        
        Features per node (total: 24):
        - Piece type (6): one-hot encoded (pawn, knight, bishop, rook, queen, king)
        - Piece color (1): 1 for white, -1 for black, 0 for empty
        - Piece value (1): normalized piece value
        - Position features (2): rank, file (normalized)
        - Mobility (1): number of legal moves from this square (normalized)
        - Under attack (1): 1 if square is under attack by opponent
        - Defended (1): 1 if piece is defended by ally
        - Pinned (1): 1 if piece is pinned
        - En passant (1): 1 if square is en passant target
        - Castling rights (1): 1 if piece is king/rook involved in castling
        - ENHANCED FEATURES (8):
        - Attack count (1): number of pieces attacking this square
        - Defense count (1): number of pieces defending this square  
        - Control value (1): difference between attackers and defenders
        - Distance to king (1): normalized distance to own king
        - Distance to enemy king (1): normalized distance to enemy king
        - Piece development (1): 1 if piece moved from starting position
        - Square control (1): strategic value of controlling this square
        - Tactical importance (1): involvement in tactical patterns
        """
        features = np.zeros((64, 24), dtype=np.float32)
        
        # Pre-compute king positions for distance calculations
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        # Strategic square values (center control, etc.)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, 
                          chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]
        
        for square in range(64):
            piece = board.piece_at(square)
            rank, file = chess.square_rank(square), chess.square_file(square)
            
            # Position features (normalized to [0, 1])
            features[square, 12] = rank / 7.0
            features[square, 13] = file / 7.0
            
            if piece is not None:
                # Piece type (one-hot)
                piece_idx = piece.piece_type - 1  # 0-5
                features[square, piece_idx] = 1.0
                
                # Piece color
                features[square, 6] = 1.0 if piece.color == chess.WHITE else -1.0
                
                # Piece value (normalized)
                features[square, 7] = self.piece_values[piece.piece_type] / 100.0
                
                # Mobility (number of legal moves from this square)
                mobility = len([move for move in board.legal_moves if move.from_square == square])
                features[square, 8] = min(mobility / 27.0, 1.0)  # Queen max ~27 moves
                
                # Under attack by opponent
                features[square, 9] = 1.0 if board.is_attacked_by(not piece.color, square) else 0.0
                
                # Defended by ally
                features[square, 10] = 1.0 if board.is_attacked_by(piece.color, square) else 0.0
                
                # Pinned
                features[square, 11] = 1.0 if board.is_pinned(piece.color, square) else 0.0
                
                # ENHANCED FEATURES:
                
                # Attack count (normalized)
                attackers = len(board.attackers(not piece.color, square))
                features[square, 16] = min(attackers / 8.0, 1.0)
                
                # Defense count (normalized)
                defenders = len(board.attackers(piece.color, square))
                features[square, 17] = min(defenders / 8.0, 1.0)
                
                # Control value (attackers - defenders, normalized)
                control_diff = defenders - attackers
                features[square, 18] = max(-1.0, min(1.0, control_diff / 4.0))
                
                # Distance to own king (normalized)
                own_king = white_king_sq if piece.color == chess.WHITE else black_king_sq
                if own_king is not None:
                    king_dist = chess.square_distance(square, own_king)
                    features[square, 19] = king_dist / 7.0
                
                # Distance to enemy king (normalized)
                enemy_king = black_king_sq if piece.color == chess.WHITE else white_king_sq
                if enemy_king is not None:
                    enemy_king_dist = chess.square_distance(square, enemy_king)
                    features[square, 20] = enemy_king_dist / 7.0
                
                # Piece development (1 if moved from starting position)
                starting_squares = {
                    chess.WHITE: {
                        chess.PAWN: list(range(chess.A2, chess.H2 + 1)),
                        chess.ROOK: [chess.A1, chess.H1],
                        chess.KNIGHT: [chess.B1, chess.G1],
                        chess.BISHOP: [chess.C1, chess.F1],
                        chess.QUEEN: [chess.D1],
                        chess.KING: [chess.E1]
                    },
                    chess.BLACK: {
                        chess.PAWN: list(range(chess.A7, chess.H7 + 1)),
                        chess.ROOK: [chess.A8, chess.H8],
                        chess.KNIGHT: [chess.B8, chess.G8],
                        chess.BISHOP: [chess.C8, chess.F8],
                        chess.QUEEN: [chess.D8],
                        chess.KING: [chess.E8]
                    }
                }
                
                is_developed = square not in starting_squares[piece.color].get(piece.piece_type, [])
                features[square, 21] = 1.0 if is_developed else 0.0
                
            # Square control value (strategic importance)
            square_value = 0.0
            if square in center_squares:
                square_value = 1.0
            elif square in extended_center:
                square_value = 0.6
            elif rank in [1, 6]:  # 2nd and 7th ranks
                square_value = 0.4
            features[square, 22] = square_value
            
            # Tactical importance (simplified)
            tactical_value = 0.0
            if piece is not None:
                # High value if piece can capture or be captured
                if board.is_attacked_by(not piece.color, square):
                    tactical_value += 0.5
                if len([move for move in board.legal_moves 
                       if move.from_square == square and board.is_capture(move)]) > 0:
                    tactical_value += 0.5
            features[square, 23] = min(tactical_value, 1.0)
            
            # En passant target
            if board.ep_square == square:
                features[square, 14] = 1.0
            
            # Castling rights
            if piece and piece.piece_type == chess.KING:
                if board.has_castling_rights(piece.color):
                    features[square, 15] = 1.0
            elif piece and piece.piece_type == chess.ROOK:
                if (square in [chess.A1, chess.H1] and piece.color == chess.WHITE and 
                    board.has_castling_rights(chess.WHITE)) or \
                   (square in [chess.A8, chess.H8] and piece.color == chess.BLACK and 
                    board.has_castling_rights(chess.BLACK)):
                    features[square, 15] = 1.0
        
        return features
    
    def _extract_edges(self, board: chess.Board) -> Dict:
        """Extract edges between squares."""
        edge_index = []
        edge_features = []
        edge_types = []
        
        # Attack edges
        if EdgeType.ATTACK in self.config.edge_types:
            self._add_attack_edges(board, edge_index, edge_features, edge_types)
        
        # Defense edges
        if EdgeType.DEFEND in self.config.edge_types:
            self._add_defense_edges(board, edge_index, edge_features, edge_types)
        
        # Spatial edges
        if EdgeType.SPATIAL in self.config.edge_types:
            self._add_spatial_edges(board, edge_index, edge_features, edge_types)
        
        # Mobility edges
        if EdgeType.MOBILITY in self.config.edge_types:
            self._add_mobility_edges(board, edge_index, edge_features, edge_types)
        
        # Tactical edges (pins, forks, discoveries)
        if self.config.include_tactical_edges:
            self._add_tactical_edges(board, edge_index, edge_features, edge_types)
        
        return {
            'edge_index': edge_index,
            'edge_features': edge_features,
            'edge_types': edge_types
        }
    
    def _add_attack_edges(self, board: chess.Board, edge_index: List, 
                         edge_features: List, edge_types: List):
        """Add edges for piece attacks."""
        for square in range(64):
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Get all squares this piece attacks
            attacks = board.attacks(square)
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                
                # Edge from attacker to target
                edge_index.append([square, target_square])
                edge_types.append(EdgeType.ATTACK.value)
                
                # Enhanced edge features: [distance, piece_value_diff, is_capture, is_check, 
                #                        move_direction, piece_mobility, tactical_value, positional_gain]
                distance = self.spatial_distances[(square, target_square)]
                piece_value = self.piece_values[piece.piece_type]
                target_value = self.piece_values[target_piece.piece_type] if target_piece else 0
                value_diff = (target_value - piece_value) / 100.0  # Normalized
                
                is_capture = 1.0 if target_piece and target_piece.color != piece.color else 0.0
                
                # Check if this attack gives check
                temp_board = board.copy()
                is_check = 0.0
                try:
                    move = chess.Move(square, target_square)
                    if move in board.legal_moves:
                        temp_board.push(move)
                        is_check = 1.0 if temp_board.is_check() else 0.0
                        temp_board.pop()
                except:
                    pass
                
                # Move direction (8 directions: N, NE, E, SE, S, SW, W, NW)
                rank_diff = chess.square_rank(target_square) - chess.square_rank(square)
                file_diff = chess.square_file(target_square) - chess.square_file(square)
                
                direction = 0.0
                if rank_diff > 0 and file_diff == 0: direction = 0.0    # N
                elif rank_diff > 0 and file_diff > 0: direction = 0.125  # NE
                elif rank_diff == 0 and file_diff > 0: direction = 0.25  # E
                elif rank_diff < 0 and file_diff > 0: direction = 0.375  # SE
                elif rank_diff < 0 and file_diff == 0: direction = 0.5   # S
                elif rank_diff < 0 and file_diff < 0: direction = 0.625  # SW
                elif rank_diff == 0 and file_diff < 0: direction = 0.75  # W
                elif rank_diff > 0 and file_diff < 0: direction = 0.875  # NW
                
                # Piece mobility after move (simplified)
                piece_mobility = len([m for m in board.legal_moves if m.from_square == square]) / 27.0
                
                # Tactical value (captures high-value pieces, creates threats)
                tactical_value = 0.0
                if is_capture:
                    tactical_value += target_value / 100.0  # Value of captured piece
                if is_check:
                    tactical_value += 0.3  # Check bonus
                
                # Positional gain (moving to center, advancing pawns, etc.)
                positional_gain = 0.0
                target_rank, target_file = chess.square_rank(target_square), chess.square_file(target_square)
                
                # Center control bonus
                if target_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    positional_gain += 0.5
                elif target_square in [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, 
                                     chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]:
                    positional_gain += 0.3
                
                # Pawn advancement bonus
                if piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE and target_rank > chess.square_rank(square):
                        positional_gain += (target_rank - 1) / 6.0  # Normalize to [0,1]
                    elif piece.color == chess.BLACK and target_rank < chess.square_rank(square):
                        positional_gain += (6 - target_rank) / 6.0  # Normalize to [0,1]
                
                edge_features.append([distance / 7.0, value_diff, is_capture, is_check, 
                                    direction, piece_mobility, tactical_value, positional_gain])
    
    def _add_defense_edges(self, board: chess.Board, edge_index: List,
                          edge_features: List, edge_types: List):
        """Add edges for piece defenses."""
        for square in range(64):
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Get all squares this piece attacks (including friendly pieces = defends)
            attacks = board.attacks(square)
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                
                # Defense edge: same color pieces
                if target_piece and target_piece.color == piece.color:
                    edge_index.append([square, target_square])
                    edge_types.append(EdgeType.DEFEND.value)
                    
                    distance = self.spatial_distances[(square, target_square)]
                    piece_value = self.piece_values[piece.piece_type]
                    target_value = self.piece_values[target_piece.piece_type]
                    value_ratio = target_value / piece_value  # How valuable is defended piece
                    
                    edge_features.append([distance / 7.0, value_ratio / 10.0, 1.0, 0.0])
    
    def _add_spatial_edges(self, board: chess.Board, edge_index: List,
                          edge_features: List, edge_types: List):
        """Add edges for spatial proximity."""
        for sq1 in range(64):
            for sq2 in range(sq1 + 1, 64):  # Avoid duplicates
                distance = self.spatial_distances[(sq1, sq2)]
                
                if distance <= self.config.max_spatial_distance:
                    # Bidirectional spatial edges
                    edge_index.extend([[sq1, sq2], [sq2, sq1]])
                    edge_types.extend([EdgeType.SPATIAL.value, EdgeType.SPATIAL.value])
                    
                    # Edge features: [distance, rank_diff, file_diff, diagonal]
                    rank1, file1 = chess.square_rank(sq1), chess.square_file(sq1)
                    rank2, file2 = chess.square_rank(sq2), chess.square_file(sq2)
                    
                    rank_diff = abs(rank1 - rank2) / 7.0
                    file_diff = abs(file1 - file2) / 7.0
                    is_diagonal = 1.0 if abs(rank1 - rank2) == abs(file1 - file2) else 0.0
                    
                    spatial_features = [distance / 7.0, rank_diff, file_diff, is_diagonal]
                    edge_features.extend([spatial_features, spatial_features])
    
    def _add_mobility_edges(self, board: chess.Board, edge_index: List,
                           edge_features: List, edge_types: List):
        """Add edges for piece mobility (legal moves)."""
        for move in board.legal_moves:
            from_sq, to_sq = move.from_square, move.to_square
            piece = board.piece_at(from_sq)
            
            if piece is not None:
                edge_index.append([from_sq, to_sq])
                edge_types.append(EdgeType.MOBILITY.value)
                
                distance = self.spatial_distances[(from_sq, to_sq)]
                piece_value = self.piece_values[piece.piece_type]
                
                # Check if move is capture, check, or promotion
                is_capture = 1.0 if board.is_capture(move) else 0.0
                is_check = 1.0 if board.gives_check(move) else 0.0
                is_promotion = 1.0 if move.promotion else 0.0
                
                edge_features.append([distance / 7.0, piece_value / 100.0, is_capture, is_check])
    
    def _add_tactical_edges(self, board: chess.Board, edge_index: List,
                           edge_features: List, edge_types: List):
        """Add edges for tactical relationships (pins, forks, etc.)."""
        # This is a simplified version - full tactical analysis is complex
        
        # Pin detection
        for square in range(64):
            piece = board.piece_at(square)
            if piece and board.is_pinned(piece.color, square):
                # Find the pinning piece
                for attacker_sq in range(64):
                    attacker = board.piece_at(attacker_sq)
                    if (attacker and attacker.color != piece.color and 
                        square in board.attacks(attacker_sq)):
                        
                        edge_index.append([attacker_sq, square])
                        edge_types.append(EdgeType.PIN.value)
                        
                        distance = self.spatial_distances[(attacker_sq, square)]
                        edge_features.append([distance / 7.0, 1.0, 0.0, 0.0])  # Pin strength = 1.0
    
    def get_node_feature_names(self) -> List[str]:
        """Get names of node features for interpretability."""
        return [
            'pawn', 'knight', 'bishop', 'rook', 'queen', 'king',  # 0-5
            'color', 'value', 'mobility', 'under_attack', 'defended', 'pinned',  # 6-11
            'rank', 'file', 'en_passant', 'castling_rights',  # 12-15
            'attack_count', 'defense_count', 'control_value', 'distance_to_king',  # 16-19
            'distance_to_enemy_king', 'piece_development', 'square_control', 'tactical_importance'  # 20-23
        ]
    
    def get_edge_feature_names(self) -> List[str]:
        """Get names of edge features for interpretability."""
        return ['distance', 'value_diff', 'is_capture', 'is_check', 
                'move_direction', 'piece_mobility', 'tactical_value', 'positional_gain']

# Utility function for easy integration
def board_to_graph(board: chess.Board, config: ChessGraphConfig = None) -> Dict:
    """Convert chess board to graph representation."""
    converter = ChessGraphConverter(config)
    return converter.board_to_graph(board)
