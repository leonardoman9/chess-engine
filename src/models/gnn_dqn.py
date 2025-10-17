"""
Graph Neural Network DQN for Chess.

This module implements a complete GNN-based DQN architecture for chess,
combining graph representation learning with reinforcement learning for
strategic game play.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .gnn_layers import (ChessGATLayer, ChessGINELayer, ChessGraphPooling, 
                        PositionalEncoding, ResGATEAUBlock, MultiScaleGNNBlock)
from ..utils.chess_graph import ChessGraphConverter, ChessGraphConfig

@dataclass
class GNNDQNConfig:
    """Configuration for GNN-DQN model."""
    # Graph construction
    graph_config: ChessGraphConfig = None
    
    # Model architecture
    node_input_dim: int = 24  # Enhanced node features
    edge_input_dim: int = 8   # Enhanced edge features  
    hidden_dim: int = 512     # Optimized for A40 (48GB VRAM)
    num_gnn_layers: int = 8   # Deeper network for better performance
    gnn_type: str = 'resgateau'  # 'gat', 'gine', 'hybrid', 'resgateau', 'multiscale'
    
    # Advanced architecture options
    use_residual_blocks: bool = True
    use_multiscale: bool = False
    multiscale_scales: List[str] = None
    
    # GAT specific
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # GINE specific
    gine_eps: float = 0.1
    train_eps: bool = True
    
    # Pooling (AlphaGateau-inspired)
    pooling_types: List[str] = None
    num_pooling_heads: int = 8
    
    # Output
    action_size: int = 4672  # Chess action space
    dueling: bool = True     # Use dueling architecture
    
    # Regularization
    dropout: float = 0.2
    layer_norm: bool = True
    
    # Positional encoding
    use_positional_encoding: bool = True
    
    def __post_init__(self):
        if self.graph_config is None:
            self.graph_config = ChessGraphConfig()
        if self.pooling_types is None:
            self.pooling_types = ['attention', 'hierarchical', 'strategic']
        if self.multiscale_scales is None:
            self.multiscale_scales = ['local', 'tactical', 'strategic']

class HybridGNNLayer(nn.Module):
    """
    Hybrid GNN layer combining GAT and GINE.
    
    Uses GAT for attention-based message passing and GINE for 
    structural information propagation.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 num_edge_types: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Split channels between GAT and GINE
        gat_channels = out_channels // 2
        gine_channels = out_channels - gat_channels
        
        self.gat = ChessGATLayer(
            in_channels=in_channels,
            out_channels=gat_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            num_edge_types=num_edge_types
        )
        
        self.gine = ChessGINELayer(
            in_channels=in_channels,
            out_channels=gine_channels,
            edge_dim=edge_dim,
            num_edge_types=num_edge_types,
            dropout=dropout
        )
        
        # Combine GAT and GINE outputs
        self.combination = nn.Sequential(
            nn.Linear(gat_channels * num_heads + gine_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid layer."""
        # GAT branch
        gat_out = self.gat(x, edge_index, edge_attr, edge_types)
        
        # GINE branch  
        gine_out = self.gine(x, edge_index, edge_attr, edge_types)
        
        # Combine outputs
        combined = torch.cat([gat_out, gine_out], dim=-1)
        return self.combination(combined)

class GNNDQN(nn.Module):
    """
    Complete GNN-DQN architecture for chess.
    
    Architecture:
    1. Graph construction from chess board
    2. Node/edge feature embedding
    3. Stack of GNN layers (GAT/GINE/Hybrid)
    4. Graph pooling to fixed-size representation
    5. Dueling DQN head for Q-value prediction
    """
    
    def __init__(self, config: GNNDQNConfig):
        super().__init__()
        self.config = config
        
        # Graph converter
        self.graph_converter = ChessGraphConverter(config.graph_config)
        
        # Input embeddings
        self.node_embedding = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(config.edge_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(config.hidden_dim)
        else:
            self.pos_encoding = None
        
        # GNN layers (optimized architecture)
        self.gnn_layers = nn.ModuleList()
        for i in range(config.num_gnn_layers):
            if config.gnn_type == 'resgateau':
                # ResGATEAU blocks (AlphaGateau-inspired)
                layer = ResGATEAUBlock(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    edge_dim=config.hidden_dim,
                    num_edge_types=len(config.graph_config.edge_types),
                    num_heads=config.num_attention_heads,
                    dropout=config.dropout,
                    use_hybrid=True
                )
            elif config.gnn_type == 'multiscale':
                # Multi-scale processing
                layer = MultiScaleGNNBlock(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    edge_dim=config.hidden_dim,
                    num_edge_types=len(config.graph_config.edge_types),
                    scales=config.multiscale_scales
                )
            elif config.gnn_type == 'gat':
                layer = ChessGATLayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    concat=True,
                    dropout=config.attention_dropout,
                    edge_dim=config.hidden_dim,
                    num_edge_types=len(config.graph_config.edge_types)
                )
            elif config.gnn_type == 'gine':
                layer = ChessGINELayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    edge_dim=config.hidden_dim,
                    num_edge_types=len(config.graph_config.edge_types),
                    eps=config.gine_eps,
                    train_eps=config.train_eps,
                    dropout=config.dropout
                )
            elif config.gnn_type == 'hybrid':
                layer = HybridGNNLayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    edge_dim=config.hidden_dim,
                    num_edge_types=len(config.graph_config.edge_types),
                    num_heads=config.num_attention_heads,
                    dropout=config.dropout
                )
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Advanced graph pooling (AlphaGateau-inspired)
        self.pooling = ChessGraphPooling(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            pooling_types=config.pooling_types,
            num_attention_heads=config.num_pooling_heads
        )
        
        # Dueling DQN head
        if config.dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.action_size)
            )
        else:
            # Standard DQN head
            self.q_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.action_size)
            )
    
    def forward(self, board_state, return_graph_features: bool = False) -> torch.Tensor:
        """
        Forward pass through GNN-DQN.
        
        Args:
            board_state: Chess board state (chess.Board or pre-computed graph)
            return_graph_features: Whether to return intermediate graph features
            
        Returns:
            Q-values for all actions [batch_size, action_size]
        """
        # Convert board to graph if needed
        if hasattr(board_state, 'piece_at'):  # chess.Board object
            graph_data = self.graph_converter.board_to_graph(board_state)
        else:  # Pre-computed graph
            graph_data = board_state
        
        # Extract graph components
        x = graph_data['node_features']  # [num_nodes, node_features]
        edge_index = graph_data['edge_index']  # [2, num_edges]
        edge_attr = graph_data['edge_features']  # [num_edges, edge_features]
        edge_types = graph_data['edge_types']  # [num_edges]
        
        # Ensure tensors are on the correct device
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        edge_types = edge_types.to(device)
        
        # Embed node and edge features
        x = self.node_embedding(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_embedding(edge_attr)  # [num_edges, hidden_dim]
        
        # Add positional encoding
        if self.pos_encoding is not None:
            positions = torch.arange(64, device=device)  # Square indices 0-63
            x = self.pos_encoding(x, positions)
        
        # Store intermediate features for analysis
        graph_features = [x.clone()]
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr, edge_types)
            graph_features.append(x.clone())
        
        # Graph pooling to get fixed-size representation
        pooled = self.pooling(x)  # [1, hidden_dim] for single graph
        
        # Dueling DQN computation
        if self.config.dueling:
            value = self.value_stream(pooled)  # [1, 1]
            advantage = self.advantage_stream(pooled)  # [1, action_size]
            
            # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(pooled)  # [1, action_size]
        
        if return_graph_features:
            return q_values, {
                'node_features_history': graph_features,
                'final_pooled': pooled,
                'graph_data': graph_data
            }
        
        return q_values
    
    def get_attention_weights(self, board_state) -> Dict:
        """
        Extract attention weights for interpretability.
        
        Returns attention weights from GAT layers for visualization.
        """
        attention_weights = {}
        
        # This would require modifying the forward pass to store attention weights
        # For now, return empty dict - can be implemented later for analysis
        return attention_weights
    
    def analyze_graph_structure(self, board_state) -> Dict:
        """
        Analyze the graph structure for a given board state.
        
        Returns statistics about nodes, edges, and connectivity.
        """
        if hasattr(board_state, 'piece_at'):
            graph_data = self.graph_converter.board_to_graph(board_state)
        else:
            graph_data = board_state
        
        edge_index = graph_data['edge_index']
        edge_types = graph_data['edge_types']
        
        # Basic statistics
        num_nodes = graph_data['num_nodes']
        num_edges = edge_index.size(1)
        
        # Edge type distribution
        edge_type_counts = {}
        for edge_type in edge_types:
            edge_type_counts[edge_type.item()] = edge_type_counts.get(edge_type.item(), 0) + 1
        
        # Node degree statistics
        degrees = torch.zeros(num_nodes)
        for i in range(num_edges):
            degrees[edge_index[0, i]] += 1
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'edge_type_distribution': edge_type_counts,
            'avg_degree': degrees.mean().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item()
        }

# Factory function for easy model creation
def create_gnn_dqn(model_size: str = 'small', 
                   gnn_type: str = 'gat',
                   dueling: bool = True) -> GNNDQN:
    """
    Factory function to create GNN-DQN models with predefined configurations.
    
    Args:
        model_size: 'small', 'medium', or 'large'
        gnn_type: 'gat', 'gine', or 'hybrid'
        dueling: Whether to use dueling architecture
    """
    if model_size == 'small':
        config = GNNDQNConfig(
            hidden_dim=128,
            num_gnn_layers=2,
            num_attention_heads=4,
            gnn_type=gnn_type,
            dueling=dueling
        )
    elif model_size == 'medium':
        config = GNNDQNConfig(
            hidden_dim=256,
            num_gnn_layers=4,
            num_attention_heads=8,
            gnn_type=gnn_type,
            dueling=dueling
        )
    elif model_size == 'large':
        config = GNNDQNConfig(
            hidden_dim=512,
            num_gnn_layers=6,
            num_attention_heads=16,
            gnn_type=gnn_type,
            dueling=dueling
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    return GNNDQN(config)
