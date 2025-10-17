"""
Graph Neural Network layers specialized for chess.

This module implements Graph Attention Networks (GAT) and Graph Isomorphism Networks (GIN)
tailored for chess position understanding. The layers incorporate chess-specific inductive biases
and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
from typing import Optional, Tuple, Union
import math

class ChessGATLayer(MessagePassing):
    """
    Graph Attention Network layer specialized for chess.
    
    Incorporates chess-specific attention mechanisms:
    - Edge-type aware attention
    - Multi-head attention for different chess concepts
    - Residual connections for stable training
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 heads: int = 8,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.1,
                 add_self_loops: bool = True,
                 bias: bool = True,
                 edge_dim: Optional[int] = None,
                 num_edge_types: int = 7):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types
        
        # Linear transformations for nodes
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge feature processing
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = None
        
        # Edge type embeddings
        self.edge_type_emb = nn.Embedding(num_edge_types, heads * out_channels)
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * out_channels, heads * out_channels)
        else:
            self.lin_out = nn.Linear(heads * out_channels, out_channels)
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(heads * out_channels if concat else out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        
        nn.init.xavier_uniform_(self.edge_type_emb.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None,
                size: Optional[Tuple[int, int]] = None,
                return_attention_weights: bool = False):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            edge_types: Edge types [num_edges]
            size: Size of the graph
            return_attention_weights: Whether to return attention weights
        """
        H, C = self.heads, self.out_channels
        
        # Add self-loops
        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, num_nodes=x.size(0))
                if edge_types is not None:
                    # Add self-loop edge types (use a special type, e.g., max + 1)
                    self_loop_types = torch.full((x.size(0),), self.num_edge_types - 1, 
                                               dtype=edge_types.dtype, device=edge_types.device)
                    edge_types = torch.cat([edge_types, self_loop_types], dim=0)
        
        # Linear transformations
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        
        # Propagate messages
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                           edge_types=edge_types, size=size)
        
        # Apply output transformation
        out = self.lin_out(out.view(-1, H * C))
        
        # Add bias
        if self.bias is not None:
            out += self.bias
        
        # Layer normalization
        out = self.layer_norm(out)
        
        # Residual connection
        if self.in_channels == (H * C if self.concat else C):
            out = out + x
        
        if return_attention_weights:
            # This would require storing attention weights during propagation
            # Simplified version for now
            return out, None
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                edge_types: Optional[torch.Tensor],
                index: torch.Tensor, ptr: Optional[torch.Tensor],
                size_i: Optional[int]) -> torch.Tensor:
        """Compute messages between nodes."""
        # Compute attention scores
        alpha = (x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1)
        
        # Add edge features to attention
        if edge_attr is not None and self.lin_edge is not None:
            edge_features = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha += (edge_features * self.att_edge).sum(dim=-1)
        
        # Add edge type information
        if edge_types is not None:
            edge_type_features = self.edge_type_emb(edge_types).view(-1, self.heads, self.out_channels)
            alpha += (edge_type_features * self.att_edge if self.att_edge is not None 
                     else edge_type_features).sum(dim=-1)
        
        # Apply LeakyReLU and softmax
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to messages
        return x_j * alpha.unsqueeze(-1)

class ChessGINELayer(MessagePassing):
    """
    Graph Isomorphism Network with Edge features (GINE) for chess.
    
    Incorporates edge features and chess-specific message passing.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_dim: Optional[int] = None,
                 num_edge_types: int = 7,
                 eps: float = 0.0,
                 train_eps: bool = False,
                 activation: str = 'relu',
                 dropout: float = 0.1):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types
        
        # Learnable epsilon parameter
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # MLP for node updates
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.BatchNorm1d(2 * out_channels),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        
        # Edge feature processing
        if edge_dim is not None:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_channels),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.edge_encoder = None
        
        # Edge type embeddings
        self.edge_type_emb = nn.Embedding(num_edge_types, out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.eps.data.fill_(self.initial_eps)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        if self.edge_encoder is not None:
            for layer in self.edge_encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.edge_type_emb.weight)
        
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Store original features for residual connection
        x_orig = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_types=edge_types)
        
        # Add self-loops with learnable epsilon
        out = (1 + self.eps) * x + out
        
        # Apply MLP
        out = self.mlp(out)
        
        # Residual connection
        out = out + self.residual(x_orig)
        
        return out
    
    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor],
                edge_types: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute messages."""
        msg = x_j
        
        # Add edge features
        if edge_attr is not None and self.edge_encoder is not None:
            edge_features = self.edge_encoder(edge_attr)
            msg = msg + edge_features
        
        # Add edge type information
        if edge_types is not None:
            edge_type_features = self.edge_type_emb(edge_types)
            msg = msg + edge_type_features
        
        return msg

class ChessGraphPooling(nn.Module):
    """
    Advanced graph pooling layer for chess positions inspired by AlphaGateau.
    
    Combines multiple sophisticated pooling strategies:
    - Multi-head attention pooling
    - Hierarchical pooling (pieces vs squares)
    - Strategic importance weighting
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 pooling_types: list = ['attention', 'hierarchical', 'strategic'],
                 num_attention_heads: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling_types = pooling_types
        self.num_attention_heads = num_attention_heads
        
        # Multi-head attention pooling (AlphaGateau style)
        if 'attention' in pooling_types:
            self.multi_head_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channels, in_channels // 4),
                    nn.LeakyReLU(0.2),  # Like AlphaGateau
                    nn.Linear(in_channels // 4, 1)
                ) for _ in range(num_attention_heads)
            ])
            
            # Combine attention heads
            self.attention_combiner = nn.Sequential(
                nn.Linear(num_attention_heads * in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU()
            )
        
        # Hierarchical pooling (separate piece and empty square processing)
        if 'hierarchical' in pooling_types:
            self.piece_attention = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(in_channels // 2, 1)
            )
            
            self.empty_attention = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(in_channels // 2, 1)
            )
            
            self.hierarchical_combiner = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU()
            )
        
        # Strategic importance pooling (weight by square importance)
        if 'strategic' in pooling_types:
            # Learnable strategic square weights
            self.strategic_weights = nn.Parameter(torch.randn(64, 1))
            self.strategic_projector = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU()
            )
        
        # Output projection
        total_channels = len(pooling_types) * in_channels
        self.projection = nn.Sequential(
            nn.Linear(total_channels, out_channels * 2),
            nn.LayerNorm(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, 
                node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool node features to create graph-level representation.
        
        Args:
            x: Node features [num_nodes, in_channels]
            batch: Batch assignment for each node (for batched graphs)
            node_types: Node types for hierarchical pooling (0=empty, 1=piece)
        """
        pooled_features = []
        
        if batch is None:
            # Single graph case (64 squares)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for pool_type in self.pooling_types:
            if pool_type == 'attention':
                # Multi-head attention pooling (AlphaGateau style)
                head_outputs = []
                for head in self.multi_head_attention:
                    att_weights = head(x)  # [num_nodes, 1]
                    att_weights = softmax(att_weights.squeeze(-1), batch)  # [num_nodes]
                    pooled_head = scatter_add(x * att_weights.unsqueeze(-1), batch, dim=0)
                    head_outputs.append(pooled_head)
                
                # Combine attention heads
                combined_heads = torch.cat(head_outputs, dim=-1)
                pooled = self.attention_combiner(combined_heads)
                
            elif pool_type == 'hierarchical':
                # Separate processing for pieces and empty squares
                if node_types is not None:
                    # Identify piece and empty squares
                    piece_mask = (node_types > 0).float()  # 1 for pieces, 0 for empty
                    empty_mask = 1.0 - piece_mask
                    
                    # Piece attention
                    piece_att = self.piece_attention(x)
                    piece_att = piece_att * piece_mask.unsqueeze(-1)  # Mask empty squares
                    piece_att_weights = softmax(piece_att.squeeze(-1), batch)
                    piece_pooled = scatter_add(x * piece_att_weights.unsqueeze(-1), batch, dim=0)
                    
                    # Empty square attention
                    empty_att = self.empty_attention(x)
                    empty_att = empty_att * empty_mask.unsqueeze(-1)  # Mask pieces
                    empty_att_weights = softmax(empty_att.squeeze(-1), batch)
                    empty_pooled = scatter_add(x * empty_att_weights.unsqueeze(-1), batch, dim=0)
                    
                    # Combine hierarchical features
                    hierarchical_combined = torch.cat([piece_pooled, empty_pooled], dim=-1)
                    pooled = self.hierarchical_combiner(hierarchical_combined)
                else:
                    # Fallback to simple attention if no node types provided
                    att_weights = self.piece_attention(x)
                    att_weights = softmax(att_weights.squeeze(-1), batch)
                    pooled = scatter_add(x * att_weights.unsqueeze(-1), batch, dim=0)
                
            elif pool_type == 'strategic':
                # Strategic importance pooling
                strategic_att = torch.sigmoid(self.strategic_weights)  # [64, 1]
                
                # Apply strategic weights to features
                weighted_x = x * strategic_att
                strategic_features = self.strategic_projector(weighted_x)
                
                # Global average with strategic weighting
                pooled = scatter_add(strategic_features, batch, dim=0) / scatter_add(
                    torch.ones_like(strategic_features[:, 0:1]), batch, dim=0)
            
            pooled_features.append(pooled)
        
        # Concatenate all pooled features
        combined = torch.cat(pooled_features, dim=-1)
        
        # Project to output size
        return self.projection(combined)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for chess squares.
    
    Adds learnable positional information to help the GNN understand
    spatial relationships on the chess board.
    """
    
    def __init__(self, d_model: int, max_squares: int = 64):
        super().__init__()
        
        # Learnable positional embeddings for each square
        self.pos_embedding = nn.Embedding(max_squares, d_model)
        
        # Fixed sinusoidal encoding as alternative
        pe = torch.zeros(max_squares, d_model)
        position = torch.arange(0, max_squares, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.use_learnable = True
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to node features.
        
        Args:
            x: Node features [num_nodes, features]
            positions: Square indices [num_nodes] (0-63)
        """
        if positions is None:
            positions = torch.arange(x.size(0), device=x.device)
        
        if self.use_learnable:
            pos_enc = self.pos_embedding(positions)
        else:
            pos_enc = self.pe[positions]
        
        return x + pos_enc

class ResGATEAUBlock(nn.Module):
    """
    Residual GATEAU block inspired by AlphaGateau.
    
    Implements: ResGATEAU(h, g) = (h, g) + GATEAU(BNR(GATEAU(BNR(h, g))))
    where BNR = BatchNorm + ReLU
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 num_edge_types: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_hybrid: bool = True):
        super().__init__()
        
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            # Hybrid GAT+GINE approach
            self.gateau1 = HybridGNNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout
            )
            
            self.gateau2 = HybridGNNLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # Pure GAT approach
            self.gateau1 = ChessGATLayer(
                in_channels=in_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types
            )
            
            self.gateau2 = ChessGATLayer(
                in_channels=out_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types
            )
        
        # Batch normalization and activation (BNR)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # Residual connection projection if needed
        if in_channels != out_channels:
            self.residual_projection = nn.Linear(in_channels, out_channels)
        else:
            self.residual_projection = nn.Identity()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResGATEAU block.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            edge_types: Edge types [num_edges]
        """
        # Store residual
        residual = self.residual_projection(x)
        
        # First GATEAU layer with BNR
        out = self.gateau1(x, edge_index, edge_attr, edge_types)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second GATEAU layer with BNR
        out = self.gateau2(out, edge_index, edge_attr, edge_types)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        
        # Final activation
        out = self.activation(out)
        
        return out

class MultiScaleGNNBlock(nn.Module):
    """
    Multi-scale GNN block that processes information at different scales.
    
    Inspired by multi-scale CNN architectures, this processes chess positions
    at different levels: local (piece interactions), tactical (small groups),
    and strategic (global position).
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 num_edge_types: int,
                 scales: list = ['local', 'tactical', 'strategic']):
        super().__init__()
        
        self.scales = scales
        scale_channels = out_channels // len(scales)
        
        # Local scale: Direct piece interactions (1-hop)
        if 'local' in scales:
            self.local_gnn = ChessGATLayer(
                in_channels=in_channels,
                out_channels=scale_channels // 4,
                heads=4,
                concat=True,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types
            )
        
        # Tactical scale: Small group interactions (2-hop)
        if 'tactical' in scales:
            self.tactical_gnn = ChessGINELayer(
                in_channels=in_channels,
                out_channels=scale_channels,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types
            )
        
        # Strategic scale: Global position understanding
        if 'strategic' in scales:
            self.strategic_gnn = HybridGNNLayer(
                in_channels=in_channels,
                out_channels=scale_channels,
                edge_dim=edge_dim,
                num_edge_types=num_edge_types,
                num_heads=8
            )
        
        # Combine scales
        total_scale_channels = sum([scale_channels if scale != 'local' else scale_channels 
                                  for scale in scales])
        self.scale_combiner = nn.Sequential(
            nn.Linear(total_scale_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale block."""
        scale_outputs = []
        
        if 'local' in self.scales:
            local_out = self.local_gnn(x, edge_index, edge_attr, edge_types)
            scale_outputs.append(local_out)
        
        if 'tactical' in self.scales:
            tactical_out = self.tactical_gnn(x, edge_index, edge_attr, edge_types)
            scale_outputs.append(tactical_out)
        
        if 'strategic' in self.scales:
            strategic_out = self.strategic_gnn(x, edge_index, edge_attr, edge_types)
            scale_outputs.append(strategic_out)
        
        # Combine all scales
        combined = torch.cat(scale_outputs, dim=-1)
        return self.scale_combiner(combined)
