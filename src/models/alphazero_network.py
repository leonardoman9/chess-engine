"""
AlphaZero-style Neural Network for Chess
Implements policy + value network architecture similar to AlphaZero paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero-style network"""
    input_channels: int = 119  # AlphaZero uses 119 planes for chess
    conv_channels: List[int] = None
    residual_blocks: int = 10  # Number of residual blocks
    policy_head_channels: int = 2
    value_head_channels: int = 1
    action_size: int = 4672  # 8*8*73 possible moves
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [256, 256, 256]  # AlphaZero uses 256 channels


class ResidualBlock(nn.Module):
    """Residual block used in AlphaZero"""
    
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = F.relu(out)
        
        return out


class PolicyHead(nn.Module):
    """Policy head for move probability prediction"""
    
    def __init__(self, input_channels: int, policy_channels: int = 2, action_size: int = 4672):
        super().__init__()
        self.action_size = action_size
        
        # Convolutional layer
        self.conv = nn.Conv2d(input_channels, policy_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(policy_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(policy_channels * 8 * 8, action_size)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, 8, 8)
        
        # Convolutional processing
        out = self.conv(x)  # (batch_size, policy_channels, 8, 8)
        out = self.bn(out)
        out = F.relu(out)
        
        # Flatten for fully connected layer
        out = out.view(out.size(0), -1)  # (batch_size, policy_channels * 64)
        
        # Output policy logits
        policy_logits = self.fc(out)  # (batch_size, action_size)
        
        return policy_logits


class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, input_channels: int, value_channels: int = 1, hidden_size: int = 256):
        super().__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(input_channels, value_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(value_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(value_channels * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, 8, 8)
        
        # Convolutional processing
        out = self.conv(x)  # (batch_size, value_channels, 8, 8)
        out = self.bn(out)
        out = F.relu(out)
        
        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)  # (batch_size, value_channels * 64)
        
        # Hidden layer
        out = self.fc1(out)
        out = F.relu(out)
        
        # Output value (tanh to get [-1, 1] range)
        value = torch.tanh(self.fc2(out))  # (batch_size, 1)
        
        return value


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style network for chess
    
    Architecture:
    - Initial convolutional layer
    - Stack of residual blocks
    - Policy head (outputs move probabilities)
    - Value head (outputs position evaluation)
    """
    
    def __init__(self, config: AlphaZeroConfig = None):
        super().__init__()
        
        if config is None:
            config = AlphaZeroConfig()
        
        self.config = config
        
        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(
            config.input_channels, 
            config.conv_channels[0], 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.initial_bn = nn.BatchNorm2d(config.conv_channels[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.conv_channels[0], config.dropout)
            for _ in range(config.residual_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            config.conv_channels[0], 
            config.policy_head_channels, 
            config.action_size
        )
        self.value_head = ValueHead(
            config.conv_channels[0], 
            config.value_head_channels
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 119, 8, 8)
            
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch_size, 4672) - raw logits for each possible move
            - value: (batch_size, 1) - position evaluation in [-1, 1]
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Policy and value heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        
        return policy_logits, value
    
    def predict_move_probabilities(self, x, legal_moves_mask=None) -> torch.Tensor:
        """
        Predict move probabilities with optional legal move masking
        
        Args:
            x: Input tensor
            legal_moves_mask: Optional mask for legal moves
            
        Returns:
            Move probabilities (softmax over legal moves)
        """
        policy_logits, _ = self.forward(x)
        
        if legal_moves_mask is not None:
            # Mask illegal moves by setting their logits to -inf
            policy_logits = policy_logits.masked_fill(~legal_moves_mask, float('-inf'))
        
        # Apply softmax to get probabilities
        move_probabilities = F.softmax(policy_logits, dim=-1)
        
        return move_probabilities
    
    def predict_value(self, x) -> torch.Tensor:
        """
        Predict position value
        
        Args:
            x: Input tensor
            
        Returns:
            Position value in [-1, 1]
        """
        _, value = self.forward(x)
        return value


def create_alphazero_network(config: AlphaZeroConfig = None) -> AlphaZeroNetwork:
    """
    Factory function to create AlphaZero network
    
    Args:
        config: Network configuration
        
    Returns:
        AlphaZero network instance
    """
    if config is None:
        config = AlphaZeroConfig()
    
    return AlphaZeroNetwork(config)


# Predefined configurations
ALPHAZERO_CONFIGS = {
    "small": AlphaZeroConfig(
        conv_channels=[128],
        residual_blocks=5,
        dropout=0.1
    ),
    "medium": AlphaZeroConfig(
        conv_channels=[256],
        residual_blocks=10,
        dropout=0.1
    ),
    "large": AlphaZeroConfig(
        conv_channels=[256],
        residual_blocks=20,
        dropout=0.2
    )
}
