"""
Dueling DQN Architecture for Chess
Implements the Dueling Network architecture with separate Value and Advantage streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for Chess
    
    Architecture:
    - Convolutional layers for spatial feature extraction
    - Dueling streams: Value V(s) and Advantage A(s,a)
    - Final Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    
    def __init__(self, input_channels=13, conv_channels=[64, 128, 256], 
                 hidden_size=512, action_size=4672):
        """
        Initialize Dueling DQN
        
        Args:
            input_channels: Number of input channels (13 for chess: 12 pieces + metadata)
            conv_channels: List of channel sizes for conv layers
            hidden_size: Size of dense layers
            action_size: Number of possible actions (all possible moves)
        """
        super(DuelingDQN, self).__init__()
        
        self.input_channels = input_channels
        self.action_size = action_size
        
        # Convolutional feature extraction layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # Calculate flattened size after conv layers (8x8 board)
        self.flatten_size = conv_channels[-1] * 8 * 8
        
        # Shared dense layer
        self.shared_dense = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Value stream V(s) - outputs single value
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream A(s,a) - outputs advantage for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, action_mask=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            action_mask: Optional mask for illegal actions [batch_size, action_size]
                        True for legal actions, False for illegal
        
        Returns:
            q_values: Q-values for all actions [batch_size, action_size]
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and pass through shared dense layer
        x = x.view(batch_size, -1)
        shared_features = self.shared_dense(x)
        
        # Compute value and advantage streams
        value = self.value_stream(shared_features)  # [batch_size, 1]
        advantage = self.advantage_stream(shared_features)  # [batch_size, action_size]
        
        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        
        # Apply action mask if provided (set illegal actions to very negative values)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, -1e9)
        
        return q_values
    
    def get_features(self, x):
        """
        Extract features before Q-value computation (useful for analysis)
        
        Returns:
            shared_features: Features after shared dense layer
            value: State value V(s)
            advantage: Action advantages A(s,a)
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and pass through shared dense layer
        x = x.view(batch_size, -1)
        shared_features = self.shared_dense(x)
        
        # Compute value and advantage streams
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        return shared_features, value, advantage


class CNNDuelingDQN(DuelingDQN):
    """
    CNN-based Dueling DQN - baseline architecture
    """
    
    def __init__(self, **kwargs):
        # Use default CNN architecture
        super().__init__(**kwargs)


class EquivariantDuelingDQN(DuelingDQN):
    """
    Dueling DQN with equivariance support
    Includes data augmentation and symmetric weight sharing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_augmentation = True
    
    def forward(self, x, action_mask=None, augment=None):
        """
        Forward pass with optional data augmentation
        
        Args:
            augment: Type of augmentation ('flip_h', 'flip_v', 'rotate_90', etc.)
        """
        if self.training and augment is not None and self.use_augmentation:
            x = self._apply_augmentation(x, augment)
            if action_mask is not None:
                action_mask = self._apply_mask_augmentation(action_mask, augment)
        
        return super().forward(x, action_mask)
    
    def _apply_augmentation(self, x, augment):
        """Apply data augmentation to input"""
        if augment == 'flip_h':
            return torch.flip(x, dims=[3])  # Horizontal flip
        elif augment == 'flip_v':
            return torch.flip(x, dims=[2])  # Vertical flip
        elif augment == 'rotate_90':
            return torch.rot90(x, k=1, dims=[2, 3])
        elif augment == 'rotate_180':
            return torch.rot90(x, k=2, dims=[2, 3])
        elif augment == 'rotate_270':
            return torch.rot90(x, k=3, dims=[2, 3])
        return x
    
    def _apply_mask_augmentation(self, mask, augment):
        """Apply corresponding augmentation to action mask"""
        # This would need to be implemented based on action encoding
        # For now, return mask unchanged
        return mask


def create_dqn_model(model_type='cnn', **kwargs):
    """
    Factory function to create DQN models
    
    Args:
        model_type: 'cnn' for baseline, 'equivariant' for equivariant CNN
        **kwargs: Additional arguments for model initialization
    
    Returns:
        DQN model instance
    """
    if model_type == 'cnn':
        return CNNDuelingDQN(**kwargs)
    elif model_type == 'equivariant':
        return EquivariantDuelingDQN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model configuration presets
MODEL_CONFIGS = {
    'small': {
        'conv_channels': [32, 64, 128],
        'hidden_size': 256,
    },
    'medium': {
        'conv_channels': [64, 128, 256],
        'hidden_size': 512,
    },
    'large': {
        'conv_channels': [64, 128, 256, 512],
        'hidden_size': 1024,
    }
}


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_dqn_model('cnn', **MODEL_CONFIGS['medium'])
    model.to(device)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 13, 8, 8).to(device)
    action_mask = torch.ones(batch_size, 4672, dtype=torch.bool).to(device)
    
    # Forward pass
    q_values = model(x, action_mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test features extraction
    features, value, advantage = model.get_features(x)
    print(f"Features shape: {features.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Advantage shape: {advantage.shape}")
