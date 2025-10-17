"""
AlphaZero Loss Function
Implements the combined loss from the AlphaZero paper:
l = (z - v)² - π^T log p + c||θ||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero loss function combining value loss and policy loss
    
    Loss = MSE(value, target) + CrossEntropy(policy, target_policy) + L2_regularization
    """
    
    def __init__(self, value_loss_weight: float = 1.0, policy_loss_weight: float = 1.0, 
                 l2_regularization: float = 1e-4):
        """
        Initialize AlphaZero loss
        
        Args:
            value_loss_weight: Weight for value loss component
            policy_loss_weight: Weight for policy loss component  
            l2_regularization: L2 regularization coefficient (c in paper)
        """
        super().__init__()
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.l2_regularization = l2_regularization
        
    def forward(self, policy_logits: torch.Tensor, value_pred: torch.Tensor,
                target_policy: torch.Tensor, target_value: torch.Tensor,
                model_parameters=None) -> Tuple[torch.Tensor, dict]:
        """
        Compute AlphaZero loss
        
        Args:
            policy_logits: Raw policy logits from network (batch_size, action_size)
            value_pred: Predicted values from network (batch_size, 1)
            target_policy: Target policy distribution (batch_size, action_size)
            target_value: Target game outcome z ∈ {-1, 0, 1} (batch_size, 1)
            model_parameters: Model parameters for L2 regularization
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        batch_size = policy_logits.size(0)
        
        # 1. Value Loss: MSE between predicted value and game outcome
        # (z - v)²
        value_loss = F.mse_loss(value_pred, target_value)
        
        # 2. Policy Loss: Cross-entropy between predicted policy and target policy
        # -π^T log p
        # Convert logits to log probabilities
        log_policy = F.log_softmax(policy_logits, dim=-1)
        
        # Cross-entropy: -sum(target * log(pred))
        policy_loss = -torch.sum(target_policy * log_policy, dim=-1).mean()
        
        # 3. L2 Regularization: c||θ||²
        l2_loss = torch.tensor(0.0, device=policy_logits.device)
        if model_parameters is not None and self.l2_regularization > 0:
            for param in model_parameters:
                l2_loss += torch.sum(param ** 2)
            l2_loss *= self.l2_regularization
        
        # 4. Total Loss
        total_loss = (self.value_loss_weight * value_loss + 
                     self.policy_loss_weight * policy_loss + 
                     l2_loss)
        
        # Loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'l2_loss': l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss
        }
        
        return total_loss, loss_components


class AlphaZeroTrainingLoss:
    """
    Training wrapper for AlphaZero loss with additional utilities
    """
    
    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0, 
                 l2_reg: float = 1e-4):
        self.loss_fn = AlphaZeroLoss(value_weight, policy_weight, l2_reg)
        self.loss_history = []
        
    def compute_loss(self, network_output: Tuple[torch.Tensor, torch.Tensor],
                    targets: Tuple[torch.Tensor, torch.Tensor],
                    model_parameters=None) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss from network output and targets
        
        Args:
            network_output: (policy_logits, value_pred) from network
            targets: (target_policy, target_value) 
            model_parameters: Model parameters for regularization
            
        Returns:
            (loss, loss_components)
        """
        policy_logits, value_pred = network_output
        target_policy, target_value = targets
        
        loss, components = self.loss_fn(
            policy_logits, value_pred, target_policy, target_value, model_parameters
        )
        
        # Store loss history
        self.loss_history.append(components)
        
        return loss, components
    
    def get_recent_losses(self, n: int = 100) -> dict:
        """Get average of recent losses"""
        if not self.loss_history:
            return {}
            
        recent = self.loss_history[-n:]
        avg_losses = {}
        
        for key in recent[0].keys():
            avg_losses[f'avg_{key}'] = sum(loss[key] for loss in recent) / len(recent)
            
        return avg_losses


def create_alphazero_loss(value_weight: float = 1.0, policy_weight: float = 1.0,
                         l2_regularization: float = 1e-4) -> AlphaZeroLoss:
    """
    Factory function to create AlphaZero loss
    
    Args:
        value_weight: Weight for value loss
        policy_weight: Weight for policy loss
        l2_regularization: L2 regularization coefficient
        
    Returns:
        AlphaZero loss instance
    """
    return AlphaZeroLoss(value_weight, policy_weight, l2_regularization)


def convert_game_outcome_to_value(outcome: str, player_color: bool) -> float:
    """
    Convert chess game outcome to value for training
    
    Args:
        outcome: Game result ('1-0', '0-1', '1/2-1/2')
        player_color: True if white, False if black
        
    Returns:
        Value in [-1, 1] from player's perspective
    """
    if outcome == '1/2-1/2':  # Draw
        return 0.0
    elif outcome == '1-0':  # White wins
        return 1.0 if player_color else -1.0
    elif outcome == '0-1':  # Black wins  
        return -1.0 if player_color else 1.0
    else:
        return 0.0  # Unknown outcome, treat as draw
