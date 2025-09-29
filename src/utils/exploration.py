"""
Exploration strategies for DQN training
Implements epsilon-greedy with various decay schedules
"""

import torch
import numpy as np
import math
from typing import Union, Optional


class EpsilonGreedyExploration:
    """
    Epsilon-greedy exploration with configurable decay schedules
    """
    
    def __init__(self, 
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 1000000,
                 decay_type: str = 'linear'):
        """
        Initialize epsilon-greedy exploration
        
        Args:
            epsilon_start: Initial epsilon value
            epsilon_end: Final epsilon value  
            epsilon_decay_steps: Number of steps for decay
            decay_type: Type of decay ('linear', 'exponential', 'cosine')
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.decay_type = decay_type
        
        self.current_step = 0
        self.current_epsilon = epsilon_start
        
        # Pre-compute decay parameters
        if decay_type == 'exponential':
            self.decay_rate = (epsilon_end / epsilon_start) ** (1.0 / epsilon_decay_steps)
        elif decay_type == 'cosine':
            self.decay_amplitude = (epsilon_start - epsilon_end) / 2
            self.decay_center = (epsilon_start + epsilon_end) / 2
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        """
        Get current epsilon value
        
        Args:
            step: Optional step override (if None, uses internal counter)
            
        Returns:
            Current epsilon value
        """
        if step is None:
            step = self.current_step
        
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        progress = step / self.epsilon_decay_steps
        
        if self.decay_type == 'linear':
            epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        elif self.decay_type == 'exponential':
            epsilon = self.epsilon_start * (self.decay_rate ** step)
        elif self.decay_type == 'cosine':
            epsilon = self.decay_center + self.decay_amplitude * math.cos(math.pi * progress)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return max(epsilon, self.epsilon_end)
    
    def update(self) -> float:
        """
        Update internal step counter and return new epsilon
        
        Returns:
            Updated epsilon value
        """
        self.current_epsilon = self.get_epsilon(self.current_step)
        self.current_step += 1
        return self.current_epsilon
    
    def select_action(self, 
                     q_values: torch.Tensor, 
                     legal_actions_mask: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy strategy
        
        Args:
            q_values: Q-values for all actions [action_size]
            legal_actions_mask: Boolean mask for legal actions [action_size]
            deterministic: If True, always select greedy action (for evaluation)
            
        Returns:
            Selected action index
        """
        if deterministic or np.random.random() > self.current_epsilon:
            # Greedy action selection
            if legal_actions_mask is not None:
                masked_q_values = q_values.masked_fill(~legal_actions_mask, -float('inf'))
                return torch.argmax(masked_q_values).item()
            else:
                return torch.argmax(q_values).item()
        else:
            # Random action selection
            if legal_actions_mask is not None:
                legal_actions = torch.nonzero(legal_actions_mask, as_tuple=False).squeeze(1)
                if len(legal_actions) > 0:
                    return legal_actions[torch.randint(len(legal_actions), (1,))].item()
                else:
                    return 0  # Fallback
            else:
                return torch.randint(len(q_values), (1,)).item()
    
    def batch_select_actions(self,
                           q_values: torch.Tensor,
                           legal_actions_masks: Optional[torch.Tensor] = None,
                           deterministic: bool = False) -> torch.Tensor:
        """
        Select actions for a batch using epsilon-greedy strategy
        
        Args:
            q_values: Q-values [batch_size, action_size]
            legal_actions_masks: Boolean masks [batch_size, action_size]
            deterministic: If True, always select greedy actions
            
        Returns:
            Selected actions [batch_size]
        """
        batch_size = q_values.size(0)
        actions = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            mask = legal_actions_masks[i] if legal_actions_masks is not None else None
            actions[i] = self.select_action(q_values[i], mask, deterministic)
        
        return actions
    
    def reset(self):
        """Reset exploration to initial state"""
        self.current_step = 0
        self.current_epsilon = self.epsilon_start
    
    def set_step(self, step: int):
        """Set current step (useful for resuming training)"""
        self.current_step = step
        self.current_epsilon = self.get_epsilon(step)
    
    def get_state(self) -> dict:
        """Get exploration state for checkpointing"""
        return {
            'current_step': self.current_step,
            'current_epsilon': self.current_epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'decay_type': self.decay_type
        }
    
    def load_state(self, state: dict):
        """Load exploration state from checkpoint"""
        self.current_step = state['current_step']
        self.current_epsilon = state['current_epsilon']
        self.epsilon_start = state['epsilon_start']
        self.epsilon_end = state['epsilon_end']
        self.epsilon_decay_steps = state['epsilon_decay_steps']
        self.decay_type = state['decay_type']


class AdaptiveEpsilonGreedy(EpsilonGreedyExploration):
    """
    Adaptive epsilon-greedy that adjusts based on performance
    """
    
    def __init__(self, 
                 adaptation_window: int = 10000,
                 performance_threshold: float = 0.1,
                 **kwargs):
        """
        Initialize adaptive epsilon-greedy
        
        Args:
            adaptation_window: Window size for performance evaluation
            performance_threshold: Threshold for epsilon adjustment
            **kwargs: Arguments for base EpsilonGreedyExploration
        """
        super().__init__(**kwargs)
        self.adaptation_window = adaptation_window
        self.performance_threshold = performance_threshold
        self.performance_history = []
        self.base_epsilon_end = self.epsilon_end
    
    def update_performance(self, reward: float):
        """
        Update performance history and adapt epsilon if needed
        
        Args:
            reward: Recent reward/performance metric
        """
        self.performance_history.append(reward)
        
        # Keep only recent history
        if len(self.performance_history) > self.adaptation_window:
            self.performance_history.pop(0)
        
        # Adapt epsilon based on recent performance
        if len(self.performance_history) >= self.adaptation_window:
            recent_performance = np.mean(self.performance_history[-self.adaptation_window//2:])
            older_performance = np.mean(self.performance_history[:self.adaptation_window//2])
            
            performance_improvement = recent_performance - older_performance
            
            if performance_improvement < self.performance_threshold:
                # Poor performance - increase exploration
                self.epsilon_end = min(self.base_epsilon_end * 2, 0.2)
            else:
                # Good performance - maintain or reduce exploration
                self.epsilon_end = self.base_epsilon_end


class MultiStageExploration:
    """
    Multi-stage exploration with different strategies for different training phases
    """
    
    def __init__(self, stages: list):
        """
        Initialize multi-stage exploration
        
        Args:
            stages: List of (duration, exploration_strategy) tuples
        """
        self.stages = stages
        self.current_stage = 0
        self.stage_start_step = 0
        self.total_steps = 0
        
        # Initialize first stage
        self.current_exploration = self.stages[0][1]
    
    def update(self) -> float:
        """Update exploration strategy and get current epsilon"""
        self.total_steps += 1
        
        # Check if we should move to next stage
        if self.current_stage < len(self.stages) - 1:
            stage_duration, _ = self.stages[self.current_stage]
            
            if self.total_steps - self.stage_start_step >= stage_duration:
                self.current_stage += 1
                self.stage_start_step = self.total_steps
                self.current_exploration = self.stages[self.current_stage][1]
                print(f"Switching to exploration stage {self.current_stage}")
        
        return self.current_exploration.update()
    
    def select_action(self, *args, **kwargs):
        """Delegate action selection to current exploration strategy"""
        return self.current_exploration.select_action(*args, **kwargs)
    
    def get_epsilon(self):
        """Get current epsilon value"""
        return self.current_exploration.current_epsilon


def create_exploration_strategy(strategy_type: str = 'linear', **kwargs) -> EpsilonGreedyExploration:
    """
    Factory function to create exploration strategies
    
    Args:
        strategy_type: Type of exploration strategy
        **kwargs: Additional arguments
        
    Returns:
        Exploration strategy instance
    """
    if strategy_type == 'linear':
        return EpsilonGreedyExploration(decay_type='linear', **kwargs)
    elif strategy_type == 'exponential':
        return EpsilonGreedyExploration(decay_type='exponential', **kwargs)
    elif strategy_type == 'cosine':
        return EpsilonGreedyExploration(decay_type='cosine', **kwargs)
    elif strategy_type == 'adaptive':
        return AdaptiveEpsilonGreedy(**kwargs)
    else:
        raise ValueError(f"Unknown exploration strategy: {strategy_type}")


# Preset exploration configurations
EXPLORATION_CONFIGS = {
    'aggressive': {
        'strategy_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 500000,
    },
    'conservative': {
        'strategy_type': 'exponential',
        'epsilon_start': 0.5,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': 2000000,
    },
    'standard': {
        'strategy_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 1000000,
    }
}


if __name__ == "__main__":
    # Test exploration strategies
    import matplotlib.pyplot as plt
    
    # Test different decay types
    strategies = {
        'Linear': EpsilonGreedyExploration(decay_type='linear'),
        'Exponential': EpsilonGreedyExploration(decay_type='exponential'),
        'Cosine': EpsilonGreedyExploration(decay_type='cosine')
    }
    
    steps = range(0, 1000000, 10000)
    
    plt.figure(figsize=(12, 6))
    
    for name, strategy in strategies.items():
        epsilons = [strategy.get_epsilon(step) for step in steps]
        plt.plot(steps, epsilons, label=name)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Test action selection
    strategy = EpsilonGreedyExploration()
    q_values = torch.randn(10)
    legal_mask = torch.tensor([True, False, True, True, False, True, True, False, True, True])
    
    print("Testing action selection:")
    for i in range(5):
        action = strategy.select_action(q_values, legal_mask)
        epsilon = strategy.update()
        print(f"Step {i}: Action {action}, Epsilon {epsilon:.4f}")
