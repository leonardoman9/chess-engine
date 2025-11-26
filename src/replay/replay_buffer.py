"""
Experience Replay Buffer for DQN
Implements circular buffer with optional prioritized experience replay
"""

import torch
import numpy as np
import random
from typing import Tuple, List, Optional, NamedTuple
from collections import deque
import pickle


class Experience(NamedTuple):
    """Single experience tuple"""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """
    Standard experience replay buffer with circular storage
    """
    
    def __init__(self, 
                 capacity: int = 500000,
                 state_shape: Tuple[int, ...] = (15, 8, 8),
                 device: str = 'cpu'):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
            state_shape: Shape of state tensors
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        
        # Pre-allocate memory for efficiency
        self.states = torch.zeros((capacity,) + state_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity,) + state_shape, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.position = 0
        self.size = 0
    
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Store experience at current position
        self.states[self.position] = state.to(self.device)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.to(self.device)
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences to sample. Have {self.size}, need {batch_size}")
        
        # Sample random indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training"""
        return self.size >= min_size
    
    def clear(self):
        """Clear buffer"""
        self.position = 0
        self.size = 0
    
    def save(self, filepath: str):
        """Save buffer to file"""
        state = {
            'states': self.states[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'rewards': self.rewards[:self.size].cpu(),
            'next_states': self.next_states[:self.size].cpu(),
            'dones': self.dones[:self.size].cpu(),
            'position': self.position,
            'size': self.size,
            'capacity': self.capacity,
            'state_shape': self.state_shape
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load buffer from file"""
        state = torch.load(filepath, map_location=self.device)
        
        self.capacity = state['capacity']
        self.state_shape = state['state_shape']
        self.position = state['position']
        self.size = state['size']
        
        # Reallocate memory if needed
        if self.states.shape[0] != self.capacity:
            self.states = torch.zeros((self.capacity,) + self.state_shape, 
                                    dtype=torch.float32, device=self.device)
            self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
            self.next_states = torch.zeros((self.capacity,) + self.state_shape, 
                                         dtype=torch.float32, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        
        # Load data
        self.states[:self.size] = state['states'].to(self.device)
        self.actions[:self.size] = state['actions'].to(self.device)
        self.rewards[:self.size] = state['rewards'].to(self.device)
        self.next_states[:self.size] = state['next_states'].to(self.device)
        self.dones[:self.size] = state['dones'].to(self.device)


class SumTree:
    """
    Sum Tree data structure for Prioritized Experience Replay
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree
        
        Args:
            capacity: Maximum number of experiences
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
    
    def add(self, priority: float, data_index: int):
        """Add priority to tree"""
        tree_index = data_index + self.capacity - 1
        self.update(tree_index, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, tree_index: int, priority: float):
        """Update priority at tree index"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Update parent nodes
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """Get leaf node for given value"""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index
    
    @property
    def total_priority(self) -> float:
        """Get total priority (root of tree)"""
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer
    """
    
    def __init__(self, 
                 capacity: int = 500000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 **kwargs):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
            epsilon: Small constant to prevent zero priorities
            **kwargs: Additional arguments for base ReplayBuffer
        """
        super().__init__(capacity=capacity, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Sum tree for efficient priority sampling
        self.tree = SumTree(capacity)
        
        # Track max priority for new experiences
        self.max_priority = 1.0
    
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool,
             priority: Optional[float] = None):
        """
        Add experience with priority
        
        Args:
            priority: Experience priority (if None, uses max priority)
        """
        # Add to replay buffer
        super().push(state, action, reward, next_state, done)
        
        # Add to sum tree with priority
        if priority is None:
            priority = self.max_priority
        
        self.tree.add(priority ** self.alpha, (self.position - 1) % self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with importance sampling weights
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences to sample. Have {self.size}, need {batch_size}")
        
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample experiences
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            value = np.random.uniform(a, b)
            tree_index, priority, data_index = self.tree.get_leaf(value)
            
            indices.append(data_index)
            priorities.append(priority)
        
        # Convert to tensors
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority
        weights = (self.size * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return (
            self.states[indices],
            self.actions[indices], 
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices
        )
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priorities for given indices
        
        Args:
            indices: Experience indices
            priorities: New priorities (TD errors)
        """
        priorities = priorities.cpu().numpy()
        indices = indices.cpu().numpy()
        
        for i, priority in zip(indices, priorities):
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(i + self.capacity - 1, priority)
            self.max_priority = max(self.max_priority, priority)


class MultiStepReplayBuffer(ReplayBuffer):
    """
    Multi-step replay buffer for n-step learning
    """
    
    def __init__(self, n_step: int = 3, gamma: float = 0.99, **kwargs):
        """
        Initialize multi-step replay buffer
        
        Args:
            n_step: Number of steps for multi-step learning
            gamma: Discount factor
            **kwargs: Additional arguments for base ReplayBuffer
        """
        super().__init__(**kwargs)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """Add experience to n-step buffer"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            # Get first state and last next_state
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            # Add to main buffer
            super().push(first_state, first_action, n_step_reward, last_next_state, last_done)


def create_replay_buffer(buffer_type: str = 'standard', **kwargs) -> ReplayBuffer:
    """
    Factory function to create replay buffers
    
    Args:
        buffer_type: Type of buffer ('standard', 'prioritized', 'multistep')
        **kwargs: Additional arguments
        
    Returns:
        Replay buffer instance
    """
    if buffer_type == 'standard':
        return ReplayBuffer(**kwargs)
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(**kwargs)
    elif buffer_type == 'multistep':
        return MultiStepReplayBuffer(**kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


if __name__ == "__main__":
    # Test replay buffer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer = ReplayBuffer(capacity=1000, device=device)
    
    # Add some experiences
    for i in range(100):
        state = torch.randn(13, 8, 8)
        action = i % 64
        reward = np.random.randn()
        next_state = torch.randn(13, 8, 8)
        done = i % 20 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample batch
    if buffer.is_ready(32):
        batch = buffer.sample(32)
        print(f"Batch shapes: {[x.shape for x in batch]}")
    
    # Test prioritized buffer
    print("\nTesting prioritized buffer:")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000, device=device)
    
    for i in range(100):
        state = torch.randn(13, 8, 8)
        action = i % 64
        reward = np.random.randn()
        next_state = torch.randn(13, 8, 8)
        done = i % 20 == 0
        
        pri_buffer.push(state, action, reward, next_state, done)
    
    if pri_buffer.is_ready(32):
        batch = pri_buffer.sample(32)
        print(f"Prioritized batch length: {len(batch)}")
        print(f"Weights shape: {batch[5].shape}")
        print(f"Indices shape: {batch[6].shape}")
