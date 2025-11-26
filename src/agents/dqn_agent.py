"""
DQN Agent Implementation
Main agent class that combines all DQN components: network, replay buffer, exploration, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
from typing import Tuple, Optional, Dict, Any
import copy
import os

from ..models.dueling_dqn import DuelingDQN, create_dqn_model
from ..replay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_replay_buffer
from ..utils.action_utils import ChessActionSpace, board_to_tensor, BatchActionMasker
from ..utils.exploration import EpsilonGreedyExploration, create_exploration_strategy


class DQNAgent:
    """
    Deep Q-Network Agent for Chess
    
    Combines all components: Q-network, target network, replay buffer, exploration strategy
    """
    
    def __init__(self,
                 # Network configuration
                 model_type: str = 'cnn',
                 model_config: dict = None,
                 
                 # Training configuration
                 learning_rate: float = 1e-4,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,  # Soft update coefficient
                 
                 # Replay buffer configuration
                 buffer_type: str = 'standard',
                 buffer_size: int = 500000,
                 min_buffer_size: int = 10000,
                 
                 # Exploration configuration
                 exploration_config: dict = None,
                 
                 # Device configuration
                 device: str = None):
        """
        Initialize DQN Agent
        
        Args:
            model_type: Type of model ('cnn', 'equivariant')
            model_config: Model configuration dictionary
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update coefficient for target network
            buffer_type: Type of replay buffer ('standard', 'prioritized')
            buffer_size: Size of replay buffer
            min_buffer_size: Minimum buffer size before training
            exploration_config: Exploration strategy configuration
            device: Device to run on ('cpu', 'cuda')
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent initialized on device: {self.device}")
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.min_buffer_size = min_buffer_size
        
        # Action space
        self.action_space = ChessActionSpace()
        self.action_masker = BatchActionMasker(self.action_space)
        
        # Create networks
        model_config = model_config or {}
        self.q_network = create_dqn_model(model_type, **model_config).to(self.device)
        self.target_network = create_dqn_model(model_type, **model_config).to(self.device)
        
        # Initialize target network
        self.hard_update_target_network()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Replay buffer
        input_channels = model_config.get("input_channels", 13)
        buffer_config = {
            'capacity': buffer_size,
            'state_shape': (input_channels, 8, 8),
            'device': self.device
        }
        self.replay_buffer = create_replay_buffer(buffer_type, **buffer_config)
        self.buffer_type = buffer_type
        
        # Exploration strategy
        exploration_config = exploration_config or {'strategy_type': 'standard'}
        # Temperature scheduling (optional, removed from exploration kwargs before creation)
        temp_start = exploration_config.pop('temperature_start', 1.0)
        temp_end = exploration_config.pop('temperature_end', temp_start)
        temp_decay_steps = exploration_config.pop('temperature_decay_steps', 0)

        self.exploration = create_exploration_strategy(**exploration_config)

        self.temperature_start = temp_start
        self.temperature_end = temp_end
        self.temp_decay_steps = temp_decay_steps
        self.temperature = self.temperature_start  # softmax temperature for action sampling
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        print(f"Q-Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"Replay buffer type: {buffer_type}, capacity: {buffer_size:,}")
    
    def select_action(self, 
                     board: chess.Board, 
                     deterministic: bool = False) -> int:
        """
        Select action for given board state
        
        Args:
            board: Current chess board
            deterministic: If True, select greedy action (for evaluation)
            
        Returns:
            Selected action index
        """
        # Update exploration step for training (one step per action)
        if not deterministic:
            self.exploration.update()
            # Temperature annealing (optional)
            if self.temp_decay_steps and self.temp_decay_steps > 0:
                step = getattr(self.exploration, "current_step", 0)
                frac = min(step / self.temp_decay_steps, 1.0)
                self.temperature = self.temperature_start + frac * (self.temperature_end - self.temperature_start)
        
        # Convert board to tensor
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        
        # Get legal actions mask
        legal_mask = self.action_space.get_legal_actions_mask(board).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state, legal_mask.unsqueeze(0))
            q_values = q_values.squeeze(0)
        
        epsilon = getattr(self.exploration, "current_epsilon", self.exploration.get_epsilon())

        if deterministic:
            masked_q = q_values.masked_fill(~legal_mask, -float('inf'))
            action = torch.argmax(masked_q).item()
        else:
            if np.random.random() < epsilon:
                legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(1)
                action = np.random.choice(legal_indices.cpu().numpy())
            else:
                masked_q = q_values.masked_fill(~legal_mask, -float('inf'))
                # Focus on top-k legal actions to reduce noise
                top_k = 10
                legal_indices = masked_q.topk(k=min(top_k, legal_mask.sum().item()), dim=0).indices
                sub_q = masked_q[legal_indices]
                logits = sub_q / max(self.temperature, 1e-3)
                probs = torch.softmax(logits, dim=0)
                probs_np = probs.cpu().numpy()
                probs_np = probs_np / probs_np.sum() if probs_np.sum() > 0 else np.ones_like(probs_np) / len(probs_np)
                chosen_idx = np.random.choice(len(probs_np), p=probs_np)
                action = legal_indices[chosen_idx].item()
        
        return action
    
    def act(self, board: chess.Board, deterministic: bool = False) -> chess.Move:
        """
        Select and return chess move for given board state
        
        Args:
            board: Current chess board
            deterministic: If True, select greedy action
            
        Returns:
            Chess move object
        """
        action_idx = self.select_action(board, deterministic)
        move_uci = self.action_space.action_to_move(action_idx)
        
        if move_uci is None:
            # Fallback: select random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return np.random.choice(legal_moves)
            else:
                return None
        
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
            else:
                # Fallback: select random legal move
                legal_moves = list(board.legal_moves)
                return np.random.choice(legal_moves) if legal_moves else None
        except:
            # Fallback: select random legal move
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
    
    def remember(self, 
                state: torch.Tensor,
                action: int,
                reward: float, 
                next_state: torch.Tensor,
                done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step
        
        Returns:
            Dictionary with training metrics
        """
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return {}
        
        # Sample batch from replay buffer
        if self.buffer_type == 'prioritized':
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, weights, indices = batch
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones_like(rewards)
            indices = None
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            # Double DQN: use main network to select actions, target network to evaluate
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.buffer_type == 'prioritized' and indices is not None:
            priorities = td_errors.abs().detach() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Soft update target network
        self.soft_update_target_network()
        
        self.training_step += 1
        
        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'min_q_value': current_q_values.min().item(),
            'max_q_value': current_q_values.max().item(),
            'epsilon': self.exploration.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step
        }
    
    def soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, main_param in zip(self.target_network.parameters(), 
                                          self.q_network.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update_target_network(self):
        """Hard update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_state': self.exploration.get_state(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            
            # Configuration
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'buffer_type': self.buffer_type,
        }
        
        torch.save(checkpoint, filepath)
        
        # Also save replay buffer separately (optional)
        buffer_path = filepath.replace('.pt', '_buffer.pt')
        self.replay_buffer.save(buffer_path)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_buffer: bool = True):
        """
        Load agent checkpoint
        
        Args:
            filepath: Path to checkpoint file
            load_buffer: Whether to load replay buffer
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration.load_state(checkpoint['exploration_state'])
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.total_reward = checkpoint['total_reward']
        
        # Load replay buffer if requested
        if load_buffer:
            buffer_path = filepath.replace('.pt', '_buffer.pt')
            if os.path.exists(buffer_path):
                self.replay_buffer.load(buffer_path)
                print(f"Loaded replay buffer from {buffer_path}")
        
        print(f"Checkpoint loaded from {filepath}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.q_network.eval()
        
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        for episode in range(num_episodes):
            board = chess.Board()
            moves = 0
            
            while not board.is_game_over() and moves < 200:  # Max 200 moves
                # Agent's move
                move = self.act(board, deterministic=True)
                if move is None:
                    break
                board.push(move)
                moves += 1
                
                if board.is_game_over():
                    break
                
                # Random opponent move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(np.random.choice(legal_moves))
                    moves += 1
            
            # Determine result
            if board.is_game_over():
                result = board.result()
                if result == "1-0":  # White (agent) wins
                    wins += 1
                elif result == "0-1":  # Black wins
                    losses += 1
                else:  # Draw
                    draws += 1
            else:
                draws += 1  # Timeout = draw
            
            total_moves += moves
        
        self.q_network.train()
        
        return {
            'win_rate': wins / num_episodes,
            'draw_rate': draws / num_episodes, 
            'loss_rate': losses / num_episodes,
            'avg_moves': total_moves / num_episodes
        }
    
    def get_state_value(self, board: chess.Board) -> float:
        """
        Get state value estimate for given board
        
        Args:
            board: Chess board state
            
        Returns:
            State value estimate
        """
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        legal_mask = self.action_space.get_legal_actions_mask(board).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state, legal_mask.unsqueeze(0))
            return q_values.max().item()


if __name__ == "__main__":
    # Test DQN Agent
    agent = DQNAgent(
        model_type='cnn',
        model_config={'conv_channels': [64, 128, 256], 'hidden_size': 512},
        buffer_type='standard',
        buffer_size=10000
    )
    
    # Test with starting position
    board = chess.Board()
    
    # Test action selection
    action = agent.select_action(board)
    print(f"Selected action: {action}")
    
    # Test move generation
    move = agent.act(board)
    print(f"Selected move: {move}")
    
    # Test state value
    value = agent.get_state_value(board)
    print(f"State value: {value}")
    
    # Test remember and training (with dummy data)
    for i in range(100):
        state = board_to_tensor(board)
        action = agent.select_action(board)
        reward = np.random.randn()
        next_state = board_to_tensor(board)  # Same state for simplicity
        done = False
        
        agent.remember(state, action, reward, next_state, done)
    
    # Test training step
    if agent.replay_buffer.is_ready(agent.min_buffer_size):
        metrics = agent.train_step()
        print(f"Training metrics: {metrics}")
    
    print("DQN Agent test completed successfully!")
