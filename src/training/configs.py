"""
Training configurations for different scenarios
"""
from dataclasses import dataclass
from typing import Dict, Any

from .self_play import TrainingConfig
from ..models.dueling_dqn import MODEL_CONFIGS
from ..utils.exploration import EXPLORATION_CONFIGS


# Training configurations for different scenarios
TRAINING_CONFIGS = {
    'quick_test': TrainingConfig(
        # Game settings
        max_moves=50,
        game_timeout=10.0,
        
        # Training settings
        games_per_episode=5,
        training_frequency=2,
        target_update_frequency=20,
        
        # Evaluation settings
        eval_frequency=10,
        eval_games=3,
        stockfish_depth=1,
        stockfish_time=0.05,
        
        # Logging
        log_frequency=5,
        checkpoint_frequency=20
    ),
    
    'development': TrainingConfig(
        # Game settings
        max_moves=100,
        game_timeout=20.0,
        
        # Training settings
        games_per_episode=10,
        training_frequency=5,
        target_update_frequency=50,
        
        # Evaluation settings
        eval_frequency=25,
        eval_games=5,
        stockfish_depth=1,
        stockfish_time=0.1,
        
        # Logging
        log_frequency=10,
        checkpoint_frequency=50
    ),
    
    'production': TrainingConfig(
        # Game settings
        max_moves=200,
        game_timeout=60.0,
        
        # Training settings
        games_per_episode=20,
        training_frequency=10,
        target_update_frequency=100,
        
        # Evaluation settings
        eval_frequency=100,
        eval_games=20,
        stockfish_depth=1,
        stockfish_time=0.2,
        
        # Logging
        log_frequency=20,
        checkpoint_frequency=200
    ),
    
    'server_intensive': TrainingConfig(
        # Game settings
        max_moves=300,
        game_timeout=120.0,
        
        # Training settings
        games_per_episode=50,
        training_frequency=20,
        target_update_frequency=200,
        
        # Evaluation settings
        eval_frequency=200,
        eval_games=50,
        stockfish_depth=2,
        stockfish_time=0.5,
        
        # Logging
        log_frequency=50,
        checkpoint_frequency=500
    )
}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    description: str
    model_config: Dict[str, Any]
    exploration_config: Dict[str, Any]
    training_config: TrainingConfig
    agent_config: Dict[str, Any]
    total_games: int
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.name, "Experiment name cannot be empty"
        assert self.total_games > 0, "Total games must be positive"


# Predefined experiments
EXPERIMENTS = {
    'baseline_small': ExperimentConfig(
        name='baseline_small',
        description='Baseline CNN-DQN with small model for quick testing',
        model_config=MODEL_CONFIGS['small'],
        exploration_config=EXPLORATION_CONFIGS['standard'],
        training_config=TRAINING_CONFIGS['quick_test'],
        agent_config={
            'buffer_size': 1000,
            'min_buffer_size': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'tau': 0.005
        },
        total_games=100
    ),
    
    'baseline_medium': ExperimentConfig(
        name='baseline_medium',
        description='Baseline CNN-DQN with medium model for development',
        model_config=MODEL_CONFIGS['medium'],
        exploration_config=EXPLORATION_CONFIGS['standard'],
        training_config=TRAINING_CONFIGS['development'],
        agent_config={
            'buffer_size': 5000,
            'min_buffer_size': 500,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'tau': 0.005
        },
        total_games=500
    ),
    
    'baseline_large': ExperimentConfig(
        name='baseline_large',
        description='Baseline CNN-DQN with large model for production',
        model_config=MODEL_CONFIGS['large'],
        exploration_config=EXPLORATION_CONFIGS['conservative'],
        training_config=TRAINING_CONFIGS['production'],
        agent_config={
            'buffer_size': 20000,
            'min_buffer_size': 2000,
            'batch_size': 128,
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'tau': 0.001
        },
        total_games=2000
    ),
    
    'server_experiment': ExperimentConfig(
        name='server_experiment',
        description='Intensive training for A40 server',
        model_config=MODEL_CONFIGS['large'],
        exploration_config=EXPLORATION_CONFIGS['aggressive'],
        training_config=TRAINING_CONFIGS['server_intensive'],
        agent_config={
            'buffer_size': 100000,
            'min_buffer_size': 10000,
            'batch_size': 256,
            'learning_rate': 1e-4,
            'gamma': 0.995,
            'tau': 0.001
        },
        total_games=10000
    )
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name"""
    if name not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return EXPERIMENTS[name]


def list_experiments() -> Dict[str, str]:
    """List all available experiments with descriptions"""
    return {name: config.description for name, config in EXPERIMENTS.items()}
