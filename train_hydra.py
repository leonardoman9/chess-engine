#!/usr/bin/env python3
"""
Hydra-powered training script for chess DQN
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agents.dqn_agent import DQNAgent
from src.training.self_play import SelfPlayTrainer, TrainingConfig
from src.models.dueling_dqn import ModelConfig
from src.utils.exploration import ExplorationConfig
from src.utils.action_utils import get_action_space

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str) -> torch.device:
    """Get PyTorch device from config"""
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_config)
        logger.info(f"Using device: {device}")
    
    return device


def create_model_config(cfg: DictConfig) -> ModelConfig:
    """Create model configuration from Hydra config"""
    return ModelConfig(
        conv_channels=cfg.model.conv_channels,
        hidden_size=cfg.model.hidden_size
    )


def create_exploration_config(cfg: DictConfig) -> ExplorationConfig:
    """Create exploration configuration from Hydra config"""
    return ExplorationConfig(
        strategy_type=cfg.exploration.strategy_type,
        epsilon_start=cfg.exploration.epsilon_start,
        epsilon_end=cfg.exploration.epsilon_end,
        epsilon_decay_steps=cfg.exploration.epsilon_decay_steps,
        decay_type=cfg.exploration.decay_type
    )


def create_training_config(cfg: DictConfig) -> TrainingConfig:
    """Create training configuration from Hydra config"""
    return TrainingConfig(
        max_moves=cfg.training.max_moves,
        game_timeout=cfg.training.game_timeout,
        games_per_episode=cfg.training.games_per_episode,
        training_frequency=cfg.training.training_frequency,
        target_update_frequency=cfg.training.target_update_frequency,
        eval_frequency=cfg.training.eval_frequency,
        eval_games=cfg.training.eval_games,
        stockfish_depth=cfg.training.stockfish_depth,
        stockfish_time=cfg.training.stockfish_time,
        log_frequency=cfg.training.log_frequency,
        checkpoint_frequency=cfg.training.checkpoint_frequency
    )


def save_experiment_config(cfg: DictConfig, save_path: Path):
    """Save experiment configuration to file"""
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Add system information (convert all to strings to avoid serialization issues)
    config_dict['system'] = {
        'python_version': str(sys.version),
        'pytorch_version': str(torch.__version__),
        'cuda_available': bool(torch.cuda.is_available()),
        'device': str(get_device(cfg.device)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save as YAML
    with open(save_path / 'experiment_config.yaml', 'w') as f:
        OmegaConf.save(config_dict, f)
    
    logger.info(f"Experiment configuration saved to {save_path / 'experiment_config.yaml'}")

def save_experiment_info_legacy(cfg: DictConfig, save_path: Path):
    """Save experiment info in legacy JSON format for compatibility"""
    import json
    
    # Create legacy format experiment info
    experiment_info = {
        'experiment': {
            'name': cfg.experiment.name,
            'description': cfg.experiment.description,
            'total_games': cfg.experiment.total_games,
            'timestamp': datetime.now().isoformat()
        },
        'model_config': {
            'conv_channels': list(cfg.model.conv_channels),
            'hidden_size': cfg.model.hidden_size,
            'dropout': cfg.model.dropout,
            'activation': cfg.model.activation
        },
        'exploration_config': {
            'strategy_type': cfg.exploration.strategy_type,
            'epsilon_start': cfg.exploration.epsilon_start,
            'epsilon_end': cfg.exploration.epsilon_end,
            'epsilon_decay_steps': cfg.exploration.epsilon_decay_steps,
            'decay_type': cfg.exploration.decay_type
        },
        'agent_config': {
            'buffer_size': cfg.agent.buffer_size,
            'min_buffer_size': cfg.agent.min_buffer_size,
            'batch_size': cfg.agent.batch_size,
            'learning_rate': cfg.agent.learning_rate,
            'gamma': cfg.agent.gamma,
            'tau': cfg.agent.tau,
            'replay_type': cfg.agent.replay_type
        },
        'training_config': {
            'max_moves': cfg.training.max_moves,
            'game_timeout': cfg.training.game_timeout,
            'games_per_episode': cfg.training.games_per_episode,
            'training_frequency': cfg.training.training_frequency,
            'target_update_frequency': cfg.training.target_update_frequency,
            'eval_frequency': cfg.training.eval_frequency,
            'eval_games': cfg.training.eval_games,
            'stockfish_depth': cfg.training.stockfish_depth,
            'stockfish_time': cfg.training.stockfish_time,
            'log_frequency': cfg.training.log_frequency,
            'checkpoint_frequency': cfg.training.checkpoint_frequency
        },
        'system': {
            'python_version': str(sys.version),
            'pytorch_version': str(torch.__version__),
            'cuda_available': bool(torch.cuda.is_available()),
            'device': str(get_device(cfg.device))
        }
    }
    
    # Save as JSON
    with open(save_path / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    logger.info(f"Legacy experiment info saved to {save_path / 'experiment_info.json'}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device)
    
    # Print configuration
    if cfg.verbose:
        logger.info("Training Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
    
    # Create configurations
    model_config = create_model_config(cfg)
    exploration_config = create_exploration_config(cfg)
    training_config = create_training_config(cfg)
    
    # Create agent
    logger.info("Initializing DQN Agent...")
    agent = DQNAgent(
        model_config={
            'conv_channels': cfg.model.conv_channels,
            'hidden_size': cfg.model.hidden_size
        },
        buffer_size=cfg.agent.buffer_size,
        min_buffer_size=cfg.agent.min_buffer_size,
        batch_size=cfg.agent.batch_size,
        learning_rate=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        buffer_type=cfg.agent.replay_type,  # Map replay_type to buffer_type
        exploration_config={
            'strategy_type': cfg.exploration.strategy_type,
            'epsilon_start': cfg.exploration.epsilon_start,
            'epsilon_end': cfg.exploration.epsilon_end,
            'epsilon_decay_steps': cfg.exploration.epsilon_decay_steps,
            'decay_type': cfg.exploration.decay_type
        },
        device=device
    )
    
    logger.info(f"Agent initialized with {sum(p.numel() for p in agent.q_network.parameters())} parameters")
    
    # Create trainer
    logger.info("Initializing Self-Play Trainer...")
    
    # Get current working directory (Hydra changes it)
    results_dir = Path.cwd()
    
    trainer = SelfPlayTrainer(
        agent=agent,
        config=training_config,
        stockfish_path=os.getenv('STOCKFISH_PATH', '/usr/games/stockfish'),
        log_dir=str(results_dir / "logs"),
        checkpoint_dir=str(results_dir / "checkpoints")
    )
    
    # Save experiment configuration (both new and legacy formats)
    save_experiment_config(cfg, results_dir)
    save_experiment_info_legacy(cfg, results_dir)
    
    # Training
    logger.info(f"Starting training: {cfg.experiment.description}")
    logger.info(f"Total games: {cfg.experiment.total_games}")
    logger.info(f"Results directory: {results_dir}")

    try:
        # Train with save_path to generate final_model.pt
        logger.info("🚀 Starting training phase...")
        final_model_path = results_dir / 'final_model.pt'
        training_history = trainer.train(
            total_games=cfg.experiment.total_games,
            save_path=str(final_model_path)
        )
        
        logger.info("✅ Training phase completed!")
        logger.info("💾 Starting file saving phase...")
        
        # Save training history
        try:
            training_history_path = results_dir / 'training_history.json'
            with open(training_history_path, 'w') as f:
                import json
                json.dump(training_history, f, indent=2, default=str)
            logger.info(f"✅ Training history saved to: {training_history_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save training history: {e}")
        
        # Verify final model was saved
        if final_model_path.exists():
            logger.info(f"✅ Final model confirmed at: {final_model_path}")
        else:
            logger.error(f"❌ Final model NOT found at: {final_model_path}")
        
        logger.info("🎯 Starting final evaluation phase...")
        
        # Final evaluation
        try:
            final_results = trainer.evaluate_against_stockfish(50)
            
            logger.info("🏆 Final Results:")
            logger.info(f"  Win Rate: {final_results['win_rate']:.1%}")
            logger.info(f"  Draw Rate: {final_results['draw_rate']:.1%}")
            logger.info(f"  Loss Rate: {final_results['loss_rate']:.1%}")
        except Exception as e:
            logger.error(f"❌ Final evaluation failed: {e}")
        
        logger.info("🎉 All phases completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
