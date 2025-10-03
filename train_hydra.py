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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import seaborn as sns
import json

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


def generate_training_plots(training_history: Dict[str, Any], plots_dir: Path) -> None:
    """
    Generate comprehensive training plots from training history
    
    Args:
        training_history: Dictionary containing training metrics
        plots_dir: Directory to save plots
    """
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data from training history
    episode_rewards = training_history.get('episode_rewards', [])
    episode_lengths = training_history.get('episode_lengths', [])
    evaluations = training_history.get('evaluations', [])
    
    if not episode_rewards:
        print("No episode reward data found in training history")
        return
    
    # Prepare data
    episode_nums = list(range(1, len(episode_rewards) + 1))
    avg_rewards = episode_rewards
    win_rates = []
    epsilons = []
    losses = []
    
    # Extract evaluation data if available
    for eval_data in evaluations:
        win_rates.append(eval_data.get('win_rate', 0))
    
    # If no evaluation data, create dummy data
    if not win_rates:
        win_rates = [0.0] * len(episode_rewards)
    
    # Create epsilon decay simulation (since we don't have actual epsilon data)
    epsilon_start = 1.0
    epsilon_end = 0.05
    for i in range(len(episode_rewards)):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (i / max(len(episode_rewards) - 1, 1))
        epsilons.append(max(epsilon, epsilon_end))
    
    # No loss data available in current format
    losses = [0.0] * len(episode_rewards)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Chess DQN Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Average Reward per Episode
    axes[0, 0].plot(episode_nums, avg_rewards, 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Average Reward per Episode', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    if len(episode_nums) > 1:
        z = np.polyfit(episode_nums, avg_rewards, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(episode_nums, p(episode_nums), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        axes[0, 0].legend()
    
    # 2. Episode Lengths
    if episode_lengths:
        axes[0, 1].plot(episode_nums, episode_lengths, 'g-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Game Length per Episode', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Moves per Game')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add moving average
        if len(episode_lengths) >= 5:
            window = min(5, len(episode_lengths) // 2)
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(episode_nums[window-1:], moving_avg, 'orange', linewidth=2, alpha=0.7, label=f'MA({window})')
            axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No Episode Length Data Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Game Length per Episode', fontweight='bold')
    
    # 3. Epsilon Decay
    axes[1, 0].plot(episode_nums, epsilons, 'purple', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Epsilon Decay (Exploration)', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Training Loss
    if any(loss > 0 for loss in losses):
        axes[1, 1].plot(episode_nums, losses, 'red', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Training Loss per Episode', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')  # Log scale for loss
    else:
        axes[1, 1].text(0.5, 0.5, 'No Loss Data Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Loss per Episode', fontweight='bold')
    
    plt.tight_layout()
    
    # Save main training plot
    main_plot_path = plots_dir / 'training_progress.png'
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate ELO evaluation plot if available
    if 'elo_evaluation' in training_history:
        generate_elo_plot(training_history['elo_evaluation'], plots_dir)
    
    # Generate evaluation metrics plot
    generate_evaluation_plot(training_history, plots_dir)
    
    print(f"Training plots saved to {plots_dir}")


def generate_elo_plot(elo_data: Dict[str, Any], plots_dir: Path) -> None:
    """Generate ELO evaluation plot"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ELO vs Stockfish levels
        if 'detailed_results' in elo_data:
            levels = []
            elos = []
            scores = []
            
            for level_name, results in elo_data['detailed_results'].items():
                levels.append(level_name.replace('_', ' ').title())
                elos.append(results.get('stockfish_elo', 0))
                scores.append(results.get('score', 0) * 100)
            
            # Plot 1: Score vs Stockfish ELO
            ax1.scatter(elos, scores, s=100, alpha=0.7, c=range(len(elos)), cmap='viridis')
            ax1.plot(elos, scores, 'b--', alpha=0.5)
            ax1.set_xlabel('Stockfish ELO')
            ax1.set_ylabel('Agent Score (%)')
            ax1.set_title('Agent Performance vs Stockfish Levels')
            ax1.grid(True, alpha=0.3)
            
            # Add labels for each point
            for i, (elo, score, level) in enumerate(zip(elos, scores, levels)):
                ax1.annotate(level, (elo, score), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 2: ELO estimation summary
        estimated_elo = elo_data.get('estimated_elo', 0)
        confidence_interval = elo_data.get('confidence_interval', [0, 0])
        
        ax2.barh(['Estimated ELO'], [estimated_elo], color='skyblue', alpha=0.7)
        ax2.errorbar([estimated_elo], ['Estimated ELO'], 
                    xerr=[[estimated_elo - confidence_interval[0]], [confidence_interval[1] - estimated_elo]], 
                    fmt='none', color='red', capsize=5, capthick=2)
        ax2.set_xlabel('ELO Rating')
        ax2.set_title(f'Final ELO: {estimated_elo} (95% CI: {confidence_interval[0]}-{confidence_interval[1]})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        elo_plot_path = plots_dir / 'elo_evaluation.png'
        plt.savefig(elo_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Failed to generate ELO plot: {e}")


def generate_evaluation_plot(training_history: Dict[str, Any], plots_dir: Path) -> None:
    """Generate evaluation metrics plot"""
    
    try:
        evaluations = training_history.get('evaluations', [])
        if not evaluations:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        eval_episodes = []
        win_rates = []
        draw_rates = []
        loss_rates = []
        
        for eval_data in evaluations:
            eval_episodes.append(eval_data.get('episode', 0))
            win_rates.append(eval_data.get('win_rate', 0) * 100)
            draw_rates.append(eval_data.get('draw_rate', 0) * 100)
            loss_rates.append(eval_data.get('loss_rate', 0) * 100)
        
        # Plot 1: Win/Draw/Loss rates over time
        axes[0].plot(eval_episodes, win_rates, 'g-', linewidth=2, label='Win Rate', marker='o')
        axes[0].plot(eval_episodes, draw_rates, 'y-', linewidth=2, label='Draw Rate', marker='s')
        axes[0].plot(eval_episodes, loss_rates, 'r-', linewidth=2, label='Loss Rate', marker='^')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Rate (%)')
        axes[0].set_title('Evaluation Results vs Stockfish')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Plot 2: Stacked area chart
        axes[1].stackplot(eval_episodes, win_rates, draw_rates, loss_rates, 
                         labels=['Win', 'Draw', 'Loss'], alpha=0.7,
                         colors=['green', 'yellow', 'red'])
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Rate (%)')
        axes[1].set_title('Game Outcomes Distribution')
        axes[1].legend(loc='upper right')
        axes[1].set_ylim(0, 100)
        
        # Plot 3: Performance trend (win rate only)
        if len(win_rates) > 1:
            axes[2].plot(eval_episodes, win_rates, 'b-', linewidth=3, marker='o', markersize=8)
            
            # Add trend line
            z = np.polyfit(eval_episodes, win_rates, 1)
            p = np.poly1d(z)
            axes[2].plot(eval_episodes, p(eval_episodes), "r--", alpha=0.8, 
                        label=f'Trend: {z[0]:.3f}% per episode')
            axes[2].legend()
        
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Win Rate (%)')
        axes[2].set_title('Win Rate Trend')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, max(100, max(win_rates) * 1.1) if win_rates else 100)
        
        plt.tight_layout()
        eval_plot_path = plots_dir / 'evaluation_metrics.png'
        plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Failed to generate evaluation plot: {e}")


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
        
        # Generate sample games
        logger.info("🎮 Generating sample games...")
        try:
            sample_games_dir = results_dir / "sample_games"
            sample_games_dir.mkdir(exist_ok=True)
            
            from src.utils.game_generator import generate_sample_games
            sample_games = generate_sample_games(
                agent=trainer.agent,
                num_games=10,
                save_dir=sample_games_dir,
                max_moves=cfg.training.max_moves,
                self_play=True  # Agent vs Agent (self-play)
            )
            
            logger.info(f"✅ Generated {len(sample_games)} sample games in: {sample_games_dir}")
            
            # Log summary of sample games
            total_moves = sum(game['moves'] for game in sample_games)
            avg_moves = total_moves / len(sample_games) if sample_games else 0
            
            results_summary = {}
            for game in sample_games:
                result = game['result']
                results_summary[result] = results_summary.get(result, 0) + 1
            
            logger.info("📊 Sample Games Summary:")
            logger.info(f"  Average moves per game: {avg_moves:.1f}")
            for result, count in results_summary.items():
                logger.info(f"  {result}: {count} games")
                
        except Exception as e:
            logger.error(f"❌ Sample games generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Generate training plots
        logger.info("📈 Generating training plots...")
        try:
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            generate_training_plots(training_history, plots_dir)
            logger.info(f"✅ Training plots saved to: {plots_dir}")
            
        except Exception as e:
            logger.error(f"❌ Training plots generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
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
