#!/usr/bin/env python3
"""
Main training script for Chess DQN
"""
import argparse
import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agents.dqn_agent import DQNAgent
from src.training.self_play import SelfPlayTrainer
from src.training.configs import get_experiment_config, list_experiments, EXPERIMENTS


def setup_directories():
    """Create necessary directories"""
    dirs = ['logs', 'checkpoints', 'results', 'data/training']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_experiment_info(config, save_dir: Path):
    """Save experiment configuration and system info"""
    info = {
        'experiment': {
            'name': config.name,
            'description': config.description,
            'total_games': config.total_games,
            'timestamp': datetime.now().isoformat()
        },
        'model_config': config.model_config,
        'exploration_config': config.exploration_config,
        'agent_config': config.agent_config,
        'training_config': config.training_config.__dict__,
        'system': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
        }
    }
    
    with open(save_dir / 'experiment_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description='Train Chess DQN')
    parser.add_argument(
        'experiment', 
        help='Experiment name',
        choices=list(EXPERIMENTS.keys())
    )
    parser.add_argument(
        '--device', 
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    parser.add_argument(
        '--stockfish-path',
        default='/usr/games/stockfish',
        help='Path to Stockfish executable'
    )
    parser.add_argument(
        '--resume',
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List available experiments and exit'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Setup everything but don\'t start training'
    )
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list_experiments:
        print("Available experiments:")
        for name, description in list_experiments().items():
            print(f"  {name}: {description}")
        return
    
    # Setup directories
    setup_directories()
    
    # Get experiment configuration
    config = get_experiment_config(args.experiment)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting experiment: {config.name}")
    print(f"📝 Description: {config.description}")
    print(f"💻 Device: {device}")
    print(f"🎮 Total games: {config.total_games}")
    print(f"🐟 Stockfish path: {args.stockfish_path}")
    
    # Create output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_dir) / f"{config.name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {experiment_dir}")
    
    # Save experiment info
    save_experiment_info(config, experiment_dir)
    
    # Create DQN agent
    print("🧠 Initializing DQN Agent...")
    agent = DQNAgent(
        model_config=config.model_config,
        exploration_config=config.exploration_config,
        device=device,
        **config.agent_config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)
    
    # Create trainer
    print("🏋️ Initializing Self-Play Trainer...")
    trainer = SelfPlayTrainer(
        agent=agent,
        config=config.training_config,
        stockfish_path=args.stockfish_path,
        log_dir=str(experiment_dir / 'logs'),
        checkpoint_dir=str(experiment_dir / 'checkpoints')
    )
    
    if args.dry_run:
        print("🔍 Dry run complete - everything looks good!")
        print(f"Agent parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
        print(f"Replay buffer capacity: {agent.replay_buffer.capacity:,}")
        print(f"Training will save to: {experiment_dir}")
        return
    
    # Start training
    print("🎯 Starting training...")
    try:
        history = trainer.train(
            total_games=config.total_games,
            save_path=str(experiment_dir / 'final_model.pt')
        )
        
        # Save training history
        with open(experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print("✅ Training completed successfully!")
        print(f"📊 Results saved to: {experiment_dir}")
        
        # Print final statistics
        if history.get('final_evaluation'):
            eval_results = history['final_evaluation']
            print(f"🏆 Final Performance vs Stockfish:")
            print(f"   Win Rate: {eval_results['win_rate']:.1%}")
            print(f"   Draw Rate: {eval_results['draw_rate']:.1%}")
            print(f"   Loss Rate: {eval_results['loss_rate']:.1%}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save current state
        checkpoint_path = experiment_dir / 'interrupted_checkpoint.pt'
        agent.save_checkpoint(str(checkpoint_path))
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save current state for debugging
        checkpoint_path = experiment_dir / 'error_checkpoint.pt'
        try:
            agent.save_checkpoint(str(checkpoint_path))
            print(f"💾 Error checkpoint saved: {checkpoint_path}")
        except:
            pass
        
        sys.exit(1)


if __name__ == '__main__':
    main()
