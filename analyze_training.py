#!/usr/bin/env python3
"""
Comprehensive training analysis and visualization
"""
import sys
import json
import chess
import chess.pgn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agents.dqn_agent import DQNAgent
from src.training.configs import get_experiment_config


def analyze_training_progress(results_dir: str):
    """Analyze training progress from results directory"""
    results_path = Path(results_dir)
    
    print(f"📊 Analyzing training results from: {results_path}")
    print("=" * 60)
    
    # Load experiment info
    experiment_info_path = results_path / 'experiment_info.json'
    if not experiment_info_path.exists():
        print("❌ experiment_info.json not found")
        return None, None
    
    with open(experiment_info_path) as f:
        experiment_info = json.load(f)
    
    # Load training history
    history_path = results_path / 'training_history.json'
    if not history_path.exists():
        print("⚠️  training_history.json not found (maybe run interrupted)")
        return experiment_info, None
    
    with open(history_path) as f:
        history = json.load(f)
    
    # Print experiment summary
    exp = experiment_info['experiment']
    print(f"🎯 Experiment: {exp['name']}")
    print(f"📝 Description: {exp['description']}")
    print(f"🎮 Total Games: {exp['total_games']}")
    print(f"⏰ Started: {exp['timestamp']}")
    print(f"💻 Device: {experiment_info['system']['device']}")
    
    # Analyze training metrics
    print(f"\n📈 Training Progress Analysis:")
    
    # Episode rewards
    episode_rewards = history['episode_rewards']
    print(f"  Average Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Reward Range: {np.min(episode_rewards):.3f} - {np.max(episode_rewards):.3f}")
    
    # Training losses
    training_losses = history['training_losses']
    print(f"  Average Training Loss: {np.mean(training_losses):.4f} ± {np.std(training_losses):.4f}")
    print(f"  Loss Trend: {training_losses[0]:.4f} → {training_losses[-1]:.4f}")
    
    # Game lengths
    episode_lengths = history['episode_lengths']
    print(f"  Average Game Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} moves")
    
    # Evaluation results
    eval_results = history['eval_results']
    if eval_results:
        final_eval = history.get('final_evaluation', eval_results[-1])
        print(f"\n🏆 Performance vs Stockfish:")
        print(f"  Win Rate: {final_eval['win_rate']:.1%}")
        print(f"  Draw Rate: {final_eval['draw_rate']:.1%}")
        print(f"  Loss Rate: {final_eval['loss_rate']:.1%}")
        print(f"  Total Games: {final_eval['total_games']}")
    
    # Learning indicators
    print(f"\n🧠 Learning Indicators:")
    
    # Loss trend
    if len(training_losses) >= 5:
        early_loss = np.mean(training_losses[:5])
        late_loss = np.mean(training_losses[-5:])
        loss_improvement = (early_loss - late_loss) / early_loss * 100
        
        if loss_improvement > 10:
            print(f"  🟢 Loss Decreasing: -{loss_improvement:.1f}% (GOOD)")
        elif loss_improvement > 0:
            print(f"  🟡 Loss Slightly Decreasing: -{loss_improvement:.1f}% (OK)")
        else:
            print(f"  🔴 Loss Not Decreasing: +{abs(loss_improvement):.1f}% (CONCERNING)")
    
    # Reward stability
    reward_std = np.std(episode_rewards)
    if reward_std < 0.1:
        print(f"  🟢 Rewards Stable: σ={reward_std:.3f} (GOOD)")
    elif reward_std < 0.5:
        print(f"  🟡 Rewards Moderately Stable: σ={reward_std:.3f} (OK)")
    else:
        print(f"  🔴 Rewards Unstable: σ={reward_std:.3f} (CONCERNING)")
    
    # Evaluation trend
    if len(eval_results) >= 3:
        early_loss_rate = np.mean([r['loss_rate'] for r in eval_results[:3]])
        late_loss_rate = np.mean([r['loss_rate'] for r in eval_results[-3:]])
        
        if late_loss_rate < early_loss_rate:
            improvement = (early_loss_rate - late_loss_rate) * 100
            print(f"  🟢 Stockfish Performance Improving: -{improvement:.1f}% losses (EXCELLENT)")
        elif late_loss_rate == early_loss_rate:
            print(f"  🟡 Stockfish Performance Stable (OK)")
        else:
            decline = (late_loss_rate - early_loss_rate) * 100
            print(f"  🔴 Stockfish Performance Declining: +{decline:.1f}% losses (CONCERNING)")
    
    return experiment_info, history


def create_training_plots(results_dir: str, save_plots: bool = True):
    """Create training visualization plots"""
    results_path = Path(results_dir)
    
    # Load data
    with open(results_path / 'training_history.json') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Analysis', fontsize=16)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(history['episode_rewards'], 'b-', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    axes[0, 1].plot(history['training_losses'], 'r-', alpha=0.7)
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Game Lengths
    axes[1, 0].plot(history['episode_lengths'], 'g-', alpha=0.7)
    axes[1, 0].set_title('Average Game Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Moves')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Evaluation Results
    if history['eval_results']:
        eval_episodes = list(range(0, len(history['eval_results'])))
        win_rates = [r['win_rate'] for r in history['eval_results']]
        draw_rates = [r['draw_rate'] for r in history['eval_results']]
        loss_rates = [r['loss_rate'] for r in history['eval_results']]
        
        axes[1, 1].plot(eval_episodes, win_rates, 'g-', label='Win Rate', alpha=0.7)
        axes[1, 1].plot(eval_episodes, draw_rates, 'y-', label='Draw Rate', alpha=0.7)
        axes[1, 1].plot(eval_episodes, loss_rates, 'r-', label='Loss Rate', alpha=0.7)
        axes[1, 1].set_title('Performance vs Stockfish')
        axes[1, 1].set_xlabel('Evaluation')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = results_path / 'training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_path}")
    
    return fig


def compare_checkpoints(results_dir: str, num_games: int = 3):
    """Compare different checkpoints to see learning progress"""
    results_path = Path(results_dir)
    checkpoints_dir = results_path / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found")
        return
    
    # Find all checkpoint files
    checkpoint_files = sorted([f for f in checkpoints_dir.glob('checkpoint_game_*.pt') 
                              if not f.name.endswith('_buffer.pt')])
    
    if len(checkpoint_files) < 2:
        print("❌ Need at least 2 checkpoints to compare")
        return
    
    print(f"\n🔄 Comparing Checkpoints:")
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Load experiment config
    with open(results_path / 'experiment_info.json') as f:
        experiment_info = json.load(f)
    
    # Compare first and last checkpoint
    first_checkpoint = checkpoint_files[0]
    last_checkpoint = checkpoint_files[-1]
    
    print(f"\n📊 Comparing:")
    print(f"  Early: {first_checkpoint.name}")
    print(f"  Late:  {last_checkpoint.name}")
    
    # This would require loading the actual models and testing them
    # For now, we'll show the structure
    print(f"\n💡 To compare checkpoints, use:")
    print(f"  make analyze-checkpoint CHECKPOINT={first_checkpoint}")
    print(f"  make analyze-checkpoint CHECKPOINT={last_checkpoint}")


def generate_sample_games(results_dir: str, checkpoint_name: str = None, num_games: int = 3):
    """Generate sample games from a checkpoint"""
    results_path = Path(results_dir)
    
    # Load experiment config (if available)
    exp_info_path = results_path / 'experiment_info.json'
    experiment_name = None
    if exp_info_path.exists():
        with open(exp_info_path) as f:
            experiment_info = json.load(f)
            experiment_name = experiment_info['experiment']['name']
    
    # Determine checkpoint to use
    if checkpoint_name:
        cp = Path(checkpoint_name)
        if cp.exists():
            checkpoint_path = cp
        else:
            checkpoint_path = results_path / 'checkpoints' / checkpoint_name
    else:
        checkpoint_path = results_path / 'final_model.pt'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n🎮 Generating {num_games} sample games from: {checkpoint_path}")
    
    # Load agent: prefer experiment config, otherwise auto-detect via checkpoint
    agent = None
    if experiment_name:
        try:
            config = get_experiment_config(experiment_name)
            agent = DQNAgent(
                model_config=config.model_config,
                exploration_config=config.exploration_config,
                device='cpu',
                **config.agent_config
            )
            agent.load_checkpoint(str(checkpoint_path))
            print("✅ Checkpoint loaded successfully (experiment config)")
        except Exception as e:
            print(f"⚠️ Failed with experiment config ({experiment_name}): {e}")
            agent = None
    if agent is None:
        # Fallback: auto-detect model from checkpoint
        try:
            from evaluate_elo import load_agent_from_checkpoint
            agent = load_agent_from_checkpoint(str(checkpoint_path), device=torch.device('cpu'))
            print("✅ Checkpoint loaded successfully (auto-detected config)")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return
    
    # Generate games
    games_dir = results_path / 'sample_games'
    games_dir.mkdir(exist_ok=True)
    
    for game_idx in range(num_games):
        print(f"\nGame {game_idx + 1}:")
        
        board = chess.Board()
        moves = []
        
        # Play game
        for move_num in range(100):  # Max 100 moves
            if board.is_game_over():
                break
            
            # Agent selects move
            action = agent.select_action(board, deterministic=False)
            
            # Convert to move
            try:
                from src.utils.action_utils import action_to_move
                move = action_to_move(action, board)
                
                if move is None or move not in board.legal_moves:
                    # Fallback to random legal move
                    import random
                    move = random.choice(list(board.legal_moves))
                
                moves.append(move)
                board.push(move)
                
            except Exception as e:
                print(f"  Error on move {move_num + 1}: {e}")
                break
        
        # Create PGN
        game = chess.pgn.Game()
        game.headers["Event"] = f"DQN Analysis Game {game_idx + 1}"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_idx + 1)
        game.headers["White"] = f"DQN ({checkpoint_path.name})"
        game.headers["Black"] = f"DQN ({checkpoint_path.name})"
        
        # Determine result
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
        else:
            result = "*"
        
        game.headers["Result"] = result
        
        # Add moves
        node = game
        temp_board = chess.Board()
        for move in moves:
            if move in temp_board.legal_moves:
                node = node.add_variation(move)
                temp_board.push(move)
        
        # Save PGN
        pgn_path = games_dir / f"game_{game_idx + 1}_{checkpoint_path.stem}.pgn"
        with open(pgn_path, 'w') as f:
            f.write(str(game))
        
        print(f"  Moves: {len(moves)}")
        print(f"  Result: {result}")
        print(f"  Saved: {pgn_path}")
    
    print(f"\n📁 All games saved to: {games_dir}")
    print(f"💡 View games with any chess GUI (e.g., lichess.org/analysis)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--plots', action='store_true', help='Generate training plots')
    parser.add_argument('--compare', action='store_true', help='Compare checkpoints')
    parser.add_argument('--games', type=int, default=3, help='Number of sample games to generate')
    parser.add_argument('--checkpoint', help='Specific checkpoint to analyze')
    parser.add_argument('--generate-games', action='store_true', help='Generate sample games')
    
    args = parser.parse_args()
    
    if not Path(args.results_dir).exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        return
    
    # Main analysis
    experiment_info, history = analyze_training_progress(args.results_dir)
    
    # Generate plots (only if history is available)
    if args.plots and history:
        create_training_plots(args.results_dir)
    elif args.plots and not history:
        print("⚠️  Skipping plots: no training_history.json")
    
    # Compare checkpoints
    if args.compare:
        compare_checkpoints(args.results_dir)
    
    # Generate sample games (can work without history)
    if args.generate_games:
        generate_sample_games(args.results_dir, args.checkpoint, args.games)
    
    print(f"\n🎉 Analysis complete!")


if __name__ == '__main__':
    main()
