#!/usr/bin/env python3
"""
Standalone ELO Evaluation Script

This script evaluates the ELO rating of a trained chess model by testing it
against multiple Stockfish configurations with known ELO ratings.

Usage:
    python evaluate_elo.py [checkpoint_path] [--quick] [--save-dir results/]
    
Examples:
    # Evaluate latest checkpoint with full evaluation
    python evaluate_elo.py checkpoints/latest.pt
    
    # Quick evaluation (fewer games)
    python evaluate_elo.py checkpoints/latest.pt --quick
    
    # Evaluate without checkpoint (random model)
    python evaluate_elo.py --quick
"""

import argparse
import logging
import torch
from pathlib import Path
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.dqn_agent import DQNAgent
from src.models.dueling_dqn import MODEL_CONFIGS, ModelConfig
from src.utils.exploration import EXPLORATION_CONFIGS
from src.utils.elo_calculator import ELOCalculator, full_elo_evaluation, quick_elo_estimate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_model_config_from_checkpoint(checkpoint_path: str) -> dict:
    """
    Detect model configuration from checkpoint by examining the state dict
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['q_network_state_dict']
        
        # Detect conv_channels from the first conv layer
        first_conv_weight = state_dict['conv_layers.0.0.weight']
        input_channels = first_conv_weight.shape[1]  # Should be 13 for chess
        first_conv_channels = first_conv_weight.shape[0]
        
        # Detect other conv layers
        conv_channels = [first_conv_channels]
        layer_idx = 1
        while f'conv_layers.{layer_idx}.0.weight' in state_dict:
            conv_weight = state_dict[f'conv_layers.{layer_idx}.0.weight']
            conv_channels.append(conv_weight.shape[0])
            layer_idx += 1
        
        # Detect hidden size from shared dense layer
        shared_dense_weight = state_dict['shared_dense.0.weight']
        hidden_size = shared_dense_weight.shape[0]
        
        # Find matching model config
        detected_config = {
            'input_channels': input_channels,
            'conv_channels': conv_channels,
            'hidden_size': hidden_size,
            'action_size': 4672,  # Fixed for chess
            'dropout': 0.1,  # Default
            'activation': 'relu'  # Default
        }
        
        # Try to match with existing configs
        for config_name, config in MODEL_CONFIGS.items():
            cfg_conv = config['conv_channels'] if isinstance(config, dict) else config.conv_channels
            cfg_hidden = config['hidden_size'] if isinstance(config, dict) else config.hidden_size
            if (cfg_conv == conv_channels and cfg_hidden == hidden_size):
                logger.info(f"Detected model config: {config_name}")
                return ModelConfig(**config) if isinstance(config, dict) else config
        
        # If no exact match, create custom config
        logger.info(f"Custom model config detected: conv_channels={conv_channels}, hidden_size={hidden_size}")
        from src.models.dueling_dqn import ModelConfig
        return ModelConfig(**detected_config)
        
    except Exception as e:
        logger.warning(f"Could not detect model config from checkpoint: {e}")
        logger.info("Falling back to 'small' model config")
        return MODEL_CONFIGS['small']

def detect_agent_config_from_checkpoint(checkpoint_path: str) -> dict:
    """
    Try to detect agent configuration from checkpoint metadata
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to get config from checkpoint metadata
        if 'config' in checkpoint:
            config = checkpoint['config']
            return {
                'buffer_size': config.get('buffer_size', 10000),
                'min_buffer_size': config.get('min_buffer_size', 1000),
                'batch_size': config.get('batch_size', 64),
                'learning_rate': config.get('learning_rate', 1e-4),
                'gamma': config.get('gamma', 0.99),
                'tau': config.get('tau', 0.005)
            }
    except Exception as e:
        logger.debug(f"Could not detect agent config: {e}")
    
    # Default configuration
    return {
        'buffer_size': 10000,
        'min_buffer_size': 1000,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'tau': 0.005
    }

def load_agent_from_checkpoint(checkpoint_path: str, device: torch.device, forced_model: str = None) -> DQNAgent:
    """Load agent from checkpoint with automatic (or forced) config detection"""
    
    # Model config: forced via CLI or auto-detect
    if forced_model:
        model_config = MODEL_CONFIGS[forced_model]
        logger.info(f"Using forced model config: {forced_model}")
    else:
        model_config = detect_model_config_from_checkpoint(checkpoint_path)
    agent_config = detect_agent_config_from_checkpoint(checkpoint_path)
    exploration_config = EXPLORATION_CONFIGS['standard']
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    logger.info(f"Model config: conv_channels={getattr(model_config, 'conv_channels', 'unknown')}, "
               f"hidden_size={getattr(model_config, 'hidden_size', 'unknown')}")
    
    # Create agent with detected parameters
    # Ensure we pass only supported kwargs to create_dqn_model / DuelingDQN
    allowed_keys = {"input_channels", "conv_channels", "hidden_size", "action_size", "dropout"}
    if isinstance(model_config, ModelConfig):
        model_kwargs = {
            'input_channels': model_config.input_channels,
            'conv_channels': model_config.conv_channels,
            'hidden_size': model_config.hidden_size,
            'action_size': model_config.action_size,
            'dropout': model_config.dropout,
        }
    else:
        model_kwargs = {k: v for k, v in dict(model_config).items() if k in allowed_keys}

    agent = DQNAgent(
        model_config=model_kwargs,
        buffer_size=agent_config['buffer_size'],
        min_buffer_size=agent_config['min_buffer_size'],
        batch_size=agent_config['batch_size'],
        learning_rate=agent_config['learning_rate'],
        gamma=agent_config['gamma'],
        tau=agent_config['tau'],
        exploration_config=exploration_config,
        device=device
    )
    
    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)
    logger.info(f"Agent loaded from {checkpoint_path}")
    
    return agent

def create_random_agent(device: torch.device) -> DQNAgent:
    """Create a random (untrained) agent for baseline testing"""
    
    model_config = MODEL_CONFIGS['small']  # Use small model for random baseline
    exploration_config = EXPLORATION_CONFIGS['standard']
    
    agent = DQNAgent(
        model_config=model_config,
        buffer_size=1000,
        min_buffer_size=100,
        batch_size=32,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        exploration_config=exploration_config,
        device=device
    )
    
    logger.info("Created random (untrained) agent for baseline testing")
    return agent

def main():
    parser = argparse.ArgumentParser(description="Evaluate ELO rating of chess model")
    parser.add_argument("checkpoint", nargs='?', help="Path to model checkpoint (optional)")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation (fewer games)")
    parser.add_argument("--save-dir", default="results/elo_evaluation", 
                       help="Directory to save results")
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish",
                       help="Path to Stockfish executable")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--model-config", choices=list(MODEL_CONFIGS.keys()),
                        help="Force a model config (small/medium/large) instead of auto-detect")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Determine save directory based on checkpoint location
    if args.checkpoint and Path(args.checkpoint).exists():
        # Save in the same directory as the checkpoint
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.parent.name == "checkpoints":
            # If checkpoint is in checkpoints/ folder, save in parent directory
            save_dir = checkpoint_path.parent.parent / "elo_evaluation"
        else:
            # Save alongside the checkpoint
            save_dir = checkpoint_path.parent / "elo_evaluation"
    else:
        # Fallback to specified save directory
        save_dir = Path(args.save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create agent
    if args.checkpoint and Path(args.checkpoint).exists():
        agent = load_agent_from_checkpoint(args.checkpoint, device, args.model_config)
        model_name = Path(args.checkpoint).stem
    else:
        if args.checkpoint:
            logger.warning(f"Checkpoint {args.checkpoint} not found. Using random agent.")
        agent = create_random_agent(device)
        model_name = "random_baseline"
    
    # Run evaluation
    logger.info("🎯 Starting ELO Evaluation")
    logger.info(f"Model: {model_name}")
    logger.info(f"Evaluation type: {'Quick' if args.quick else 'Full'}")
    logger.info(f"Stockfish path: {args.stockfish_path}")
    
    try:
        if args.quick:
            # Quick evaluation
            estimated_elo = quick_elo_estimate(agent, args.stockfish_path)
            logger.info(f"🏆 Quick ELO Estimate: {estimated_elo}")
            
            # Save simple result
            result_file = save_dir / f"{model_name}_quick_elo.txt"
            with open(result_file, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Quick ELO Estimate: {estimated_elo}\n")
                f.write(f"Evaluation Date: {datetime.now().isoformat()}\n")
            
        else:
            # Full evaluation
            elo_evaluation = full_elo_evaluation(agent, save_dir, args.stockfish_path)
            
            logger.info("🎉 Full ELO Evaluation Complete!")
            logger.info(f"🏆 Estimated ELO: {elo_evaluation.estimated_elo}")
            logger.info(f"📊 95% Confidence Interval: {elo_evaluation.confidence_interval[0]}-{elo_evaluation.confidence_interval[1]}")
            logger.info(f"🎮 Total Games: {elo_evaluation.total_games}")
            logger.info(f"📈 Overall Score: {elo_evaluation.overall_score:.3f}")
            
            # Print detailed results
            logger.info("\n📋 Detailed Results:")
            for config_name, eval_result in elo_evaluation.evaluations.items():
                logger.info(f"  {config_name} (ELO {eval_result.stockfish_config.estimated_elo}): "
                          f"W{eval_result.wins} D{eval_result.draws} L{eval_result.losses} "
                          f"(Score: {eval_result.score:.3f})")
            
            logger.info(f"💾 Detailed results saved to: {save_dir / 'elo_evaluation.json'}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
