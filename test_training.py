#!/usr/bin/env python3
"""
Test training setup - quick validation
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agents.dqn_agent import DQNAgent
from src.training.self_play import SelfPlayTrainer, TrainingConfig
from src.training.configs import get_experiment_config


def test_training_setup():
    """Test that training components work"""
    print("🧪 Testing Training Setup")
    print("=" * 50)
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test experiment config
    print("\n📋 Testing experiment configs...")
    config = get_experiment_config('baseline_small')
    print(f"✅ Loaded experiment: {config.name}")
    print(f"   Description: {config.description}")
    print(f"   Total games: {config.total_games}")
    
    # Test agent creation
    print("\n🤖 Testing agent creation...")
    agent = DQNAgent(
        model_config=config.model_config,
        exploration_config=config.exploration_config,
        device=device,
        **config.agent_config
    )
    print(f"✅ Agent created with {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
    
    # Test trainer creation
    print("\n🏋️ Testing trainer creation...")
    trainer = SelfPlayTrainer(
        agent=agent,
        config=config.training_config,
        stockfish_path='/usr/games/stockfish',  # May not exist, that's ok
        log_dir='test_logs',
        checkpoint_dir='test_checkpoints'
    )
    print("✅ Trainer created successfully")
    
    # Test single game (without Stockfish)
    print("\n🎮 Testing single game...")
    try:
        result, experiences = trainer.play_game()
        print(f"✅ Game completed: {result.termination} in {result.moves} moves")
        print(f"   Experiences collected: {len(experiences)}")
        
        # Test experience storage
        if experiences:
            for exp in experiences[:5]:  # Store first 5 experiences
                agent.replay_buffer.push(*exp)
            print(f"✅ Stored {min(5, len(experiences))} experiences in replay buffer")
        
        # Test training step (if enough experiences)
        if len(agent.replay_buffer) >= agent.min_buffer_size:
            metrics = agent.train_step()
            if metrics:
                print(f"✅ Training step completed: loss={metrics.get('loss', 0):.4f}")
            else:
                print("⚠️ Training step returned no metrics (buffer too small)")
        else:
            print(f"⚠️ Not enough experiences for training ({len(agent.replay_buffer)}/{agent.min_buffer_size})")
            
    except Exception as e:
        print(f"❌ Game test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test checkpoint save/load
    print("\n💾 Testing checkpoint save/load...")
    try:
        test_path = 'test_checkpoint.pt'
        agent.save_checkpoint(test_path)
        print("✅ Checkpoint saved")
        
        # Create new agent and load
        new_agent = DQNAgent(
            model_config=config.model_config,
            exploration_config=config.exploration_config,
            device=device,
            **config.agent_config
        )
        new_agent.load_checkpoint(test_path)
        print("✅ Checkpoint loaded")
        
        # Cleanup
        Path(test_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📝 Ready for training! Use:")
    print("   python train_dqn.py baseline_small --dry-run")
    print("   python train_dqn.py baseline_small")
    
    return True


if __name__ == '__main__':
    success = test_training_setup()
    sys.exit(0 if success else 1)
