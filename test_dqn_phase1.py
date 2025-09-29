#!/usr/bin/env python3
"""
Test script for Phase 1 DQN implementation
Tests all core components to ensure they work correctly
"""

import torch
import chess
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.dqn_agent import DQNAgent
from src.models.dueling_dqn import create_dqn_model, MODEL_CONFIGS
from src.utils.action_utils import ChessActionSpace, board_to_tensor
from src.utils.exploration import create_exploration_strategy, EXPLORATION_CONFIGS
from src.replay.replay_buffer import create_replay_buffer


def test_dueling_dqn():
    """Test Dueling DQN architecture"""
    print("🧠 Testing Dueling DQN Architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different model configurations
    for config_name, config in MODEL_CONFIGS.items():
        print(f"  Testing {config_name} configuration...")
        
        model = create_dqn_model('cnn', **config).to(device)
        
        # Test input
        batch_size = 4
        x = torch.randn(batch_size, 13, 8, 8).to(device)
        action_mask = torch.ones(batch_size, 4672, dtype=torch.bool).to(device)
        
        # Forward pass
        q_values = model(x, action_mask)
        
        assert q_values.shape == (batch_size, 4672), f"Wrong output shape: {q_values.shape}"
        assert not torch.isnan(q_values).any(), "NaN values in output"
        
        # Test feature extraction
        features, value, advantage = model.get_features(x)
        assert features.shape[0] == batch_size, "Wrong feature batch size"
        assert value.shape == (batch_size, 1), f"Wrong value shape: {value.shape}"
        assert advantage.shape == (batch_size, 4672), f"Wrong advantage shape: {advantage.shape}"
        
        print(f"    ✅ {config_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("  ✅ Dueling DQN tests passed!\n")


def test_action_space():
    """Test action space and utilities"""
    print("🎯 Testing Action Space...")
    
    action_space = ChessActionSpace()
    
    # Test with various positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After e4
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),  # Italian Game
    ]
    
    for i, board in enumerate(test_positions):
        print(f"  Testing position {i+1}...")
        
        # Test legal actions mask
        mask = action_space.get_legal_actions_mask(board)
        legal_actions = action_space.get_legal_actions(board)
        
        assert mask.sum().item() == len(legal_actions), "Mask and actions count mismatch"
        assert len(legal_actions) == len(list(board.legal_moves)), "Wrong legal moves count"
        
        # Test move conversions
        for move in list(board.legal_moves)[:5]:  # Test first 5 moves
            action_idx = action_space.move_to_action(move)
            recovered_move = action_space.action_to_move(action_idx)
            
            assert recovered_move == move.uci(), f"Move conversion failed: {move.uci()} -> {recovered_move}"
        
        # Test board to tensor conversion
        tensor = board_to_tensor(board)
        assert tensor.shape == (13, 8, 8), f"Wrong tensor shape: {tensor.shape}"
        assert tensor.dtype == torch.float32, "Wrong tensor dtype"
        
        print(f"    ✅ Position {i+1}: {len(legal_actions)} legal moves")
    
    print("  ✅ Action space tests passed!\n")


def test_exploration():
    """Test exploration strategies"""
    print("🔍 Testing Exploration Strategies...")
    
    # Test different exploration strategies
    for strategy_name, config in EXPLORATION_CONFIGS.items():
        print(f"  Testing {strategy_name} exploration...")
        
        exploration = create_exploration_strategy(**config)
        
        # Test epsilon decay
        initial_epsilon = exploration.get_epsilon(0)
        mid_epsilon = exploration.get_epsilon(config['epsilon_decay_steps'] // 2)
        final_epsilon = exploration.get_epsilon(config['epsilon_decay_steps'])
        
        assert initial_epsilon == config['epsilon_start'], "Wrong initial epsilon"
        assert final_epsilon == config['epsilon_end'], "Wrong final epsilon"
        assert initial_epsilon > mid_epsilon > final_epsilon, "Epsilon not decreasing"
        
        # Test action selection
        q_values = torch.randn(4672)
        legal_mask = torch.ones(4672, dtype=torch.bool)
        legal_mask[100:] = False  # Only first 100 actions are legal
        
        # Test deterministic selection
        action_det = exploration.select_action(q_values, legal_mask, deterministic=True)
        assert action_det < 100, "Illegal action selected in deterministic mode"
        
        # Test random selection
        exploration.current_epsilon = 1.0  # Force random selection
        action_rand = exploration.select_action(q_values, legal_mask, deterministic=False)
        assert action_rand < 100, "Illegal action selected in random mode"
        
        print(f"    ✅ {strategy_name}: eps {initial_epsilon:.3f} -> {final_epsilon:.3f}")
    
    print("  ✅ Exploration tests passed!\n")


def test_replay_buffer():
    """Test replay buffer implementations"""
    print("💾 Testing Replay Buffers...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test standard buffer
    print("  Testing standard replay buffer...")
    buffer = create_replay_buffer('standard', capacity=1000, device=device)
    
    # Add experiences
    for i in range(100):
        state = torch.randn(13, 8, 8)
        action = i % 64
        reward = np.random.randn()
        next_state = torch.randn(13, 8, 8)
        done = i % 20 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    assert len(buffer) == 100, f"Wrong buffer size: {len(buffer)}"
    
    # Test sampling
    if buffer.is_ready(32):
        batch = buffer.sample(32)
        assert len(batch) == 5, "Wrong batch tuple length"
        assert batch[0].shape == (32, 13, 8, 8), f"Wrong states shape: {batch[0].shape}"
        assert batch[1].shape == (32,), f"Wrong actions shape: {batch[1].shape}"
    
    print("    ✅ Standard buffer: 100 experiences, 32 batch sampling")
    
    # Test prioritized buffer
    print("  Testing prioritized replay buffer...")
    pri_buffer = create_replay_buffer('prioritized', capacity=1000, device=device)
    
    for i in range(100):
        state = torch.randn(13, 8, 8)
        action = i % 64
        reward = np.random.randn()
        next_state = torch.randn(13, 8, 8)
        done = i % 20 == 0
        
        pri_buffer.push(state, action, reward, next_state, done)
    
    if pri_buffer.is_ready(32):
        batch = pri_buffer.sample(32)
        assert len(batch) == 7, "Wrong prioritized batch tuple length"  # includes weights and indices
        
        # Test priority updates
        indices = batch[6]
        priorities = torch.abs(torch.randn(32)) + 0.1
        pri_buffer.update_priorities(indices, priorities)
    
    print("    ✅ Prioritized buffer: 100 experiences, priority updates")
    print("  ✅ Replay buffer tests passed!\n")


def test_dqn_agent():
    """Test complete DQN agent"""
    print("🤖 Testing DQN Agent...")
    
    # Test with small configuration for speed
    config = {
        'model_type': 'cnn',
        'model_config': MODEL_CONFIGS['small'],
        'buffer_type': 'standard',
        'buffer_size': 1000,
        'batch_size': 32,
        'min_buffer_size': 100,
        'exploration_config': EXPLORATION_CONFIGS['aggressive']
    }
    
    agent = DQNAgent(**config)
    
    # Test with chess positions
    board = chess.Board()
    
    # Test action selection
    action = agent.select_action(board)
    assert isinstance(action, int), "Action should be integer"
    assert 0 <= action < 4672, f"Action out of range: {action}"
    
    # Test move selection
    move = agent.act(board)
    assert move is None or move in board.legal_moves, "Illegal move selected"
    
    # Test state value
    value = agent.get_state_value(board)
    assert isinstance(value, float), "Value should be float"
    
    print(f"  Action selection: {action}")
    print(f"  Move selection: {move}")
    print(f"  State value: {value:.4f}")
    
    # Test experience storage and training
    print("  Testing training loop...")
    
    # Add experiences
    for i in range(150):  # More than min_buffer_size
        state = board_to_tensor(board)
        action = agent.select_action(board)
        reward = np.random.randn()
        
        # Make a random move to get next state
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(np.random.choice(legal_moves))
        next_state = board_to_tensor(board)
        done = board.is_game_over()
        
        agent.remember(state, action, reward, next_state, done)
        
        if done:
            board.reset()
    
    # Test training step
    metrics = agent.train_step()
    assert 'loss' in metrics, "Loss not in training metrics"
    assert 'epsilon' in metrics, "Epsilon not in training metrics"
    
    print(f"    Training metrics: {metrics}")
    
    # Test checkpoint saving/loading
    print("  Testing checkpoint save/load...")
    
    checkpoint_path = "test_checkpoint.pt"
    agent.save_checkpoint(checkpoint_path)
    
    # Create new agent and load checkpoint
    agent2 = DQNAgent(**config)
    agent2.load_checkpoint(checkpoint_path, load_buffer=False)
    
    # Clean up
    os.remove(checkpoint_path)
    if os.path.exists(checkpoint_path.replace('.pt', '_buffer.pt')):
        os.remove(checkpoint_path.replace('.pt', '_buffer.pt'))
    
    print("    ✅ Checkpoint save/load successful")
    print("  ✅ DQN Agent tests passed!\n")


def test_integration():
    """Test integration between components"""
    print("🔗 Testing Component Integration...")
    
    # Create minimal training loop
    agent = DQNAgent(
        model_config=MODEL_CONFIGS['small'],
        buffer_size=500,
        min_buffer_size=50,
        batch_size=16,
        exploration_config=EXPLORATION_CONFIGS['aggressive']
    )
    
    # Play a few moves and train
    board = chess.Board()
    total_reward = 0
    
    for step in range(100):
        # Agent's turn
        action = agent.select_action(board)
        move_uci = agent.action_space.action_to_move(action)
        
        if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves:
            old_state = board_to_tensor(board)
            board.push(chess.Move.from_uci(move_uci))
            reward = 0.01  # Small positive reward for legal moves
        else:
            # Illegal move penalty
            old_state = board_to_tensor(board)
            reward = -0.1
        
        # Random opponent move
        legal_moves = list(board.legal_moves)
        if legal_moves and not board.is_game_over():
            board.push(np.random.choice(legal_moves))
        
        new_state = board_to_tensor(board)
        done = board.is_game_over()
        
        # Store experience
        agent.remember(old_state, action, reward, new_state, done)
        total_reward += reward
        
        # Train if buffer is ready
        if agent.replay_buffer.is_ready(agent.min_buffer_size):
            metrics = agent.train_step()
            if step % 20 == 0 and metrics:
                print(f"    Step {step}: Loss {metrics.get('loss', 0):.4f}, "
                      f"Epsilon {metrics.get('epsilon', 0):.3f}")
        
        if done:
            board.reset()
    
    print(f"    Total reward: {total_reward:.2f}")
    print("  ✅ Integration tests passed!\n")


def main():
    """Run all tests"""
    print("🚀 Starting Phase 1 DQN Tests\n")
    
    try:
        test_dueling_dqn()
        test_action_space()
        test_exploration()
        test_replay_buffer()
        test_dqn_agent()
        test_integration()
        
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("\n✅ Phase 1 Implementation is ready for Phase 2!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
