#!/usr/bin/env python3
"""
Analyze chess games to check if they are valid and meaningful
"""
import sys
import json
import chess
import chess.pgn
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agents.dqn_agent import DQNAgent
from src.training.configs import get_experiment_config


def analyze_single_game(agent, max_moves=100, verbose=False):
    """Analyze a single game played by the agent"""
    board = chess.Board()
    moves = []
    move_types = defaultdict(int)
    
    if verbose:
        print(f"Starting position: {board.fen()}")
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
            
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        # Agent selects move
        action = agent.select_action(board, deterministic=False)
        
        # Convert to move
        try:
            from src.utils.action_utils import action_to_move
            move = action_to_move(action, board)
            
            if move is None or move not in legal_moves:
                # Agent selected invalid move, pick random legal move
                move = chess.choice(legal_moves) if legal_moves else None
                move_types['invalid_agent'] += 1
            else:
                move_types['valid_agent'] += 1
                
        except Exception as e:
            if verbose:
                print(f"Error converting action {action}: {e}")
            move = chess.choice(legal_moves) if legal_moves else None
            move_types['error'] += 1
        
        if move is None:
            break
            
        # Analyze move type
        if board.is_capture(move):
            move_types['captures'] += 1
        if board.is_check():
            move_types['checks'] += 1
        if move.promotion:
            move_types['promotions'] += 1
        
        # Make move
        moves.append(move)
        board.push(move)
        
        if verbose and move_num < 10:
            print(f"Move {move_num + 1}: {move} -> {board.fen()}")
    
    # Game result analysis
    result_info = {
        'total_moves': len(moves),
        'final_position': board.fen(),
        'game_over': board.is_game_over(),
        'result': None,
        'termination': 'max_moves'
    }
    
    if board.is_game_over():
        if board.is_checkmate():
            result_info['result'] = '1-0' if board.turn == chess.BLACK else '0-1'
            result_info['termination'] = 'checkmate'
        elif board.is_stalemate():
            result_info['result'] = '1/2-1/2'
            result_info['termination'] = 'stalemate'
        elif board.is_insufficient_material():
            result_info['result'] = '1/2-1/2'
            result_info['termination'] = 'insufficient_material'
        else:
            result_info['result'] = '1/2-1/2'
            result_info['termination'] = 'draw'
    
    return {
        'moves': moves,
        'move_types': dict(move_types),
        'result_info': result_info,
        'board': board
    }


def create_pgn_from_moves(moves, result='*'):
    """Create PGN string from moves"""
    game = chess.pgn.Game()
    game.headers["Event"] = "DQN Self-Play Analysis"
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2025.10.03"
    game.headers["Round"] = "1"
    game.headers["White"] = "DQN Agent"
    game.headers["Black"] = "DQN Agent"
    game.headers["Result"] = result
    
    node = game
    board = chess.Board()
    
    for move in moves:
        if move in board.legal_moves:
            node = node.add_variation(move)
            board.push(move)
        else:
            break
    
    return str(game)


def analyze_agent_quality(experiment_name='baseline_small', num_games=5):
    """Analyze the quality of games played by the agent"""
    print(f"🔍 Analyzing agent quality for experiment: {experiment_name}")
    print("=" * 60)
    
    # Load experiment config
    config = get_experiment_config(experiment_name)
    
    # Create agent
    device = 'cpu'  # For analysis
    agent = DQNAgent(
        model_config=config.model_config,
        exploration_config=config.exploration_config,
        device=device,
        **config.agent_config
    )
    
    print(f"Agent initialized with {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
    
    # Analyze multiple games
    all_stats = {
        'games': [],
        'total_moves': 0,
        'move_types': defaultdict(int),
        'terminations': defaultdict(int),
        'avg_game_length': 0
    }
    
    print(f"\n🎮 Playing {num_games} games for analysis...")
    
    for game_idx in range(num_games):
        print(f"\nGame {game_idx + 1}:")
        game_analysis = analyze_single_game(agent, max_moves=100, verbose=(game_idx == 0))
        
        # Update statistics
        all_stats['games'].append(game_analysis)
        all_stats['total_moves'] += game_analysis['result_info']['total_moves']
        
        for move_type, count in game_analysis['move_types'].items():
            all_stats['move_types'][move_type] += count
            
        all_stats['terminations'][game_analysis['result_info']['termination']] += 1
        
        # Print game summary
        result_info = game_analysis['result_info']
        move_types = game_analysis['move_types']
        
        print(f"  Moves: {result_info['total_moves']}")
        print(f"  Termination: {result_info['termination']}")
        print(f"  Result: {result_info['result']}")
        print(f"  Valid moves: {move_types.get('valid_agent', 0)}")
        print(f"  Invalid moves: {move_types.get('invalid_agent', 0)}")
        print(f"  Captures: {move_types.get('captures', 0)}")
        
        # Save PGN for first game
        if game_idx == 0:
            pgn = create_pgn_from_moves(
                game_analysis['moves'], 
                game_analysis['result_info']['result'] or '*'
            )
            with open('sample_game.pgn', 'w') as f:
                f.write(pgn)
            print(f"  📝 Sample game saved to sample_game.pgn")
    
    # Calculate final statistics
    all_stats['avg_game_length'] = all_stats['total_moves'] / num_games if num_games > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total games: {num_games}")
    print(f"  Total moves: {all_stats['total_moves']}")
    print(f"  Average game length: {all_stats['avg_game_length']:.1f} moves")
    
    print(f"\n🎯 Move Quality:")
    total_agent_moves = all_stats['move_types'].get('valid_agent', 0) + all_stats['move_types'].get('invalid_agent', 0)
    if total_agent_moves > 0:
        valid_rate = all_stats['move_types'].get('valid_agent', 0) / total_agent_moves
        print(f"  Valid move rate: {valid_rate:.1%}")
        print(f"  Invalid move rate: {1-valid_rate:.1%}")
    
    print(f"  Captures: {all_stats['move_types'].get('captures', 0)}")
    print(f"  Checks: {all_stats['move_types'].get('checks', 0)}")
    print(f"  Promotions: {all_stats['move_types'].get('promotions', 0)}")
    
    print(f"\n🏁 Game Terminations:")
    for termination, count in all_stats['terminations'].items():
        print(f"  {termination}: {count} ({count/num_games:.1%})")
    
    # Quality assessment
    print(f"\n✅ Quality Assessment:")
    
    valid_rate = all_stats['move_types'].get('valid_agent', 0) / max(total_agent_moves, 1)
    if valid_rate > 0.9:
        print("  🟢 EXCELLENT: >90% valid moves")
    elif valid_rate > 0.7:
        print("  🟡 GOOD: >70% valid moves")
    elif valid_rate > 0.5:
        print("  🟠 FAIR: >50% valid moves")
    else:
        print("  🔴 POOR: <50% valid moves")
    
    checkmate_rate = all_stats['terminations'].get('checkmate', 0) / num_games
    if checkmate_rate > 0.3:
        print("  🟢 GOOD: Some games end in checkmate")
    elif checkmate_rate > 0.1:
        print("  🟡 FAIR: Few games end in checkmate")
    else:
        print("  🟠 POOR: No games end in checkmate")
    
    avg_length = all_stats['avg_game_length']
    if 20 <= avg_length <= 80:
        print("  🟢 GOOD: Reasonable game length")
    elif avg_length < 10:
        print("  🔴 POOR: Games too short")
    elif avg_length > 100:
        print("  🟠 FAIR: Games quite long")
    else:
        print("  🟡 FAIR: Game length acceptable")
    
    return all_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze chess agent game quality')
    parser.add_argument('--experiment', default='baseline_small', help='Experiment name')
    parser.add_argument('--games', type=int, default=5, help='Number of games to analyze')
    parser.add_argument('--load-checkpoint', help='Path to checkpoint to load')
    
    args = parser.parse_args()
    
    try:
        stats = analyze_agent_quality(args.experiment, args.games)
        print(f"\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
