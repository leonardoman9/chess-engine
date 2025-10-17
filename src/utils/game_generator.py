"""
Game Generator for Chess Engine

This module generates sample games from trained models for analysis and demonstration.
"""

import chess
import chess.engine
import chess.pgn
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def generate_sample_games(
    agent,
    num_games: int = 10,
    stockfish_path: str = "/usr/games/stockfish",
    save_dir: Path = None,
    max_moves: int = 200,
    stockfish_depth: int = 1,
    stockfish_time: float = 0.2,
    self_play: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate sample games between the agent and Stockfish or agent vs agent (self-play)
    
    Args:
        agent: Trained DQN agent
        num_games: Number of games to generate
        stockfish_path: Path to Stockfish executable (ignored if self_play=True)
        save_dir: Directory to save games (optional)
        max_moves: Maximum moves per game
        stockfish_depth: Stockfish search depth (ignored if self_play=True)
        stockfish_time: Stockfish time per move (ignored if self_play=True)
        self_play: If True, agent plays against itself
        
    Returns:
        List of game dictionaries with metadata
    """
    
    logger.info(f"Generating {num_games} sample games...")
    
    games = []
    
    for game_idx in range(num_games):
        logger.info(f"Playing game {game_idx + 1}/{num_games}")
        
        # Alternate who plays white
        agent_plays_white = game_idx % 2 == 0
        
        if self_play:
            game_result = play_self_play_game(
                agent=agent,
                max_moves=max_moves
            )
        else:
            game_result = play_single_game(
                agent=agent,
                agent_plays_white=agent_plays_white,
                stockfish_path=stockfish_path,
                max_moves=max_moves,
                stockfish_depth=stockfish_depth,
                stockfish_time=stockfish_time
            )
        
        # Add game metadata
        game_result['game_id'] = game_idx + 1
        if self_play:
            game_result['agent_color'] = 'both'  # Agent plays both sides
            game_result['game_type'] = 'self_play'
        else:
            game_result['agent_color'] = 'white' if agent_plays_white else 'black'
            game_result['game_type'] = 'vs_stockfish'
        game_result['timestamp'] = datetime.now().isoformat()
        
        games.append(game_result)
        
        # Save individual game if save_dir provided
        if save_dir:
            save_game_pgn(game_result, save_dir, game_idx + 1)
    
    # Save games summary
    if save_dir:
        save_games_summary(games, save_dir)
    
    logger.info(f"Generated {len(games)} sample games successfully")
    return games

def play_self_play_game(
    agent,
    max_moves: int = 200
) -> Dict[str, Any]:
    """
    Play a single self-play game where agent plays against itself
    
    Returns:
        Dictionary with game result and metadata
    """
    
    board = chess.Board()
    moves_played = 0
    game_moves = []
    move_times = []
    
    while not board.is_game_over() and moves_played < max_moves:
        start_time = datetime.now()
        
        # Agent plays both sides, but with slight randomization to avoid identical play
        if board.turn == chess.WHITE:
            # White: deterministic play
            move = get_agent_move(agent, board, deterministic=True)
            player = "agent_white"
        else:
            # Black: slightly more exploratory play
            move = get_agent_move(agent, board, deterministic=False)
            player = "agent_black"
        
        if move and move in board.legal_moves:
            # Record move with metadata
            move_time = (datetime.now() - start_time).total_seconds()
            move_data = {
                'move': move.uci(),
                'san': board.san(move),
                'player': player,
                'move_number': moves_played + 1,
                'time_seconds': move_time,
                'fen_before': board.fen(),
                'is_capture': board.is_capture(move),
                'is_check': board.gives_check(move),
                'is_castling': board.is_castling(move),
                'piece_moved': str(board.piece_at(move.from_square))
            }
            
            board.push(move)
            move_data['fen_after'] = board.fen()
            
            game_moves.append(move_data)
            move_times.append(move_time)
            moves_played += 1
        else:
            # Invalid move - end game
            logger.warning(f"Invalid move attempted: {move}")
            break
    
    # Determine game result
    result = board.result()
    if result == "*":
        result = "1/2-1/2"  # Timeout = draw
        termination = "timeout"
    elif board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material():
        termination = "insufficient_material"
    elif board.is_seventyfive_moves():
        termination = "75_moves"
    elif board.is_fivefold_repetition():
        termination = "repetition"
    else:
        termination = "other"
    
    # Calculate game statistics
    white_moves = [m for m in game_moves if m['player'] == 'agent_white']
    black_moves = [m for m in game_moves if m['player'] == 'agent_black']
    
    captures = len([m for m in game_moves if m['is_capture']])
    checks = len([m for m in game_moves if m['is_check']])
    castlings = len([m for m in game_moves if m['is_castling']])
    
    avg_white_time = sum(m['time_seconds'] for m in white_moves) / len(white_moves) if white_moves else 0
    avg_black_time = sum(m['time_seconds'] for m in black_moves) / len(black_moves) if black_moves else 0
    
    return {
        'result': result,
        'termination': termination,
        'moves': moves_played,
        'game_moves': game_moves,
        'final_fen': board.fen(),
        'game_type': 'self_play',
        'statistics': {
            'captures': captures,
            'checks': checks,
            'castlings': castlings,
            'white_moves': len(white_moves),
            'black_moves': len(black_moves),
            'avg_white_time': avg_white_time,
            'avg_black_time': avg_black_time,
            'total_time': sum(move_times)
        }
    }

def play_single_game(
    agent,
    agent_plays_white: bool,
    stockfish_path: str,
    max_moves: int = 200,
    stockfish_depth: int = 1,
    stockfish_time: float = 0.2
) -> Dict[str, Any]:
    """
    Play a single game between agent and Stockfish
    
    Returns:
        Dictionary with game result and metadata
    """
    
    board = chess.Board()
    moves_played = 0
    game_moves = []
    move_times = []
    
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            while not board.is_game_over() and moves_played < max_moves:
                start_time = datetime.now()
                
                if (board.turn == chess.WHITE and agent_plays_white) or \
                   (board.turn == chess.BLACK and not agent_plays_white):
                    # Agent's turn
                    move = get_agent_move(agent, board)
                    player = "agent"
                else:
                    # Stockfish's turn
                    move = get_stockfish_move(engine, board, stockfish_depth, stockfish_time)
                    player = "stockfish"
                
                if move and move in board.legal_moves:
                    # Record move with metadata
                    move_time = (datetime.now() - start_time).total_seconds()
                    move_data = {
                        'move': move.uci(),
                        'san': board.san(move),
                        'player': player,
                        'move_number': moves_played + 1,
                        'time_seconds': move_time,
                        'fen_before': board.fen(),
                        'is_capture': board.is_capture(move),
                        'is_check': board.gives_check(move),
                        'is_castling': board.is_castling(move),
                        'piece_moved': str(board.piece_at(move.from_square))
                    }
                    
                    board.push(move)
                    move_data['fen_after'] = board.fen()
                    
                    game_moves.append(move_data)
                    move_times.append(move_time)
                    moves_played += 1
                else:
                    # Invalid move - end game
                    logger.warning(f"Invalid move attempted: {move}")
                    break
                    
    except Exception as e:
        logger.error(f"Error during game: {e}")
    
    # Determine game result
    result = board.result()
    if result == "*":
        result = "1/2-1/2"  # Timeout = draw
        termination = "timeout"
    elif board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material():
        termination = "insufficient_material"
    elif board.is_seventyfive_moves():
        termination = "75_moves"
    elif board.is_fivefold_repetition():
        termination = "repetition"
    else:
        termination = "other"
    
    # Calculate game statistics
    agent_moves = [m for m in game_moves if m['player'] == 'agent']
    stockfish_moves = [m for m in game_moves if m['player'] == 'stockfish']
    
    captures = len([m for m in game_moves if m['is_capture']])
    checks = len([m for m in game_moves if m['is_check']])
    castlings = len([m for m in game_moves if m['is_castling']])
    
    avg_agent_time = sum(m['time_seconds'] for m in agent_moves) / len(agent_moves) if agent_moves else 0
    avg_stockfish_time = sum(m['time_seconds'] for m in stockfish_moves) / len(stockfish_moves) if stockfish_moves else 0
    
    return {
        'result': result,
        'termination': termination,
        'moves': moves_played,
        'game_moves': game_moves,
        'final_fen': board.fen(),
        'statistics': {
            'captures': captures,
            'checks': checks,
            'castlings': castlings,
            'agent_moves': len(agent_moves),
            'stockfish_moves': len(stockfish_moves),
            'avg_agent_time': avg_agent_time,
            'avg_stockfish_time': avg_stockfish_time,
            'total_time': sum(move_times)
        }
    }

def get_agent_move(agent, board: chess.Board, deterministic: bool = True) -> Optional[chess.Move]:
    """Get move from agent"""
    try:
        action = agent.select_action(board, deterministic=deterministic)
        from .action_utils import action_to_move
        move = action_to_move(action, board)
        
        if move and move in board.legal_moves:
            return move
        else:
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves) if legal_moves else None
            
    except Exception as e:
        logger.error(f"Agent move error: {e}")
        # Fallback to random legal move
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

def get_stockfish_move(engine, board: chess.Board, depth: int, time_limit: float) -> Optional[chess.Move]:
    """Get move from Stockfish"""
    try:
        limit = chess.engine.Limit(time=time_limit, depth=depth)
        result = engine.play(board, limit)
        return result.move
    except Exception as e:
        logger.error(f"Stockfish move error: {e}")
        # Fallback to random legal move
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

def save_game_pgn(game_data: Dict[str, Any], save_dir: Path, game_number: int):
    """Save game in PGN format"""
    try:
        # Create PGN game
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = "Sample Game"
        game.headers["Site"] = "Chess Engine Training"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_number)
        if game_data.get('game_type') == 'self_play':
            game.headers["White"] = "Agent (Deterministic)"
            game.headers["Black"] = "Agent (Exploratory)"
        else:
            game.headers["White"] = "Agent" if game_data.get('agent_color') == 'white' else "Stockfish"
            game.headers["Black"] = "Stockfish" if game_data.get('agent_color') == 'white' else "Agent"
        game.headers["Result"] = game_data['result']
        game.headers["Termination"] = game_data['termination']
        game.headers["Moves"] = str(game_data['moves'])
        
        # Add moves
        node = game
        board = chess.Board()
        
        for move_data in game_data['game_moves']:
            move = chess.Move.from_uci(move_data['move'])
            node = node.add_variation(move)
            
            # Add comments with metadata
            comments = []
            if move_data['is_capture']:
                comments.append("capture")
            if move_data['is_check']:
                comments.append("check")
            if move_data['is_castling']:
                comments.append("castling")
            if move_data['time_seconds'] > 1.0:
                comments.append(f"time: {move_data['time_seconds']:.2f}s")
            
            if comments:
                node.comment = ", ".join(comments)
            
            board.push(move)
        
        # Save to file
        pgn_file = save_dir / f"game_{game_number:02d}.pgn"
        with open(pgn_file, 'w') as f:
            f.write(str(game))
        
        logger.debug(f"Saved PGN: {pgn_file}")
        
    except Exception as e:
        logger.error(f"Failed to save PGN for game {game_number}: {e}")

def save_games_summary(games: List[Dict[str, Any]], save_dir: Path):
    """Save summary of all games"""
    try:
        # Calculate summary statistics
        total_games = len(games)
        results_count = {}
        terminations_count = {}
        total_moves = 0
        total_captures = 0
        total_checks = 0
        
        for game in games:
            result = game['result']
            termination = game['termination']
            
            results_count[result] = results_count.get(result, 0) + 1
            terminations_count[termination] = terminations_count.get(termination, 0) + 1
            
            total_moves += game['moves']
            total_captures += game['statistics']['captures']
            total_checks += game['statistics']['checks']
        
        # Create summary
        summary = {
            'metadata': {
                'total_games': total_games,
                'generated_at': datetime.now().isoformat(),
                'avg_moves_per_game': total_moves / total_games if total_games > 0 else 0,
                'total_captures': total_captures,
                'total_checks': total_checks
            },
            'results': results_count,
            'terminations': terminations_count,
            'games': [
                {
                    'game_id': game['game_id'],
                    'agent_color': game['agent_color'],
                    'result': game['result'],
                    'termination': game['termination'],
                    'moves': game['moves'],
                    'captures': game['statistics']['captures'],
                    'checks': game['statistics']['checks'],
                    'avg_agent_time': game['statistics'].get('avg_agent_time', game['statistics'].get('avg_white_time', 0)),
                    'total_time': game['statistics']['total_time']
                }
                for game in games
            ]
        }
        
        # Save summary
        summary_file = save_dir / "games_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Games summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save games summary: {e}")
