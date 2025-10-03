"""
ELO Rating Calculator for Chess Engine Evaluation

This module implements ELO rating calculation by testing the agent against
multiple Stockfish configurations with known ELO ratings.
"""

import chess
import chess.engine
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StockfishConfig:
    """Configuration for a Stockfish opponent with known ELO"""
    name: str
    depth: int
    time_limit: float
    estimated_elo: int
    description: str

# Stockfish configurations with estimated ELO ratings
STOCKFISH_CONFIGS = {
    "beginner": StockfishConfig(
        name="beginner",
        depth=1,
        time_limit=0.01,
        estimated_elo=400,
        description="Stockfish depth=1, 0.01s - Beginner level"
    ),
    "weak": StockfishConfig(
        name="weak", 
        depth=1,
        time_limit=0.05,
        estimated_elo=600,
        description="Stockfish depth=1, 0.05s - Weak amateur"
    ),
    "amateur": StockfishConfig(
        name="amateur",
        depth=1,
        time_limit=0.1,
        estimated_elo=800,
        description="Stockfish depth=1, 0.1s - Amateur"
    ),
    "club": StockfishConfig(
        name="club",
        depth=1,
        time_limit=0.2,
        estimated_elo=1000,
        description="Stockfish depth=1, 0.2s - Club player"
    ),
    "intermediate": StockfishConfig(
        name="intermediate",
        depth=2,
        time_limit=0.1,
        estimated_elo=1200,
        description="Stockfish depth=2, 0.1s - Intermediate"
    ),
    "advanced": StockfishConfig(
        name="advanced",
        depth=2,
        time_limit=0.2,
        estimated_elo=1400,
        description="Stockfish depth=2, 0.2s - Advanced"
    ),
    "expert": StockfishConfig(
        name="expert",
        depth=3,
        time_limit=0.1,
        estimated_elo=1600,
        description="Stockfish depth=3, 0.1s - Expert"
    )
}

@dataclass
class GameResult:
    """Result of a single game"""
    agent_color: chess.Color
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    termination: str
    stockfish_config: str

@dataclass
class EvaluationResult:
    """Result of evaluation against one Stockfish configuration"""
    stockfish_config: StockfishConfig
    games_played: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    score: float  # Chess score: win=1, draw=0.5, loss=0
    games: List[GameResult]

@dataclass
class ELOEvaluation:
    """Complete ELO evaluation result"""
    estimated_elo: int
    confidence_interval: Tuple[int, int]
    evaluations: Dict[str, EvaluationResult]
    total_games: int
    overall_score: float
    timestamp: str

class ELOCalculator:
    """Calculate ELO rating by testing against multiple Stockfish levels"""
    
    def __init__(self, stockfish_path: str = "/usr/games/stockfish"):
        self.stockfish_path = stockfish_path
        
    def play_game(self, agent, stockfish_config: StockfishConfig, 
                  agent_plays_white: bool = True, max_moves: int = 200) -> GameResult:
        """Play a single game between agent and Stockfish"""
        
        board = chess.Board()
        moves_played = 0
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                while not board.is_game_over() and moves_played < max_moves:
                    if (board.turn == chess.WHITE and agent_plays_white) or \
                       (board.turn == chess.BLACK and not agent_plays_white):
                        # Agent's turn
                        action = agent.select_action(board, deterministic=True)
                        from src.utils.action_utils import action_to_move
                        move = action_to_move(action, board)
                        
                        if move is None or move not in board.legal_moves:
                            # Agent made illegal move - pick random legal move
                            legal_moves = list(board.legal_moves)
                            if legal_moves:
                                move = np.random.choice(legal_moves)
                            else:
                                break
                    else:
                        # Stockfish's turn
                        limit = chess.engine.Limit(
                            time=stockfish_config.time_limit,
                            depth=stockfish_config.depth
                        )
                        result = engine.play(board, limit)
                        move = result.move
                    
                    if move:
                        board.push(move)
                        moves_played += 1
                    else:
                        break
                        
        except Exception as e:
            logger.error(f"Error during game: {e}")
            # Return a loss for the agent in case of error
            return GameResult(
                agent_color=chess.WHITE if agent_plays_white else chess.BLACK,
                result="0-1" if agent_plays_white else "1-0",
                moves=moves_played,
                termination="error",
                stockfish_config=stockfish_config.name
            )
        
        # Determine result
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
            
        return GameResult(
            agent_color=chess.WHITE if agent_plays_white else chess.BLACK,
            result=result,
            moves=moves_played,
            termination=termination,
            stockfish_config=stockfish_config.name
        )
    
    def evaluate_against_stockfish(self, agent, stockfish_config: StockfishConfig, 
                                 num_games: int = 20) -> EvaluationResult:
        """Evaluate agent against one Stockfish configuration"""
        
        logger.info(f"Evaluating against {stockfish_config.description}")
        logger.info(f"Playing {num_games} games...")
        
        games = []
        wins = draws = losses = 0
        
        for game_idx in range(num_games):
            # Alternate colors
            agent_plays_white = game_idx % 2 == 0
            
            game_result = self.play_game(agent, stockfish_config, agent_plays_white)
            games.append(game_result)
            
            # Count results from agent's perspective
            if game_result.result == "1-0":
                if agent_plays_white:
                    wins += 1
                else:
                    losses += 1
            elif game_result.result == "0-1":
                if agent_plays_white:
                    losses += 1
                else:
                    wins += 1
            else:  # Draw
                draws += 1
                
            if (game_idx + 1) % 5 == 0:
                logger.info(f"  Games {game_idx + 1}/{num_games}: W{wins} D{draws} L{losses}")
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games
        score = (wins + 0.5 * draws) / num_games
        
        logger.info(f"Final result vs {stockfish_config.name}: W{wins} D{draws} L{losses} (Score: {score:.3f})")
        
        return EvaluationResult(
            stockfish_config=stockfish_config,
            games_played=num_games,
            wins=wins,
            draws=draws,
            losses=losses,
            win_rate=win_rate,
            draw_rate=draw_rate,
            loss_rate=loss_rate,
            score=score,
            games=games
        )
    
    def calculate_elo_from_score(self, opponent_elo: int, score: float) -> int:
        """
        Calculate ELO rating based on score against opponent
        
        Using the ELO formula: 
        Expected score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
        Solving for player_elo given actual score
        """
        if score <= 0:
            return max(100, opponent_elo - 800)  # Very low rating
        elif score >= 1:
            return opponent_elo + 800  # Very high rating
        else:
            # Solve: score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
            # player_elo = opponent_elo - 400 * log10((1 - score) / score)
            try:
                elo_diff = 400 * np.log10((1 - score) / score)
                player_elo = opponent_elo - elo_diff
                return int(max(100, min(3000, player_elo)))  # Clamp to reasonable range
            except:
                return opponent_elo  # Fallback
    
    def comprehensive_elo_evaluation(self, agent, games_per_config: int = 20,
                                   configs_to_test: List[str] = None) -> ELOEvaluation:
        """
        Perform comprehensive ELO evaluation against multiple Stockfish levels
        """
        if configs_to_test is None:
            # Test against a range of levels, focusing on likely ELO range
            configs_to_test = ["beginner", "weak", "amateur", "club", "intermediate"]
        
        logger.info("🎯 Starting Comprehensive ELO Evaluation")
        logger.info(f"Testing against {len(configs_to_test)} Stockfish configurations")
        logger.info(f"Games per configuration: {games_per_config}")
        
        evaluations = {}
        elo_estimates = []
        total_games = 0
        total_score = 0
        
        for config_name in configs_to_test:
            if config_name not in STOCKFISH_CONFIGS:
                logger.warning(f"Unknown Stockfish config: {config_name}")
                continue
                
            config = STOCKFISH_CONFIGS[config_name]
            evaluation = self.evaluate_against_stockfish(agent, config, games_per_config)
            evaluations[config_name] = evaluation
            
            # Calculate ELO estimate from this evaluation
            elo_estimate = self.calculate_elo_from_score(config.estimated_elo, evaluation.score)
            elo_estimates.append(elo_estimate)
            
            total_games += evaluation.games_played
            total_score += evaluation.score * evaluation.games_played
            
            logger.info(f"ELO estimate vs {config_name} ({config.estimated_elo}): {elo_estimate}")
        
        # Calculate overall ELO estimate
        if elo_estimates:
            estimated_elo = int(np.mean(elo_estimates))
            elo_std = np.std(elo_estimates)
            confidence_interval = (
                int(estimated_elo - 1.96 * elo_std),
                int(estimated_elo + 1.96 * elo_std)
            )
        else:
            estimated_elo = 800
            confidence_interval = (600, 1000)
        
        overall_score = total_score / total_games if total_games > 0 else 0
        
        result = ELOEvaluation(
            estimated_elo=estimated_elo,
            confidence_interval=confidence_interval,
            evaluations=evaluations,
            total_games=total_games,
            overall_score=overall_score,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("🎉 ELO Evaluation Complete!")
        logger.info(f"📊 Estimated ELO: {estimated_elo} (95% CI: {confidence_interval[0]}-{confidence_interval[1]})")
        logger.info(f"📈 Overall Score: {overall_score:.3f} ({total_games} games)")
        
        return result
    
    def save_evaluation_results(self, evaluation: ELOEvaluation, save_path: Path):
        """Save evaluation results to JSON file"""
        
        # Convert to serializable format
        data = {
            "estimated_elo": evaluation.estimated_elo,
            "confidence_interval": evaluation.confidence_interval,
            "total_games": evaluation.total_games,
            "overall_score": evaluation.overall_score,
            "timestamp": evaluation.timestamp,
            "evaluations": {}
        }
        
        for config_name, eval_result in evaluation.evaluations.items():
            data["evaluations"][config_name] = {
                "stockfish_elo": eval_result.stockfish_config.estimated_elo,
                "description": eval_result.stockfish_config.description,
                "games_played": eval_result.games_played,
                "wins": eval_result.wins,
                "draws": eval_result.draws,
                "losses": eval_result.losses,
                "win_rate": eval_result.win_rate,
                "draw_rate": eval_result.draw_rate,
                "loss_rate": eval_result.loss_rate,
                "score": eval_result.score,
                "games": [
                    {
                        "agent_color": "white" if game.agent_color == chess.WHITE else "black",
                        "result": game.result,
                        "moves": game.moves,
                        "termination": game.termination
                    }
                    for game in eval_result.games
                ]
            }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Evaluation results saved to {save_path}")


def quick_elo_estimate(agent, stockfish_path: str = "/usr/games/stockfish") -> int:
    """Quick ELO estimate using fewer games"""
    calculator = ELOCalculator(stockfish_path)
    
    # Test against 3 levels with fewer games for quick estimate
    evaluation = calculator.comprehensive_elo_evaluation(
        agent, 
        games_per_config=10,
        configs_to_test=["weak", "amateur", "club"]
    )
    
    return evaluation.estimated_elo


def full_elo_evaluation(agent, save_dir: Path, stockfish_path: str = "/usr/games/stockfish") -> ELOEvaluation:
    """Full ELO evaluation with comprehensive testing"""
    calculator = ELOCalculator(stockfish_path)
    
    # Test against all relevant levels
    evaluation = calculator.comprehensive_elo_evaluation(
        agent,
        games_per_config=20,
        configs_to_test=["beginner", "weak", "amateur", "club", "intermediate", "advanced"]
    )
    
    # Save results
    calculator.save_evaluation_results(evaluation, save_dir / "elo_evaluation.json")
    
    return evaluation
