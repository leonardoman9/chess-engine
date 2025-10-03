"""
Self-play training environment for chess DQN
"""
import chess
import chess.engine
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from ..agents.dqn_agent import DQNAgent
from ..utils.action_utils import board_to_tensor, action_to_move, move_to_action


@dataclass
class GameResult:
    """Result of a single game"""
    winner: Optional[str]  # 'white', 'black', or None for draw
    moves: int
    game_length: float  # seconds
    final_position: str  # FEN
    termination: str  # 'checkmate', 'stalemate', 'draw', 'timeout'


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Game settings
    max_moves: int = 200
    game_timeout: float = 30.0  # seconds per game
    
    # Training settings
    games_per_episode: int = 10
    training_frequency: int = 5  # train every N games
    target_update_frequency: int = 100  # update target network every N training steps
    
    # Evaluation settings
    eval_frequency: int = 50  # evaluate every N games
    eval_games: int = 10
    stockfish_depth: int = 1
    stockfish_time: float = 0.1
    
    # Logging
    log_frequency: int = 10  # log every N games
    checkpoint_frequency: int = 100  # save checkpoint every N games


class SelfPlayTrainer:
    """Self-play trainer for chess DQN"""
    
    def __init__(
        self,
        agent: DQNAgent,
        config: TrainingConfig,
        stockfish_path: str = "/usr/games/stockfish",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints"
    ):
        self.agent = agent
        self.config = config
        self.stockfish_path = stockfish_path
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Training statistics
        self.games_played = 0
        self.training_steps = 0
        self.total_rewards = []
        self.game_lengths = []
        self.win_rates = {'white': 0, 'black': 0, 'draw': 0}
        
        # Stockfish engine (lazy initialization)
        self._stockfish_engine = None
    
    @property
    def stockfish_engine(self):
        """Lazy initialization of Stockfish engine"""
        if self._stockfish_engine is None:
            try:
                self._stockfish_engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                self.logger.warning(f"Could not initialize Stockfish: {e}")
                self._stockfish_engine = None
        return self._stockfish_engine
    
    def play_game(self, opponent_agent: Optional[DQNAgent] = None) -> Tuple[GameResult, List[Tuple]]:
        """
        Play a single game and return result + experiences
        
        Args:
            opponent_agent: If None, agent plays against itself
            
        Returns:
            GameResult and list of (state, action, reward, next_state, done) tuples
        """
        board = chess.Board()
        experiences = []
        start_time = time.time()
        
        # Determine players
        white_agent = self.agent
        black_agent = opponent_agent if opponent_agent else self.agent
        
        move_count = 0
        game_reward = 0
        
        while not board.is_game_over() and move_count < self.config.max_moves:
            if time.time() - start_time > self.config.game_timeout:
                break
                
            # Current player
            current_agent = white_agent if board.turn else black_agent
            
            # Get current state
            state = board_to_tensor(board)
            
            # Agent selects action
            action = current_agent.select_action(board)
            
            # Convert action to move
            try:
                move = action_to_move(action, board)
                if move not in board.legal_moves:
                    # Invalid move - penalize and select random legal move
                    move = random.choice(list(board.legal_moves))
                    reward = -0.1  # Small penalty for invalid moves
                else:
                    reward = 0.01  # Small reward for valid moves
            except:
                # Fallback to random move
                move = random.choice(list(board.legal_moves))
                reward = -0.1
            
            # Make move
            board.push(move)
            next_state = board_to_tensor(board)
            
            # Check if game is over
            done = board.is_game_over()
            
            # Calculate final reward
            if done:
                if board.is_checkmate():
                    # Reward winner, penalize loser
                    if board.turn:  # Black wins (white to move but checkmated)
                        reward = -1.0 if current_agent == white_agent else 1.0
                    else:  # White wins (black to move but checkmated)
                        reward = 1.0 if current_agent == white_agent else -1.0
                elif board.is_stalemate() or board.is_insufficient_material():
                    reward = 0.0  # Draw
                else:
                    reward = 0.0  # Other draw conditions
            
            # Store experience (only for the agent we're training)
            if current_agent == self.agent:
                experiences.append((state, action, reward, next_state, done))
                game_reward += reward
            
            move_count += 1
        
        # Determine game result
        if board.is_checkmate():
            winner = 'white' if not board.turn else 'black'
            termination = 'checkmate'
        elif board.is_stalemate():
            winner = None
            termination = 'stalemate'
        elif board.is_insufficient_material():
            winner = None
            termination = 'draw'
        elif move_count >= self.config.max_moves:
            winner = None
            termination = 'timeout'
        else:
            winner = None
            termination = 'draw'
        
        game_time = time.time() - start_time
        result = GameResult(
            winner=winner,
            moves=move_count,
            game_length=game_time,
            final_position=board.fen(),
            termination=termination
        )
        
        return result, experiences
    
    def evaluate_against_stockfish(self, num_games: int = 10) -> Dict[str, float]:
        """Evaluate agent against Stockfish"""
        if not self.stockfish_engine:
            self.logger.warning("Stockfish not available for evaluation")
            return {'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0}
        
        results = {'wins': 0, 'draws': 0, 'losses': 0}
        
        for game_idx in range(num_games):
            board = chess.Board()
            agent_is_white = game_idx % 2 == 0  # Alternate colors
            
            while not board.is_game_over():
                if (board.turn and agent_is_white) or (not board.turn and not agent_is_white):
                    # Agent's turn
                    action = self.agent.select_action(board, deterministic=True)  # No exploration
                    try:
                        move = action_to_move(action, board)
                        if move not in board.legal_moves:
                            move = random.choice(list(board.legal_moves))
                    except:
                        move = random.choice(list(board.legal_moves))
                else:
                    # Stockfish's turn
                    try:
                        result = self.stockfish_engine.play(
                            board, 
                            chess.engine.Limit(depth=self.config.stockfish_depth, time=self.config.stockfish_time)
                        )
                        move = result.move
                    except:
                        move = random.choice(list(board.legal_moves))
                
                board.push(move)
            
            # Determine result from agent's perspective
            if board.is_checkmate():
                if (board.turn and not agent_is_white) or (not board.turn and agent_is_white):
                    results['wins'] += 1  # Agent won
                else:
                    results['losses'] += 1  # Agent lost
            else:
                results['draws'] += 1  # Draw
        
        total_games = sum(results.values())
        return {
            'win_rate': results['wins'] / total_games,
            'draw_rate': results['draws'] / total_games,
            'loss_rate': results['losses'] / total_games,
            'total_games': total_games
        }
    
    def train_episode(self, num_games: int) -> Dict[str, Any]:
        """Train for one episode (multiple games)"""
        episode_stats = {
            'games_played': 0,
            'total_reward': 0,
            'avg_game_length': 0,
            'win_rate': 0,
            'training_loss': 0,
            'training_steps': 0
        }
        
        episode_rewards = []
        episode_lengths = []
        episode_results = []
        
        for game_idx in range(num_games):
            # Play game
            result, experiences = self.play_game()
            
            # Store experiences in replay buffer
            for exp in experiences:
                self.agent.replay_buffer.push(*exp)
            
            # Update statistics
            self.games_played += 1
            episode_stats['games_played'] += 1
            
            game_reward = sum(exp[2] for exp in experiences)  # Sum of rewards
            episode_rewards.append(game_reward)
            episode_lengths.append(result.moves)
            episode_results.append(result.winner)
            
            # Train agent
            if (self.games_played % self.config.training_frequency == 0 and 
                len(self.agent.replay_buffer) >= self.agent.min_buffer_size):
                
                metrics = self.agent.train_step()
                if metrics:
                    episode_stats['training_loss'] += metrics.get('loss', 0)
                    episode_stats['training_steps'] += 1
                    self.training_steps += 1
            
            # Update target network
            if self.training_steps % self.config.target_update_frequency == 0:
                self.agent.soft_update_target_network()
            
            # Logging
            if self.games_played % self.config.log_frequency == 0:
                self.logger.info(f"Game {self.games_played}: {result.termination} in {result.moves} moves")
        
        # Calculate episode statistics
        episode_stats['total_reward'] = sum(episode_rewards)
        episode_stats['avg_game_length'] = np.mean(episode_lengths)
        
        # Calculate win rates
        wins = sum(1 for r in episode_results if r == 'white')
        draws = sum(1 for r in episode_results if r is None)
        episode_stats['win_rate'] = wins / len(episode_results) if episode_results else 0
        episode_stats['draw_rate'] = draws / len(episode_results) if episode_results else 0
        
        if episode_stats['training_steps'] > 0:
            episode_stats['training_loss'] /= episode_stats['training_steps']
        
        return episode_stats
    
    def train(self, total_games: int, save_path: Optional[str] = None) -> Dict[str, List]:
        """
        Main training loop
        
        Args:
            total_games: Total number of games to play
            save_path: Path to save final model
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {total_games} games")
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'training_losses': [],
            'eval_results': []
        }
        
        games_remaining = total_games
        episode_num = 0
        
        while games_remaining > 0:
            episode_num += 1
            games_this_episode = min(self.config.games_per_episode, games_remaining)
            
            # Train episode
            episode_stats = self.train_episode(games_this_episode)
            
            # Update history
            history['episode_rewards'].append(episode_stats['total_reward'])
            history['episode_lengths'].append(episode_stats['avg_game_length'])
            history['win_rates'].append(episode_stats['win_rate'])
            history['training_losses'].append(episode_stats['training_loss'])
            
            # Evaluation
            if self.games_played % self.config.eval_frequency == 0:
                self.logger.info("Running evaluation against Stockfish...")
                eval_results = self.evaluate_against_stockfish(self.config.eval_games)
                history['eval_results'].append(eval_results)
                
                self.logger.info(
                    f"Evaluation - Win: {eval_results['win_rate']:.3f}, "
                    f"Draw: {eval_results['draw_rate']:.3f}, "
                    f"Loss: {eval_results['loss_rate']:.3f}"
                )
            
            # Checkpoint
            if self.games_played % self.config.checkpoint_frequency == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_game_{self.games_played}.pt"
                self.agent.save_checkpoint(str(checkpoint_path))
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Progress logging
            self.logger.info(
                f"Episode {episode_num}: Games {self.games_played}/{total_games}, "
                f"Avg Reward: {episode_stats['total_reward']:.3f}, "
                f"Win Rate: {episode_stats['win_rate']:.3f}, "
                f"Epsilon: {self.agent.exploration.get_epsilon():.3f}"
            )
            
            games_remaining -= games_this_episode
        
        # Save final model
        if save_path:
            self.agent.save_checkpoint(save_path)
            self.logger.info(f"Final model saved: {save_path}")
        
        # Final evaluation
        self.logger.info("Running final evaluation...")
        final_eval = self.evaluate_against_stockfish(50)  # More games for final eval
        history['final_evaluation'] = final_eval
        
        self.logger.info(
            f"Final Evaluation - Win: {final_eval['win_rate']:.3f}, "
            f"Draw: {final_eval['draw_rate']:.3f}, "
            f"Loss: {final_eval['loss_rate']:.3f}"
        )
        
        # Cleanup
        if self._stockfish_engine:
            self._stockfish_engine.quit()
        
        return history
