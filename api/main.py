from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chess Engine API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_value = nn.Linear(512, 1)
        self.fc_policy = nn.Linear(512, 4672)  # All possible moves

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = torch.tanh(self.fc_value(x))
        policy = F.log_softmax(self.fc_policy(x), dim=1)

        return value, policy

# Initialize the model
logger.info("Initializing chess engine model...")
model = ChessNet()
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoint_100.pt")
logger.info(f"Loading model from: {model_path}")
try:
    model.load_state_dict(torch.load(model_path))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise
model.eval()
logger.info("Model set to evaluation mode")

def board_to_input(board):
    # Convert chess board to input tensor
    input_tensor = torch.zeros(13, 8, 8)  # 13 channels for piece types and additional features
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = divmod(i, 8)
            input_tensor[piece_to_idx[piece.symbol()], rank, file] = 1
    
    # Add current player channel
    input_tensor[12] = 1 if board.turn else 0
    
    return input_tensor.unsqueeze(0)

def get_best_move(board, model):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    input_tensor = board_to_input(board)
    with torch.no_grad():
        value, policy = model(input_tensor)
        policy = policy.exp()[0]  # Convert log probabilities back to probabilities
    
    # Map each legal move to its policy score
    move_scores = {}
    for move in legal_moves:
        # Convert move to index in policy vector
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square)
        if piece is None:
            continue
            
        # Calculate move index based on piece type and squares
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }[piece.symbol()]
        
        # Each piece type has 64*64 possible moves
        move_idx = piece_idx * 64 * 64 + from_square * 64 + to_square
        if move_idx < 4672:  # Make sure we don't exceed the policy size
            move_scores[move] = policy[move_idx].item()
    
    if not move_scores:
        return None
        
    # Select the move with highest policy score
    best_move = max(move_scores.items(), key=lambda x: x[1])[0]
    return best_move

class MoveRequest(BaseModel):
    fen: str
    move: Optional[str] = None
    time_limit: float = 1.0

class GameState(BaseModel):
    fen: str
    legal_moves: List[str]
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool

@app.get("/")
async def root():
    logger.info("Received root request")
    return {"message": "Chess Engine API"}

@app.post("/move", response_model=dict)
async def make_move(request: MoveRequest):
    logger.info(f"Received move request: {request}")
    try:
        board = chess.Board(request.fen)
        
        # If a move is provided, validate and make it
        if request.move:
            try:
                move = chess.Move.from_uci(request.move)
                if move in board.legal_moves:
                    board.push(move)
                    logger.info(f"Move {move} made successfully")
                    return {
                        "success": True,
                        "fen": board.fen(),
                        "move": move.uci()
                    }
                else:
                    logger.warning(f"Invalid move: {move}")
                    return {
                        "success": False,
                        "error": "Invalid move"
                    }
            except ValueError:
                logger.error(f"Invalid move format: {request.move}")
                return {
                    "success": False,
                    "error": "Invalid move format"
                }
        else:
            logger.warning("No move provided")
            return {
                "success": False,
                "error": "No move provided"
            }
    except Exception as e:
        logger.error(f"Error in make_move: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/engine_move", response_model=dict)
async def get_engine_move(request: MoveRequest):
    logger.info(f"Received engine move request: {request}")
    try:
        board = chess.Board(request.fen)
        
        # First check if the game is over
        if board.is_game_over():
            reason = "Checkmate" if board.is_checkmate() else "Stalemate" if board.is_stalemate() else "Game Over"
            logger.info(f"Game is over: {reason}")
            return {
                "success": False,
                "error": f"{reason} - No moves available",
                "game_over": True,
                "reason": reason
            }
            
        best_move = get_best_move(board, model)
        
        if best_move is None:
            logger.warning("No legal moves available")
            return {
                "success": False,
                "error": "No legal moves available"
            }
        
        board.push(best_move)
        logger.info(f"Engine made move: {best_move}")
        return {
            "success": True,
            "fen": board.fen(),
            "move": best_move.uci()
        }
    except Exception as e:
        logger.error(f"Error in get_engine_move: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/game-state", response_model=GameState)
async def get_game_state(fen: str):
    try:
        board = chess.Board(fen)
        return {
            "fen": board.fen(),
            "legal_moves": [move.uci() for move in board.legal_moves],
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate", response_model=dict)
async def evaluate_position(fen: str, time_limit: float = 1.0):
    logger.info(f"Received evaluation request for position: {fen}")
    try:
        board = chess.Board(fen)
        input_tensor = board_to_input(board)
        
        # Log the board state for debugging
        logger.info(f"Current board state:\n{board}")
        
        with torch.no_grad():
            value, _ = model(input_tensor)
            raw_score = value.item()
            
            # Scale the raw score (-1 to 1) to centipawns (-2000 to 2000)
            # This gives us a more meaningful range for chess evaluations
            scaled_score = raw_score * 2000
            
            # If black to move, invert the score to maintain perspective
            if not board.turn:
                scaled_score = -scaled_score
            
            logger.info(f"Raw model output: {raw_score}")
            logger.info(f"Scaled evaluation: {scaled_score} centipawns")
            
            # Count material difference for comparison
            material_score = 0
            piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.symbol().upper()]
                    material_score += value if piece.color else -value
            
            logger.info(f"Material difference: {material_score} pawns")
            
        return {
            "success": True,
            "score": int(scaled_score),
            "material_difference": material_score,
            "depth": 1
        }
    except Exception as e:
        logger.error(f"Error in evaluate_position: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 