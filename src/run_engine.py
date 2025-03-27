#!/usr/bin/env python3
import torch
from src.model import load_model
from src.agent import ChessAgent
from src.interface import ChessCLI

def main():
    # Load the trained model
    model_path = "models/checkpoint_10.pt"  # Update this path to your downloaded model
    print(f"Loading model from {model_path}...")
    
    # Create agent with the loaded model
    model = load_model(model_path)
    agent = ChessAgent(model)
    
    # Start the CLI interface
    cli = ChessCLI(model_path)
    cli.play(human_plays_white=True)

if __name__ == "__main__":
    main() 