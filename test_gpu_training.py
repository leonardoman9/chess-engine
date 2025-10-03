#!/usr/bin/env python3
"""
Test script to verify the updated GPU training commands work correctly
"""

import subprocess
import sys
import json
from pathlib import Path

def test_gpu_training_command():
    """Test the updated train-gpu-quick command"""
    
    print("🧪 Testing Updated GPU Training Command")
    print("=" * 60)
    
    # Test command with very few games for quick verification
    cmd = "make train-gpu-quick GPU=0"
    
    print(f"Running: {cmd}")
    print("This should now use train_hydra.py and generate all files...")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            print("❌ Command failed!")
            print("STDERR:", result.stderr[-1000:])
            return False
        
        print("✅ Command completed successfully!")
        
        # Find the latest results directory
        results_dirs = list(Path("results").glob("baseline_*"))
        if not results_dirs:
            print("❌ No results directory found!")
            return False
        
        latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        print(f"📁 Latest results directory: {latest_dir}")
        
        # Check for required files
        required_files = [
            "experiment_info.json",
            "experiment_config.yaml", 
            "final_model.pt",
            "final_model_buffer.pt",
            "training_history.json",
            "elo_evaluation.json"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = latest_dir / file
            if file_path.exists():
                print(f"✅ {file} - Found ({file_path.stat().st_size} bytes)")
            else:
                print(f"❌ {file} - Missing!")
                missing_files.append(file)
        
        # Check for sample games directory
        sample_games_dir = latest_dir / "sample_games"
        if sample_games_dir.exists():
            pgn_files = list(sample_games_dir.glob("*.pgn"))
            print(f"✅ sample_games directory - Found with {len(pgn_files)} PGN files")
            
            # Check a sample PGN file
            if pgn_files:
                sample_pgn = pgn_files[0]
                with open(sample_pgn, 'r') as f:
                    content = f.read()
                    if "Agent (Deterministic)" in content and "Agent (Exploratory)" in content:
                        print("✅ PGN files contain self-play games")
                    else:
                        print("⚠️ PGN files may not be self-play format")
        else:
            print("❌ sample_games directory - Missing!")
            missing_files.append("sample_games/")
        
        # Check training history content
        history_file = latest_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    if 'elo_evaluation' in history:
                        print("✅ Training history contains ELO evaluation")
                    else:
                        print("⚠️ Training history missing ELO evaluation")
            except Exception as e:
                print(f"⚠️ Could not parse training history: {e}")
        
        # Summary
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            return False
        else:
            print("\n🎉 All expected files and features are present!")
            print("✅ The updated GPU training command works correctly!")
            return True
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out!")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu_training_command()
    
    if success:
        print("\n🎯 RECOMMENDATION:")
        print("The updated commands are working correctly. Use these instead:")
        print("  make train-gpu-quick GPU=0     # Quick test (100 games)")
        print("  make train-gpu-1000 GPU=0      # Medium training (1000 games)")  
        print("  make train-gpu-5000 GPU=0      # Long training (5000 games)")
        print("  make train-hydra-gpu-small GPU=0 GAMES=500  # Custom games")
    else:
        print("\n⚠️ Issues found. Check the output above.")
    
    sys.exit(0 if success else 1)
