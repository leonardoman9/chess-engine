#!/usr/bin/env python3
"""
Test script to verify GPU training works with all configurations
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, timeout=300):
    """Run a command with timeout"""
    print(f"\n🧪 Testing: {cmd}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            return True
        else:
            print("❌ FAILED")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def test_gpu_commands():
    """Test all GPU training commands"""
    
    # Test configurations
    experiments = ["baseline_small", "baseline_medium", "baseline_large"]
    gpu_id = "0"  # Use GPU 0 for testing
    
    results = {}
    
    print("🚀 Testing GPU Training Commands")
    print("=" * 60)
    
    # 1. Test basic GPU setup
    print("\n📋 Phase 1: Basic GPU Setup")
    cmd = f"make test-gpu GPU={gpu_id}"
    results["test-gpu"] = run_command(cmd, timeout=120)
    
    # 2. Test Hydra GPU training with different experiments
    print("\n📋 Phase 2: Hydra GPU Training")
    for exp in experiments:
        cmd = f"make train-hydra-gpu GPU={gpu_id} EXP={exp} PARAMS='experiment.total_games=5'"
        results[f"hydra-gpu-{exp}"] = run_command(cmd, timeout=300)
        time.sleep(5)  # Brief pause between tests
    
    # 3. Test custom parameters
    print("\n📋 Phase 3: Custom Parameters")
    cmd = f"make train-hydra-gpu GPU={gpu_id} PARAMS='experiment=baseline_small experiment.total_games=3 device=cuda:0'"
    results["hydra-gpu-custom"] = run_command(cmd, timeout=300)
    
    # 4. Test ELO evaluation on GPU
    print("\n📋 Phase 4: ELO Evaluation")
    # First find a checkpoint to test
    results_dirs = list(Path("results").glob("baseline_*"))
    if results_dirs:
        latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_path = latest_dir / "final_model.pt"
        if checkpoint_path.exists():
            cmd = f"make evaluate-elo-gpu GPU={gpu_id} CHECKPOINT={checkpoint_path} QUICK=1"
            results["elo-gpu"] = run_command(cmd, timeout=180)
        else:
            print("⚠️ No checkpoint found for ELO testing")
            results["elo-gpu"] = False
    else:
        print("⚠️ No results directory found for ELO testing")
        results["elo-gpu"] = False
    
    # 5. Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All GPU configurations working correctly!")
        return True
    else:
        print("⚠️ Some GPU configurations need attention")
        return False

if __name__ == "__main__":
    success = test_gpu_commands()
    sys.exit(0 if success else 1)
