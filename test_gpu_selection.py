#!/usr/bin/env python3
"""
Test script to verify GPU selection works correctly
"""

import torch
import os
import sys

def test_gpu_selection():
    """Test which GPU is being used"""
    
    print("🖥️  GPU Selection Test")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        
        # Check environment variables
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Test tensor on GPU
        try:
            device = torch.device('cuda')
            test_tensor = torch.randn(100, 100).to(device)
            print(f"✅ Successfully created tensor on: {test_tensor.device}")
            
            # Simple computation to verify GPU is working
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✅ GPU computation successful: {result.shape}")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
    else:
        print("⚠️  No CUDA GPUs available - running on CPU")
        
        # Test CPU fallback
        test_tensor = torch.randn(100, 100)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ CPU computation successful: {result.shape}")
    
    print("=" * 50)
    print("🎯 GPU Selection Test Complete")

if __name__ == "__main__":
    test_gpu_selection()
