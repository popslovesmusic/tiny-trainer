#!/usr/bin/env python3
'''
K80 Compatibility Test for Tiny Agent Trainer
'''

import torch
import sys

def test_k80_compatibility():
    print("Tesla K80 Compatibility Test")
    print("=" * 40)
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        print("Install CUDA-enabled PyTorch:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print(f"SUCCESS: CUDA Available: {torch.cuda.device_count()} GPU(s)")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory/1e9:.1f}GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        if "K80" in props.name:
            print("  SUCCESS: Tesla K80 detected!")
            
            # Test basic operations
            try:
                device = torch.device(f'cuda:{i}')
                x = torch.randn(1000, 1000, device=device)
                y = torch.mm(x, x)
                print("  SUCCESS: Basic tensor operations working")
                
                # Test neural network layer
                import torch.nn as nn
                layer = nn.Linear(1000, 100).to(device)
                output = layer(x[:100])
                print("  SUCCESS: Neural network layers working")
                
                memory_used = torch.cuda.memory_allocated(i) / 1e9
                print(f"  INFO: Memory used: {memory_used:.2f}GB")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                return False
    
    print("\nSUCCESS: K80 compatibility confirmed!")
    print("\nRecommended PyTorch versions for K80:")
    print("- PyTorch 1.12.1 + CUDA 11.3 (most stable)")
    print("- PyTorch 2.0+ + CUDA 11.8 (latest features)")
    
    return True

if __name__ == "__main__":
    test_k80_compatibility()
