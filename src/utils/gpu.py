# Phase 1: MI300X GPU optimization utilities
"""MI300X GPU optimization utilities."""

import torch
import os


def setup_gpu():
    """Configure PyTorch for MI300X (AMD ROCm)."""
    if not torch.cuda.is_available():
        print("Warning: CUDA/ROCm not available")
        return torch.device("cpu")
    
    device = torch.device("cuda")
    
    # MI300X specific optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable TF32 for Ampere+ (MI300X supports similar)
    # Note: MI300X uses ROCm, but PyTorch abstracts this
    torch.set_float32_matmul_precision('high')
    
    # Memory management for 192GB VRAM
    # Allow large allocations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def optimize_model(model: torch.nn.Module, mode: str = "default") -> torch.nn.Module:
    """
    Optimize model for MI300X using PyTorch 2.0 compile.
    
    Args:
        model: PyTorch model
        mode: 'default', 'reduce-overhead', or 'max-autotune'
    """
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=mode)
            print(f"Model compiled with mode: {mode}")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    return model


def empty_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_stats():
    """Return GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }