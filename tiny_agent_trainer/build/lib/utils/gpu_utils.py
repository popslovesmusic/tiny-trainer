"""
GPU utilities for optimal hardware utilization and monitoring.
"""

import torch
import torch.nn as nn
import subprocess
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information container."""
    id: int
    name: str
    memory_total: int
    memory_free: int
    memory_used: int
    utilization: float
    temperature: Optional[int] = None
    power_usage: Optional[float] = None
    compute_capability: Tuple[int, int] = (0, 0)
    is_available: bool = True


class GPUMonitor:
    """Monitor GPU usage during training."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU monitoring disabled")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}")
                break
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current GPU metrics."""
        if not torch.cuda.is_available():
            return {}
        
        metrics = {
            'timestamp': time.time(),
            'gpus': []
        }
        
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_metrics = {
                    'id': i,
                    'memory_allocated_mb': memory_allocated / 1024 / 1024,
                    'memory_reserved_mb': memory_reserved / 1024 / 1024,
                    'memory_total_mb': memory_total / 1024 / 1024,
                    'memory_utilization': memory_allocated / memory_total,
                }
                
                metrics['gpus'].append(gpu_metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics for GPU {i}: {e}")
        
        return metrics
    
    def get_peak_memory_usage(self) -> Dict[int, float]:
        """Get peak memory usage for each GPU."""
        peak_usage = {}
        
        for metric in self.metrics:
            for gpu in metric.get('gpus', []):
                gpu_id = gpu['id']
                usage = gpu['memory_utilization']
                
                if gpu_id not in peak_usage or usage > peak_usage[gpu_id]:
                    peak_usage[gpu_id] = usage
        
        return peak_usage
    
    def save_metrics(self, filepath: str):
        """Save monitoring metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"GPU metrics saved to {filepath}")


class GPUOptimizer:
    """Optimize GPU settings and memory management."""
    
    def __init__(self):
        self.original_settings = {}
    
    def optimize_for_training(self):
        """Apply optimal settings for training."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU optimization skipped")
            return
        
        try:
            # Enable cudNN benchmark mode for consistent input sizes
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cudNN benchmark mode")
            
            # Enable cudNN deterministic mode for reproducibility (optional)
            # torch.backends.cudnn.deterministic = True
            
            # Optimize memory allocator
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
            
            # Set memory fraction if needed (uncomment if memory issues)
            # torch.cuda.set_per_process_memory_fraction(0.9)
            
        except Exception as e:
            logger.error(f"Error optimizing GPU settings: {e}")
    
    def optimize_memory_allocation(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Optimize memory allocation for model."""
        if not torch.cuda.is_available():
            return {}
        
        device = next(model.parameters()).device
        
        optimization_info = {
            'device': str(device),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
        
        try:
            # Test forward pass memory usage
            model.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(sample_input.to(device))
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            optimization_info['inference_memory_mb'] = peak_memory
            
            logger.info(f"Model memory usage: {peak_memory:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error measuring memory usage: {e}")
        
        return optimization_info
    
    def suggest_batch_size(self, model: nn.Module, sample_input: torch.Tensor, 
                          target_memory_usage: float = 0.8) -> int:
        """Suggest optimal batch size based on GPU memory."""
        if not torch.cuda.is_available():
            return 32  # Default for CPU
        
        device = next(model.parameters()).device
        total_memory = torch.cuda.get_device_properties(device).total_memory
        target_memory = total_memory * target_memory_usage
        
        # Binary search for optimal batch size
        min_batch_size = 1
        max_batch_size = 1024
        optimal_batch_size = 1
        
        while min_batch_size <= max_batch_size:
            batch_size = (min_batch_size + max_batch_size) // 2
            
            try:
                # Test batch size
                torch.cuda.empty_cache()
                batch_input = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
                batch_input = batch_input.to(device)
                
                model.eval()
                with torch.no_grad():
                    _ = model(batch_input)
                
                current_memory = torch.cuda.memory_allocated()
                
                if current_memory <= target_memory:
                    optimal_batch_size = batch_size
                    min_batch_size = batch_size + 1
                else:
                    max_batch_size = batch_size - 1
                
                del batch_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_batch_size = batch_size - 1
                else:
                    logger.error(f"Error testing batch size {batch_size}: {e}")
                    break
        
        logger.info(f"Suggested batch size: {optimal_batch_size}")
        return optimal_batch_size


class MultiGPUManager:
    """Manage multiple GPU training and inference."""
    
    def __init__(self):
        self.device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.primary_device = 0 if self.device_ids else None
    
    def setup_data_parallel(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """Setup model for data parallel training."""
        if not torch.cuda.is_available() or len(self.device_ids) < 2:
            logger.info("Multi-GPU training not available, using single GPU/CPU")
            return model
        
        device_ids = device_ids or self.device_ids
        
        if len(device_ids) > 1:
            logger.info(f"Setting up DataParallel training on GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        
        return model
    
    def get_device_for_inference(self, prefer_gpu_id: Optional[int] = None) -> torch.device:
        """Get optimal device for inference."""
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        if prefer_gpu_id is not None and prefer_gpu_id in self.device_ids:
            return torch.device(f'cuda:{prefer_gpu_id}')
        
        # Find GPU with most free memory
        best_gpu = 0
        best_free_memory = 0
        
        for gpu_id in self.device_ids:
            torch.cuda.set_device(gpu_id)
            props = torch.cuda.get_device_properties(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            free_memory = props.total_memory - allocated
            
            if free_memory > best_free_memory:
                best_free_memory = free_memory
                best_gpu = gpu_id
        
        return torch.device(f'cuda:{best_gpu}')
    
    def balance_load(self, models: List[nn.Module]) -> List[torch.device]:
        """Balance models across available GPUs."""
        if not self.device_ids or not models:
            return [torch.device('cpu')] * len(models)
        
        devices = []
        for i, model in enumerate(models):
            gpu_id = self.device_ids[i % len(self.device_ids)]
            devices.append(torch.device(f'cuda:{gpu_id}'))
        
        return devices


@contextmanager
def gpu_memory_context():
    """Context manager for GPU memory monitoring."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.debug(f"Peak GPU memory usage: {peak_memory:.1f}MB")
            torch.cuda.empty_cache()


def get_gpu_info() -> List[GPUInfo]:
    """Get detailed information about available GPUs."""
    gpus = []
    
    if not torch.cuda.is_available():
        return gpus
    
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_free = memory_total - memory_allocated
            
            gpu_info = GPUInfo(
                id=i,
                name=props.name,
                memory_total=memory_total,
                memory_used=memory_allocated,
                memory_free=memory_free,
                utilization=memory_allocated / memory_total,
                compute_capability=(props.major, props.minor)
            )
            
            gpus.append(gpu_info)
            
        except Exception as e:
            logger.error(f"Error getting info for GPU {i}: {e}")
    
    return gpus


def check_gpu_compatibility(min_compute_capability: Tuple[int, int] = (3, 5)) -> bool:
    """Check if GPUs meet minimum compute capability requirements."""
    if not torch.cuda.is_available():
        return False
    
    gpus = get_gpu_info()
    
    for gpu in gpus:
        if (gpu.compute_capability[0], gpu.compute_capability[1]) >= min_compute_capability:
            return True
    
    logger.warning(f"No GPU found with compute capability >= {min_compute_capability}")
    return False


def optimize_gpu_settings():
    """Apply global GPU optimizations."""
    if not torch.cuda.is_available():
        return
    
    optimizer = GPUOptimizer()
    optimizer.optimize_for_training()


def print_gpu_summary():
    """Print a summary of available GPU resources."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    gpus = get_gpu_info()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        print(f"  Memory: {gpu.memory_used/1e9:.1f}GB / {gpu.memory_total/1e9:.1f}GB used")
        print(f"  Utilization: {gpu.utilization*100:.1f}%")


# Compatibility functions
def get_device_info() -> Dict[str, Any]:
    """Get device information (compatible with existing code)."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': []
    }
    
    gpus = get_gpu_info()
    for gpu in gpus:
        gpu_dict = {
            'id': gpu.id,
            'name': gpu.name,
            'memory_total': gpu.memory_total,
            'memory_free': gpu.memory_free,
            'compute_capability': gpu.compute_capability[0] * 10 + gpu.compute_capability[1]
        }
        info['gpus'].append(gpu_dict)
    
    return info


def select_best_device() -> torch.device:
    """Select the best available device."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    gpus = get_gpu_info()
    if not gpus:
        return torch.device('cpu')
    
    # Select GPU with most free memory
    best_gpu = max(gpus, key=lambda g: g.memory_free)
    
    logger.info(f"Selected GPU {best_gpu.id}: {best_gpu.name} "
               f"({best_gpu.memory_free/1e9:.1f}GB free)")
    
    return torch.device(f'cuda:{best_gpu.id}')


if __name__ == "__main__":
    # Test GPU utilities
    print("GPU Utilities Test")
    print("=" * 30)
    
    print_gpu_summary()
    
    print(f"\nBest device: {select_best_device()}")
    print(f"GPU compatibility (3.5+): {check_gpu_compatibility()}")
    
    # Test monitoring
    monitor = GPUMonitor(update_interval=0.5)
    monitor.start_monitoring()
    
    time.sleep(2)
    
    monitor.stop_monitoring()
    print(f"Collected {len(monitor.metrics)} monitoring samples")
