"""
GPU-aware training system with proper error handling and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
import time
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and optimization."""
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpus': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    'id': i,
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'memory_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                    'compute_capability': props.major * 10 + props.minor
                }
                info['gpus'].append(gpu_info)
        
        return info
    
    @staticmethod
    def select_best_device() -> torch.device:
        """Select the best available device."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
        
        device_info = GPUManager.get_device_info()
        
        # Select GPU with most free memory
        best_gpu = max(device_info['gpus'], key=lambda x: x['memory_free'])
        device = torch.device(f'cuda:{best_gpu["id"]}')
        
        logger.info(f"Selected device: {device} ({best_gpu['name']}) "
                   f"with {best_gpu['memory_free']/1e9:.1f}GB free memory")
        
        return device
    
    @staticmethod
    def optimize_batch_size(model: nn.Module, sample_input: torch.Tensor, 
                           device: torch.device, max_batch_size: int = 1024) -> int:
        """Automatically find optimal batch size for the given model and device."""
        if device.type == 'cpu':
            return min(32, max_batch_size)  # Conservative for CPU
        
        model.eval()
        batch_size = 1
        
        try:
            with torch.no_grad():
                while batch_size <= max_batch_size:
                    try:
                        # Test with current batch size
                        batch_input = sample_input.repeat(batch_size, 1).to(device)
                        _ = model(batch_input)
                        
                        # Clear cache and try larger batch
                        torch.cuda.empty_cache()
                        batch_size *= 2
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            batch_size //= 4  # Go back to safe size
                            break
                        else:
                            raise e
            
            batch_size = max(1, min(batch_size, max_batch_size))
            logger.info(f"Optimal batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return 16  # Safe default


class TrainingMetrics:
    """Track and manage training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.best_metrics = {}
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            self.metrics[key].append(float(value))
    
    def end_epoch(self, epoch: int):
        """Compute epoch-level metrics."""
        for key, values in self.metrics.items():
            if values:  # Only if we have values for this epoch
                epoch_avg = np.mean(values[-len(values):])
                self.epoch_metrics[key].append(epoch_avg)
                
                # Track best metrics
                if key not in self.best_metrics or epoch_avg < self.best_metrics[key]:
                    self.best_metrics[key] = epoch_avg
        
        # Log epoch summary
        logger.info(f"Epoch {epoch} completed:")
        for key, values in self.epoch_metrics.items():
            if values:
                logger.info(f"  {key}: {values[-1]:.4f} (best: {self.best_metrics.get(key, float('inf')):.4f})")
    
    def save(self, filepath: Path):
        """Save metrics to file."""
        data = {
            'metrics': dict(self.metrics),
            'epoch_metrics': dict(self.epoch_metrics),
            'best_metrics': dict(self.best_metrics)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.epoch_metrics = defaultdict(list, data.get('epoch_metrics', {}))
        self.best_metrics = data.get('best_metrics', {})


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.stopped = False
    
    def __call__(self, metric_value: float) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            improved = metric_value < self.best_value - self.min_delta
        else:
            improved = metric_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.stopped = True
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return self.stopped


class Trainer:
    """Main training orchestrator with GPU support and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = GPUManager.select_best_device()
        self.metrics = TrainingMetrics()
        self.early_stopping = None
        
        # Training configuration
        self.num_epochs = config.get('num_epochs', 200)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        self.save_every = config.get('save_every', 50)
        
        # Early stopping
        if config.get('early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 15),
                min_delta=config.get('early_stopping_min_delta', 0.001)
            )
        
        # Mixed precision training
        self.use_mixed_precision = (
            config.get('mixed_precision', True) and 
            self.device.type == 'cuda' and 
            torch.cuda.is_available()
        )
        
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
    
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Setup optimizer with proper parameters."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.config.get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.num_epochs // 3, gamma=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    attention_mask = None
                else:
                    inputs, targets, attention_mask = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs, attention_mask=attention_mask)
                        loss = criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(inputs, attention_mask=attention_mask)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                    
                    optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                if outputs.dim() > 1:  # Classification
                    predicted = outputs.argmax(dim=1)
                    total_correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
                
                # Log batch progress
                if batch_idx % 100 == 0 and batch_idx > 0:
                    current_loss = total_loss / (batch_idx + 1)
                    logger.debug(f"Batch {batch_idx}/{len(dataloader)}: Loss = {current_loss:.4f}")
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warning(f"CUDA OOM in batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module) -> Dict[str, float]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Move batch to device
                    if len(batch) == 2:
                        inputs, targets = batch
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        attention_mask = None
                    else:
                        inputs, targets, attention_mask = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                    
                    outputs = model(inputs, attention_mask=attention_mask)
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    
                    if outputs.dim() > 1:  # Classification
                        predicted = outputs.argmax(dim=1)
                        total_correct += (predicted == targets).sum().item()
                        total_samples += targets.size(0)
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.warning("CUDA OOM during validation, skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None, 
             save_dir: Path = Path('./models')) -> Dict[str, Any]:
        """Main training loop."""
        
        # Setup
        model = model.to(self.device)
        optimizer = self.setup_optimizer(model)
        
        num_training_steps = self.num_epochs * len(train_loader)
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        if hasattr(self.config, 'class_weights') and self.config.class_weights:
            weights = torch.tensor(self.config.class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            self.metrics.update(**train_metrics)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(model, val_loader, criterion)
                self.metrics.update(**val_metrics)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                else:
                    scheduler.step()
            
            # End epoch metrics
            self.metrics.end_epoch(epoch)
            
            # Early stopping check
            if self.early_stopping is not None:
                monitor_metric = val_metrics.get('val_loss', train_metrics['train_loss'])
                if self.early_stopping(monitor_metric):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % self.save_every == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        # Save final model
        final_model_path = save_dir / "final_model.pth"
        best_model_path = save_dir / "best_model.pth"
        
        torch.save(model.state_dict(), final_model_path)
        torch.save(model.state_dict(), best_model_path)  # For now, same as final
        
        # Save metrics
        metrics_path = save_dir / "training_metrics.json"
        self.metrics.save(metrics_path)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'final_model_path': final_model_path,
            'best_model_path': best_model_path,
            'metrics': dict(self.metrics.best_metrics),
            'training_time': training_time
        }
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, filepath: Path):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'metrics': dict(self.metrics.metrics),
            'device': str(self.device)
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
