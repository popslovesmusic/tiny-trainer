"""
Tiny Agent Trainer - A production-ready framework for domain-specific AI agents.

This package provides tools for training, deploying, and managing small specialized
neural networks for focused tasks like sentiment analysis, code generation, and
signal processing.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Fix for local imports
# This ensures that modules within the same package can be found
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Version information
__version__ = "2.0.0"
__author__ = "Tiny Agent Trainer Team"
__license__ = "MIT"

# --- HELPER FUNCTIONS AND CLASSES ---

def setup_logging(level=logging.INFO):
    """Setup package-wide logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def check_environment():
    """Check if the environment is properly configured."""
    issues = []
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")

    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - will use CPU (slower training)")
    except ImportError:
        issues.append("PyTorch not installed")

    return issues

class _GlobalConfig:
    """Internal class to manage package-wide configuration and paths."""
    def __init__(self):
        self.package_dir = Path(__file__).parent
        self.project_root = self.package_dir.parent
        self.models_dir = self.project_root / "models"
        self.configs_dir = self.project_root / "configs"
        self.logs_dir = self.project_root / "logs"

        # This is the correct place to create directories on startup
        self.models_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    @property
    def julia_available(self):
        """Check if Julia is available for VSL features."""
        import shutil
        return shutil.which('julia') is not None

# --- MAIN PUBLIC CLASSES AND FUNCTIONS ---

# Create a single, global instance of the config for the package to use
config = _GlobalConfig()

class TinyAgentTrainer:
    """Main trainer class - simplified interface for common use cases."""
    def __init__(self, task_name: str = None, task_type: str = "classification"):
        from .config import ConfigManager

        self.config_manager = ConfigManager(config.configs_dir)

        if task_name:
            try:
                self.config = self.config_manager.get_config(task_name)
            except ValueError:
                self.config = self.config_manager.create_config_template(task_name, task_type)
        else:
            self.config = None

    def train(self, save_config: bool = True):
        """Train the model with current configuration."""
        if not self.config:
            raise ValueError("No configuration loaded.")

        if save_config:
            self.config_manager.save_config(self.config)

        from .core.training import Trainer
        from .core.models import create_model
        from .core.tokenizers import TokenizerFactory
        from .core.datasets import create_classification_dataset, create_dataloader, split_dataset

        texts = [item[0] for item in self.config.data.corpus]
        tokenizer = TokenizerFactory.create_for_task(self.config.task_type)
        tokenizer.fit(texts)

        dataset = create_classification_dataset(self.config.data.corpus, tokenizer)

        if len(dataset) > 10:
            train_dataset, val_dataset, _ = split_dataset(dataset)
        else:
            train_dataset, val_dataset = dataset, None

        train_loader = create_dataloader(train_dataset, batch_size=self.config.training.batch_size)
        val_loader = create_dataloader(val_dataset, batch_size=self.config.training.batch_size) if val_dataset else None

        model_config = self.config.model.__dict__
        model_config['vocab_size'] = tokenizer.vocab_size
        model_config['output_dim'] = dataset.get_num_classes()

        model = create_model(model_config)

        trainer = Trainer(self.config.training.__dict__)
        save_dir = config.models_dir / self.config.task_name
        results = trainer.train(model, train_loader, val_loader, save_dir)

        tokenizer.save(save_dir / "tokenizer.json")

        return results

    def load_model(self, model_dir: str = None):
        """Load trained model for inference."""
        from .core.inference import ModelInference

        if model_dir is None:
            if not self.config:
                raise ValueError("No model directory specified and no config loaded")
            model_dir = config.models_dir / self.config.task_name

        return ModelInference(model_dir)

def print_info():
    """Print package information."""
    print(f"Tiny Agent Trainer v{__version__}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA {torch.version.cuda} ({torch.cuda.device_count()} GPUs)")
        else:
            print("CUDA: Not available")
    except ImportError:
        print("PyTorch: Not installed")

    print(f"Julia: {'Available' if config.julia_available else 'Not found'}")

# --- PACKAGE INITIALIZATION ---

# Convenience imports for direct access
try:
    from .config import ConfigManager
    from .core.inference import ModelInference
except ImportError as e:
    warnings.warn(f"Some modules not available during package setup: {e}")

# Run environment check on import
for issue in check_environment():
    warnings.warn(f"Environment issue: {issue}")

# Initialize logging
setup_logging()

# Define what gets imported with 'from tiny_agent_trainer import *'
__all__ = [
    'TinyAgentTrainer',
    'ConfigManager', 
    'ModelInference',
    'config',
    'print_info',
    'check_environment',
    '__version__'
]