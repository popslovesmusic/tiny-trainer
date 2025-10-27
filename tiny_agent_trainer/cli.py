#!/usr/bin/env python3
"""
Unified command line interface for the Tiny Agent Trainer.
Provides a clean, consistent interface for all operations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml # New import

from tiny_agent_trainer.config import ConfigManager, TaskConfig
from tiny_agent_trainer.core.training import Trainer, GPUManager
from tiny_agent_trainer.core.models import create_model
from tiny_agent_trainer.core.tokenizers import TokenizerFactory
from tiny_agent_trainer.core.datasets import create_classification_dataset, create_vsl_dataset, create_dataloader, split_dataset
from tiny_agent_trainer.core.inference import ModelInference
from tiny_agent_trainer.utils.security import execute_vsl_safely

logger = logging.getLogger(__name__)


# New class to replace the existing ConfigManager logic
class ConfigManager:
    """Manages loading and saving of YAML configuration files."""

    def __init__(self, configs_dir: Path = Path("./configs")):
        self.configs_dir = configs_dir
        self.configs_dir.mkdir(exist_ok=True)

    def _get_config_path(self, config_name: str) -> Path:
        """Helper to get the full path for a config file."""
        return self.configs_dir / f"{config_name.lower().replace(' ', '_')}.yaml"

    def list_configs(self):
        """List all available YAML configuration files."""
        return [f.stem for f in self.configs_dir.glob("*.yaml")]

    def get_config(self, config_name: str) -> TaskConfig:
        """Load a configuration from a YAML file."""
        config_path = self._get_config_path(config_name)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        return TaskConfig(**raw_config)

    def save_config(self, config: TaskConfig):
        """Save a TaskConfig object to a YAML file."""
        config_path = self._get_config_path(config.task_name)
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, sort_keys=False)

    def save_config_from_dict(self, config_dict: Dict[str, Any], filename: str):
        """Save a configuration dictionary to a YAML file."""
        config_path = self.configs_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)
        return str(config_path)

    def validate_config(self, config: TaskConfig) -> list[str]:
        """Validate a loaded configuration for required keys."""
        issues = []
        if not hasattr(config, 'task_name'): issues.append("Missing 'task_name'")
        if not hasattr(config, 'task_type'): issues.append("Missing 'task_type'")
        if not hasattr(config, 'model'): issues.append("Missing 'model' section")
        if not hasattr(config, 'training'): issues.append("Missing 'training' section")
        if not hasattr(config, 'data'): issues.append("Missing 'data' section")
        return issues


class TinyAgentTrainerCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.setup_logging()
    
    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('tiny_agent_trainer.log')
            ]
        )
    
    def list_configs(self, args):
        """List available configurations."""
        print("\n🔧 Available Configurations:")
        print("=" * 50)
        
        configs = self.config_manager.list_configs()
        if not configs:
            print("No configurations found.")
            return
        
        for config_name in configs:
            try:
                config = self.config_manager.get_config(config_name)
                print(f"📋 {config_name}")
                print(f"   Task: {config.task_name}")
                print(f"   Type: {config.task_type}")
                print(f"   Model: {config.model.architecture}")
                if hasattr(config.data, 'corpus') and config.data.corpus:
                    print(f"   Samples: {len(config.data.corpus)}")
                print()
            except Exception as e:
                print(f"❌ {config_name}: Error loading ({e})")
                print()
    
    def show_config(self, args):
        """Show detailed configuration."""
        try:
            config = self.config_manager.get_config(args.config)
            
            print(f"\n🔧 Configuration: {args.config}")
            print("=" * 50)
            print(f"Task Name: {config.task_name}")
            print(f"Task Type: {config.task_type}")
            print()
            
            print("📊 Model Configuration:")
            print(f"  Architecture: {config.model.architecture}")
            print(f"  Embedding Dim: {config.model.embedding_dim}")
            print(f"  Hidden Dim: {config.model.hidden_dim}")
            print(f"  Layers: {config.model.num_layers}")
            print(f"  Dropout: {config.model.dropout}")
            print()
            
            print("🎯 Training Configuration:")
            print(f"  Epochs: {config.training.num_epochs}")
            print(f"  Batch Size: {config.training.batch_size}")
            print(f"  Learning Rate: {config.training.learning_rate}")
            print(f"  Optimizer: {config.training.optimizer}")
            print(f"  Early Stopping: {config.training.early_stopping}")
            print()
            
            print("🔤 Tokenizer Configuration:")
            print(f"  Type: {config.tokenizer.type}")
            print(f"  Max Length: {config.tokenizer.max_length}")
            print(f"  Lowercase: {config.tokenizer.lowercase}")
            print()
            
            if hasattr(config.data, 'corpus') and config.data.corpus:
                print("📚 Data Samples:")
                for i, sample in enumerate(config.data.corpus[:3]):
                    if isinstance(sample, tuple) and len(sample) >= 2:
                        print(f"  {i+1}. \"{sample[0]}\" -> \"{sample[1]}\"")
                if len(config.data.corpus) > 3:
                    print(f"  ... and {len(config.data.corpus) - 3} more samples")
            
        except Exception as e:
            print(f"❌ Error showing config '{args.config}': {e}")
            sys.exit(1)
    
    def create_config(self, args):
        """Create a new configuration interactively."""
        print("\n🔧 Creating New Configuration")
        print("=" * 40)
        
        try:
            # Get basic information
            task_name = input("Task name: ").strip()
            if not task_name:
                print("❌ Task name is required")
                return
            
            print("\nSelect task type:")
            print("1. Classification (sentiment, firewall, etc.)")
            print("2. VSL Translation")
            print("3. Custom")
            
            task_type_choice = input("Choice (1-3): ").strip()
            task_type_map = {
                "1": "classification",
                "2": "vsl_translation", 
                "3": "classification"  # Default
            }
            task_type = task_type_map.get(task_type_choice, "classification")
            
            # Create config template
            # config = self.config_manager.create_config_template(task_name, task_type)
            
            # Get corpus data
            print(f"\nEnter training data for {task_type}:")
            if task_type == "classification":
                print("Format: input_text,label")
            else:
                print("Format: natural_language,vsl_code")
            
            print("Enter 'done' when finished, 'cancel' to abort:")
            
            corpus = []
            while True:
                sample = input(f"Sample {len(corpus) + 1}: ").strip()
                if sample.lower() == 'done':
                    break
                elif sample.lower() == 'cancel':
                    print("❌ Configuration creation cancelled")
                    return
                elif ',' in sample:
                    parts = sample.split(',', 1)
                    corpus.append([parts[0].strip(), parts[1].strip()])
                else:
                    print("❌ Invalid format. Use: input,output")
            
            if not corpus:
                print("❌ At least one training sample is required")
                return
            
            # Build the Python dictionary
            new_config = {
                'task_name': task_name,
                'task_type': task_type,
                'model': {
                    'architecture': 'transformer',
                    'd_model': 64,
                    'nhead': 2,
                    'num_layers': 1,
                    'embedding_dim': 32,
                    'hidden_dim': 64,
                    'dropout': 0.1,
                    'max_seq_len': 100
                },
                'training': {
                    'num_epochs': 200,
                    'batch_size': 4,
                    'learning_rate': 0.001,
                    'optimizer': 'adamw',
                    'early_stopping': True
                },
                'tokenizer': {
                    'type': 'word',
                    'max_length': 512,
                    'lowercase': True
                },
                'data': {
                    'corpus': corpus
                },
                'security': {
                    'enable_vsl_execution': True,
                    'vsl_timeout': 10,
                    'validate_inputs': True
                }
            }
            
            # Save configuration
            filename = f"{task_name.lower().replace(' ', '_')}.yaml"
            file_path = self.config_manager.save_config_from_dict(new_config, filename)
            
            print(f"\n✅ Configuration created and saved to: {file_path}")
            print(f"📝 You can now train with: python cli.py train --config {task_name.lower().replace(' ', '_')}")
            
        except KeyboardInterrupt:
            print("\n❌ Configuration creation cancelled")
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
    
    def check_system(self, args):
        """Check system capabilities and requirements."""
        print("\n🔍 System Check")
        print("=" * 30)
        
        # GPU Information
        device_info = GPUManager.get_device_info()
        
        print("🖥️  Hardware:")
        print(f"   CUDA Available: {'✅' if device_info['cuda_available'] else '❌'}")
        print(f"   GPU Count: {device_info['gpu_count']}")
        
        for gpu in device_info['gpus']:
            print(f"   GPU {gpu['id']}: {gpu['name']}")
            print(f"     Memory: {gpu['memory_total']/1e9:.1f}GB total, {gpu['memory_free']/1e9:.1f}GB free")
            print(f"     Compute Capability: {gpu['compute_capability']}")
        
        if not device_info['cuda_available']:
            print("   ⚠️  No GPU detected. Training will use CPU (slower)")
        
        print()
        
        # Python Environment
        print("🐍 Python Environment:")
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Version: {torch.version.cuda or 'N/A'}")
        
        # Check dependencies
        print("\n📦 Dependencies:")
        dependencies = ['numpy', 'torch', 'yaml', 'json']
        for dep in dependencies:
            try:
                __import__(dep)
                print(f"   {dep}: ✅")
            except ImportError:
                print(f"   {dep}: ❌")
        
        # VSL Dependencies (optional)
        print("\n🔧 VSL Support:")
        import shutil
        julia_available = shutil.which('julia') is not None
        print(f"   Julia Runtime: {'✅' if julia_available else '❌'}")
        if not julia_available:
            print("   ⚠️  Julia not found. VSL execution will be disabled.")
            print("   💡 Install Julia from: https://julialang.org/downloads/")
        
        print("\n✅ System check complete!")
    
    def train(self, args):
        """Train a model."""
        try:
            print(f"\n🚀 Training Model: {args.config}")
            print("=" * 50)
            
            # Load configuration
            config = self.config_manager.get_config(args.config)
            
            # Validate configuration
            issues = self.config_manager.validate_config(config)
            if issues:
                print("❌ Configuration validation failed:")
                for issue in issues:
                    print(f"   - {issue}")
                sys.exit(1)
            
            # Override config with command line arguments
            if args.epochs:
                config.training.num_epochs = args.epochs
            if args.batch_size:
                config.training.batch_size = args.batch_size
            if args.learning_rate:
                config.training.learning_rate = args.learning_rate
            
            # Show system info
            device_info = GPUManager.get_device_info()
            if device_info['cuda_available']:
                print(f"🖥️  Using GPU: {device_info['gpus'][0]['name']}")
            else:
                print("🖥️  Using CPU (consider using GPU for faster training)")
            
            # Create tokenizer
            print("🔤 Creating tokenizer...")
            if config.task_type == "vsl_translation":
                # VSL requires two tokenizers
                nl_texts = [item[0] for item in config.data.corpus]
                vsl_codes = [item[1] for item in config.data.corpus]
                
                nl_tokenizer = TokenizerFactory.create_tokenizer('word')
                nl_tokenizer.fit(nl_texts)
                
                vsl_tokenizer = TokenizerFactory.create_tokenizer('vsl') 
                vsl_tokenizer.fit(vsl_codes)
                
                tokenizer = (nl_tokenizer, vsl_tokenizer)
            else:
                # Classification tasks
                texts = [item[0] for item in config.data.corpus]
                tokenizer = TokenizerFactory.create_for_task(
                    config.task_type,
                    lowercase=config.tokenizer.lowercase,
                    min_freq=config.tokenizer.min_freq
                )
                tokenizer.fit(texts)
            
            # Create dataset
            print("📚 Creating dataset...")
            if config.task_type == "vsl_translation":
                dataset = create_vsl_dataset(
                    config.data.corpus,
                    tokenizer[0],  # NL tokenizer
                    tokenizer[1],  # VSL tokenizer
                    max_length=config.tokenizer.max_length
                )
            else:
                dataset = create_classification_dataset(
                    config.data.corpus,
                    tokenizer,
                    max_length=config.tokenizer.max_length
                )
            
            # Split dataset
            if len(dataset) > 10:  # Only split if we have enough data
                train_dataset, val_dataset, test_dataset = split_dataset(
                    dataset,
                    train_ratio=0.7,
                    val_ratio=0.15
                )
                print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
            else:
                train_dataset = dataset
                val_dataset = None
                test_dataset = None
                print(f"📊 Small dataset ({len(dataset)} samples), using all for training")
            
            # Create data loaders
            batch_size = min(config.training.batch_size, len(train_dataset))
            
            train_loader = create_dataloader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
            
            val_loader = None
            if val_dataset and len(val_dataset) > 0:
                val_loader = create_dataloader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )
            
            # Create model
            print("🤖 Creating model...")
            if config.task_type == "vsl_translation":
                model_config = {
                    'architecture': config.model.architecture,
                    'nl_vocab_size': tokenizer[0].vocab_size,
                    'vsl_vocab_size': tokenizer[1].vocab_size,
                    'd_model': config.model.d_model,
                    'nhead': config.model.nhead,
                    'num_layers': config.model.num_layers,
                    'max_seq_len': config.model.max_seq_len
                }
            else:
                model_config = {
                    'architecture': config.model.architecture,
                    'vocab_size': tokenizer.vocab_size,
                    'output_dim': dataset.get_num_classes(),
                    'embedding_dim': config.model.embedding_dim,
                    'hidden_dim': config.model.hidden_dim,
                    'd_model': config.model.d_model,
                    'nhead': config.model.nhead,
                    'num_layers': config.model.num_layers,
                    'dropout': config.model.dropout,
                    'max_seq_len': config.model.max_seq_len
                }
            
            model = create_model(model_config)
            
            print(f"📊 Model: {config.model.architecture} with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Create trainer
            trainer = Trainer(config.training.__dict__)
            
            # Create save directory
            save_dir = Path(f"./models/{config.task_name}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            print("\n🎯 Starting training...")
            results = trainer.train(model, train_loader, val_loader, save_dir)
            
            # Save tokenizer
            if config.task_type == "vsl_translation":
                tokenizer[0].save(save_dir / "nl_tokenizer.json")
                tokenizer[1].save(save_dir / "vsl_tokenizer.json")
            else:
                tokenizer.save(save_dir / "tokenizer.json")
            
            # Save configuration
            self.config_manager.save_config(config, save_dir / "config.yaml")
            
            print(f"\n✅ Training completed!")
            print(f"📁 Model saved to: {save_dir}")
            print(f"📊 Best metrics: {results['metrics']}")
            print(f"⏱️  Training time: {results['training_time']:.2f}s")
            
            # Test the model if we have test data
            if test_dataset and len(test_dataset) > 0:
                print("\n🧪 Testing model...")
                self._test_model(model, test_dataset, tokenizer, config)
                
        except KeyboardInterrupt:
            print("\n❌ Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            print(f"❌ Training failed: {e}")
            sys.exit(1)
    
    def test(self, args):
        """Test/run a trained model interactively."""
        try:
            print(f"\n🧪 Testing Model: {args.config}")
            print("=" * 40)
            
            # Load model and configuration
            model_dir = Path(f"./models/{args.config}")
            if not model_dir.exists():
                # Try loading by config name
                config = self.config_manager.get_config(args.config)
                model_dir = Path(f"./models/{config.task_name}")
            
            if not model_dir.exists():
                print(f"❌ Model not found. Train it first with: python cli.py train --config {args.config}")
                sys.exit(1)
            
            # Load inference system
            inference = ModelInference(model_dir)
            
            print(f"✅ Model loaded: {inference.config.task_name}")
            print(f"🎯 Task type: {inference.config.task_type}")
            
            # Interactive testing
            if inference.config.task_type == "vsl_translation":
                self._interactive_vsl_testing(inference)
            else:
                self._interactive_classification_testing(inference)
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            sys.exit(1)
    
    def _interactive_classification_testing(self, inference: 'ModelInference'):
        """Interactive testing for classification tasks."""
        print(f"\n🎯 Interactive {inference.config.task_name} Testing")
        print("=" * 50)
        print("Enter text to classify (type 'quit' to exit):")
        print()
        
        while True:
            try:
                user_input = input("📝 Input: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Get prediction
                result = inference.predict(user_input)
                
                print(f"🎯 Prediction: {result['predicted_label']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                
                if result['probabilities']:
                    print("📈 All probabilities:")
                    for label, prob in result['probabilities'].items():
                        print(f"   {label}: {prob:.3f}")
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
        print("👋 Testing session ended")
    
    def _interactive_vsl_testing(self, inference: 'ModelInference'):
        """Interactive testing for VSL translation."""
        print(f"\n🎯 Interactive VSL Translation Testing")
        print("=" * 50)
        print("Enter natural language to translate to VSL (type 'quit' to exit):")
        print()
        
        while True:
            try:
                user_input = input("📝 Natural Language: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Get VSL translation
                result = inference.predict(user_input)
                
                print(f"🔧 Generated VSL: {result['predicted_vsl']}")
                
                # Execute VSL if enabled
                if inference.config.security.enable_vsl_execution:
                    print("🚀 Executing VSL...")
                    execution_result = execute_vsl_safely(
                        result['predicted_vsl'],
                        timeout=inference.config.security.vsl_timeout
                    )
                    
                    if execution_result['success']:
                        print("✅ Execution successful:")
                        print(execution_result['output'])
                    else:
                        print("❌ Execution failed:")
                        print(execution_result['error'])
                        if execution_result['stderr']:
                            print("Stderr:", execution_result['stderr'])
                else:
                    print("⚠️  VSL execution disabled in configuration")
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
        print("👋 Testing session ended")
    
    def _test_model(self, model, test_dataset, tokenizer, config):
        """Run model evaluation on test set."""
        test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:  # Classification
                    inputs, targets, attention_mask = batch
                    outputs = model(inputs, attention_mask=attention_mask)
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
        
        accuracy = correct / total if total > 0 else 0
        print(f"🎯 Test Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    def add_example(self, args):
        """Add a new training example to a configuration."""
        try:
            config = self.config_manager.get_config(args.config)
            
            print(f"\n➕ Adding example to: {config.task_name}")
            print("=" * 40)
            
            if config.task_type == "vsl_translation":
                nl_text = input("Natural language: ").strip()
                vsl_code = input("VSL code: ").strip()
                
                if nl_text and vsl_code:
                    config.data.corpus.append([nl_text, vsl_code])
            else:
                text = input("Input text: ").strip()
                label = input("Label: ").strip()
                
                if text and label:
                    config.data.corpus.append([text, label])
            
            # Save updated configuration
            self.config_manager.save_config(config)
            print("✅ Example added successfully!")
            
        except Exception as e:
            print(f"❌ Error adding example: {e}")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Tiny Agent Trainer - A lightweight framework for domain-specific AI agents",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python cli.py list                           # List available configurations
  python cli.py show --config sentiment       # Show configuration details
  python cli.py train --config sentiment      # Train a model
  python cli.py test --config sentiment       # Test a trained model
  python cli.py create                         # Create new configuration interactively
  python cli.py check                          # Check system capabilities
            """
        )
        
        # Global options
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                          default="INFO", help="Set logging level")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # List command
        list_parser = subparsers.add_parser("list", help="List available configurations")
        list_parser.set_defaults(func=self.list_configs)
        
        # Show command
        show_parser = subparsers.add_parser("show", help="Show configuration details")
        show_parser.add_argument("--config", "-c", required=True, help="Configuration name")
        show_parser.set_defaults(func=self.show_config)
        
        # Create command
        create_parser = subparsers.add_parser("create", help="Create new configuration")
        create_parser.set_defaults(func=self.create_config)
        
        # Check command
        check_parser = subparsers.add_parser("check", help="Check system capabilities")
        check_parser.set_defaults(func=self.check_system)
        
        # Train command
        train_parser = subparsers.add_parser("train", help="Train a model")
        train_parser.add_argument("--config", "-c", required=True, help="Configuration name")
        train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs (override config)")
        train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size (override config)")
        train_parser.add_argument("--learning-rate", "-lr", type=float, help="Learning rate (override config)")
        train_parser.set_defaults(func=self.train)
        
        # Test command
        test_parser = subparsers.add_parser("test", help="Test/run a trained model")
        test_parser.add_argument("--config", "-c", required=True, help="Configuration name")
        test_parser.set_defaults(func=self.test)
        
        # Add example command
        add_parser = subparsers.add_parser("add", help="Add training example to configuration")
        add_parser.add_argument("--config", "-c", required=True, help="Configuration name")
        add_parser.set_defaults(func=self.add_example)
        
        return parser
    
    def run(self, args=None):
        """Run the CLI application."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging level
        if hasattr(parsed_args, 'log_level'):
            self.setup_logging(parsed_args.log_level)
        
        if not parsed_args.command:
            parser.print_help()
            return
        
        # Run the specified command
        try:
            parsed_args.func(parsed_args)
        except AttributeError:
            parser.print_help()


def main():
    """Main entry point."""
    cli = TinyAgentTrainerCLI()
    cli.run()


if __name__ == "__main__":
    main()
