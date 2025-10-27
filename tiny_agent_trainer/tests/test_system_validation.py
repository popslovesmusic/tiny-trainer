#!/usr/bin/env python3
"""
System validation tests for Tiny Agent Trainer.

This script validates the complete system functionality and can be run
to ensure everything is working correctly after installation.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import torch
import json
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_agent_trainer.config import ConfigManager, TaskConfig
from tiny_agent_trainer.core.models import create_model
from tiny_agent_trainer.core.tokenizers import TokenizerFactory
from tiny_agent_trainer.core.datasets import create_classification_dataset, create_dataloader
from tiny_agent_trainer.core.training import Trainer
from tiny_agent_trainer.core.inference import ModelInference
from tiny_agent_trainer.utils.security import execute_vsl_safely, validate_vsl_safely
from tiny_agent_trainer.utils.gpu_utils import get_device_info, select_best_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Validates system components and functionality."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
        self.device = select_best_device()
    
    def setup_temp_environment(self):
        """Setup temporary testing environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tat_test_"))
        logger.info(f"Created temp directory: {self.temp_dir}")
    
    def cleanup_temp_environment(self):
        """Clean up temporary testing environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp directory")
    
    def test_imports(self):
        """Test all critical imports."""
        logger.info("Testing imports...")
        
        try:
            # Core framework imports
            from tiny_agent_trainer import TinyAgentTrainer, ConfigManager, ModelInference
            from tiny_agent_trainer.core import models, training, tokenizers, datasets, inference
            from tiny_agent_trainer.utils import security, gpu_utils
            from tiny_agent_trainer import config
            
            # External dependencies
            import torch
            import yaml
            import numpy as np
            
            self.results['imports'] = {'status': 'PASS', 'details': 'All imports successful'}
            logger.info("✅ Imports test passed")
            return True
            
        except ImportError as e:
            self.results['imports'] = {'status': 'FAIL', 'details': f'Import error: {e}'}
            logger.error(f"❌ Imports test failed: {e}")
            return False
    
    def test_gpu_functionality(self):
        """Test GPU detection and functionality."""
        logger.info("Testing GPU functionality...")
        
        try:
            device_info = get_device_info()
            
            details = {
                'cuda_available': device_info['cuda_available'],
                'gpu_count': device_info['gpu_count'],
                'selected_device': str(self.device)
            }
            
            if device_info['cuda_available']:
                # Test basic GPU operations
                test_tensor = torch.randn(100, 100).to(self.device)
                result = torch.mm(test_tensor, test_tensor)
                
                details['gpu_operations'] = 'working'
                logger.info(f"✅ GPU test passed - {device_info['gpu_count']} GPU(s) available")
            else:
                details['gpu_operations'] = 'cpu_only'
                logger.info("⚠️ GPU not available, using CPU")
            
            self.results['gpu'] = {'status': 'PASS', 'details': details}
            return True
            
        except Exception as e:
            self.results['gpu'] = {'status': 'FAIL', 'details': f'GPU test error: {e}'}
            logger.error(f"❌ GPU test failed: {e}")
            return False
    
    def test_tokenization(self):
        """Test tokenization functionality."""
        logger.info("Testing tokenization...")
        
        try:
            # Test different tokenizer types
            test_texts = [
                "Hello world, this is a test!",
                "Another example with numbers 123",
                "Special characters: @#$%"
            ]
            
            tokenizer_types = ['word', 'vsl', 'smart']
            results_details = {}
            
            for tok_type in tokenizer_types:
                tokenizer = TokenizerFactory.create_tokenizer(tok_type)
                tokenizer.fit(test_texts)
                
                # Test tokenization
                for text in test_texts:
                    tokens = tokenizer.tokenize(text)
                    decoded = tokenizer.decode(tokens)
                    
                    if not tokens:
                        raise ValueError(f"No tokens generated for: {text}")
                
                results_details[tok_type] = {
                    'vocab_size': tokenizer.vocab_size,
                    'test_passed': True
                }
            
            self.results['tokenization'] = {'status': 'PASS', 'details': results_details}
            logger.info("✅ Tokenization test passed")
            return True
            
        except Exception as e:
            self.results['tokenization'] = {'status': 'FAIL', 'details': f'Tokenization error: {e}'}
            logger.error(f"❌ Tokenization test failed: {e}")
            return False
    
    def test_model_creation(self):
        """Test model creation and basic operations."""
        logger.info("Testing model creation...")
        
        try:
            # Test different model architectures
            architectures = ['rnn', 'transformer']
            results_details = {}
            
            for arch in architectures:
                model_config = {
                    'architecture': arch,
                    'vocab_size': 1000,
                    'output_dim': 3,
                    'embedding_dim': 64,
                    'hidden_dim': 128,
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2
                }
                
                model = create_model(model_config)
                model = model.to(self.device)
                
                # Test forward pass
                batch_size = 4
                seq_len = 10
                test_input = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                attention_mask = torch.ones(batch_size, seq_len).to(self.device)
                
                with torch.no_grad():
                    output = model(test_input, attention_mask=attention_mask)
                
                expected_shape = (batch_size, 3)
                if output.shape != expected_shape:
                    raise ValueError(f"Wrong output shape: {output.shape}, expected {expected_shape}")
                
                param_count = sum(p.numel() for p in model.parameters())
                results_details[arch] = {
                    'parameters': param_count,
                    'output_shape': list(output.shape),
                    'test_passed': True
                }
            
            self.results['models'] = {'status': 'PASS', 'details': results_details}
            logger.info("✅ Model creation test passed")
            return True
            
        except Exception as e:
            self.results['models'] = {'status': 'FAIL', 'details': f'Model creation error: {e}'}
            logger.error(f"❌ Model creation test failed: {e}")
            return False
    
    def test_training_pipeline(self):
        """Test end-to-end training pipeline."""
        logger.info("Testing training pipeline...")
        
        try:
            # Create minimal dataset
            corpus = [
                ("positive example one", "positive"),
                ("positive example two", "positive"),
                ("negative example one", "negative"),
                ("negative example two", "negative"),
                ("positive example three", "positive"),
                ("negative example three", "negative")
            ]
            
            # Create tokenizer
            texts = [item[0] for item in corpus]
            tokenizer = TokenizerFactory.create_tokenizer('word')
            tokenizer.fit(texts)
            
            # Create dataset
            dataset = create_classification_dataset(corpus, tokenizer, max_length=32)
            dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
            
            # Create model
            model_config = {
                'architecture': 'rnn',
                'vocab_size': tokenizer.vocab_size,
                'output_dim': 2,
                'embedding_dim': 32,
                'hidden_dim': 64,
                'num_layers': 1
            }
            
            model = create_model(model_config)
            
            # Create trainer with minimal settings
            training_config = {
                'num_epochs': 5,
                'learning_rate': 0.01,
                'gradient_clip_norm': 1.0,
                'early_stopping': False
            }
            
            trainer = Trainer(training_config)
            
            # Train
            save_dir = self.temp_dir / "test_model"
            results = trainer.train(model, dataloader, save_dir=save_dir)
            
            # Verify results
            if 'final_model_path' not in results:
                raise ValueError("Training did not produce model file")
            
            if not Path(results['final_model_path']).exists():
                raise ValueError("Model file not saved")
            
            training_time = results.get('training_time', 0)
            
            self.results['training'] = {
                'status': 'PASS', 
                'details': {
                    'training_time': training_time,
                    'model_saved': True,
                    'epochs_completed': 5
                }
            }
            logger.info("✅ Training pipeline test passed")
            return True
            
        except Exception as e:
            self.results['training'] = {'status': 'FAIL', 'details': f'Training error: {e}'}
            logger.error(f"❌ Training pipeline test failed: {e}")
            return False
    
    def test_inference_pipeline(self):
        """Test model inference functionality."""
        logger.info("Testing inference pipeline...")
        
        try:
            # Use model from training test
            model_dir = self.temp_dir / "test_model"
            
            if not model_dir.exists():
                # Create a minimal model for testing
                logger.info("Creating minimal model for inference test...")
                self.test_training_pipeline()
            
            # Save tokenizer for inference
            texts = ["positive example", "negative example"]
            tokenizer = TokenizerFactory.create_tokenizer('word')
            tokenizer.fit(texts)
            tokenizer.save(model_dir / "tokenizer.json")
            
            # Save config
            from tiny_agent_trainer.config import TaskConfig, ModelConfig
            config = TaskConfig(
                task_name="TestModel",
                task_type="classification",
                model=ModelConfig(architecture="rnn")
            )
            config.data.corpus = [("test", "positive"), ("test2", "negative")]
            
            config_manager = ConfigManager()
            config_manager.save_config(config, model_dir / "config.yaml")
            
            # Test inference
            inference = ModelInference(model_dir)
            
            test_texts = ["positive test", "negative test"]
            results_details = []
            
            for text in test_texts:
                result = inference.predict(text)
                
                required_keys = ['predicted_label', 'confidence', 'probabilities']
                for key in required_keys:
                    if key not in result:
                        raise ValueError(f"Missing key in prediction result: {key}")
                
                results_details.append({
                    'input': text,
                    'prediction': result['predicted_label'],
                    'confidence': result['confidence']
                })
            
            self.results['inference'] = {
                'status': 'PASS',
                'details': {
                    'predictions_made': len(results_details),
                    'sample_results': results_details
                }
            }
            logger.info("✅ Inference pipeline test passed")
            return True
            
        except Exception as e:
            self.results['inference'] = {'status': 'FAIL', 'details': f'Inference error: {e}'}
            logger.error(f"❌ Inference pipeline test failed: {e}")
            return False
    
    def test_security_features(self):
        """Test security and safety features."""
        logger.info("Testing security features...")
        
        try:
            # Test VSL validation
            safe_vsl_codes = [
                "▲(A1)",
                "∫(B2)",
                "A1 + B2",
                "~(100)"
            ]
            
            dangerous_vsl_codes = [
                "system('rm -rf /')",
                "exec('malicious_code')",
                "import os; os.system('bad')"
            ]
            
            security_results = {
                'safe_codes_passed': 0,
                'dangerous_codes_blocked': 0
            }
            
            # Test safe codes
            for code in safe_vsl_codes:
                if validate_vsl_safely(code):
                    security_results['safe_codes_passed'] += 1
            
            # Test dangerous codes
            for code in dangerous_vsl_codes:
                if not validate_vsl_safely(code):
                    security_results['dangerous_codes_blocked'] += 1
            
            # Test execution with timeout
            test_result = execute_vsl_safely("A1", timeout=1)
            security_results['execution_test'] = 'completed' if test_result else 'failed'
            
            all_passed = (
                security_results['safe_codes_passed'] == len(safe_vsl_codes) and
                security_results['dangerous_codes_blocked'] == len(dangerous_vsl_codes)
            )
            
            status = 'PASS' if all_passed else 'PARTIAL'
            
            self.results['security'] = {'status': status, 'details': security_results}
            logger.info(f"✅ Security features test: {status}")
            return True
            
        except Exception as e:
            self.results['security'] = {'status': 'FAIL', 'details': f'Security test error: {e}'}
            logger.error(f"❌ Security features test failed: {e}")
            return False
    
    def test_configuration_system(self):
        """Test configuration management."""
        logger.info("Testing configuration system...")
        
        try:
            config_manager = ConfigManager(self.temp_dir / "configs")
            
            # Test built-in configurations
            built_in_configs = config_manager.list_configs()
            if not built_in_configs:
                raise ValueError("No built-in configurations found")
            
            # Test loading built-in config
            test_config = config_manager.get_config("sentiment_analysis")
            
            # Test creating custom config
            custom_config = config_manager.create_config_template("TestTask", "classification")
            custom_config.data.corpus = [("test input", "test_output")]
            
            # Test saving and loading custom config
            saved_path = config_manager.save_config(custom_config, "test_config.yaml")
            loaded_config = config_manager.load_config(saved_path)
            
            # Test validation
            issues = config_manager.validate_config(loaded_config)
            
            config_results = {
                'built_in_configs': len(built_in_configs),
                'save_load_successful': loaded_config.task_name == custom_config.task_name,
                'validation_issues': len(issues)
            }
            
            self.results['configuration'] = {'status': 'PASS', 'details': config_results}
            logger.info("✅ Configuration system test passed")
            return True
            
        except Exception as e:
            self.results['configuration'] = {'status': 'FAIL', 'details': f'Configuration error: {e}'}
            logger.error(f"❌ Configuration system test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("Starting system validation tests...")
        
        self.setup_temp_environment()
        
        tests = [
            self.test_imports,
            self.test_gpu_functionality,
            self.test_tokenization,
            self.test_model_creation,
            self.test_training_pipeline,
            self.test_inference_pipeline,
            self.test_security_features,
            self.test_configuration_system
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                failed += 1
        
        self.cleanup_temp_environment()
        
        # Generate summary
        summary = {
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(tests),
            'timestamp': time.time(),
            'system_info': {
                'python_version': sys.version,
                'pytorch_version': torch.__version__,
                'device': str(self.device)
            }
        }
        
        self.results['summary'] = summary
        
        return failed == 0
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("SYSTEM VALIDATION RESULTS")
        print("=" * 60)
        
        summary = self.results.get('summary', {})
        
        print(f"Tests Run: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test_name, result in self.results.items():
            if test_name == 'summary':
                continue
            
            status = result.get('status', 'UNKNOWN')
            emoji = "✅" if status == 'PASS' else "⚠️" if status == 'PARTIAL' else "❌"
            
            print(f"{emoji} {test_name.title()}: {status}")
            
            if 'details' in result and isinstance(result['details'], dict):
                for key, value in result['details'].items():
                    if isinstance(value, (int, float, str, bool)):
                        print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        
        if summary.get('failed', 1) == 0:
            print("🎉 ALL TESTS PASSED - System is ready!")
            print("\nNext steps:")
            print("• Try: python cli.py list")
            print("• Train a model: python cli.py train --config sentiment_analysis")
            print("• Run examples: python examples/basic_usage.py")
        else:
            print("⚠️  Some tests failed - check the details above")
            print("\nTroubleshooting:")
            print("• Check requirements: pip install -r requirements.txt")
            print("• System check: python cli.py check")
            print("• Enable debug: export LOG_LEVEL=DEBUG")
    
    def save_results(self, filepath=None):
        """Save results to file."""
        if filepath is None:
            filepath = self.temp_dir.parent / "validation_results.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")


def main():
    """Main validation function."""
    print("Tiny Agent Trainer - System Validation")
    print("This will test all major components and functionality")
    print()
    
    validator = SystemValidator()
    
    try:
        success = validator.run_all_tests()
        validator.print_results()
        
        # Save results
        results_file = Path("validation_results.json")
        validator.save_results(results_file)
        print(f"\nFull results saved to: {results_file}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nValidation crashed: {e}")
        logger.exception("Validation error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
