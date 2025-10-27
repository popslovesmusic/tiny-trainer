"""
Model inference system for trained models.
Handles loading, prediction, and result interpretation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import logging
import json

from tiny_agent_trainer.core.models import create_model
from tiny_agent_trainer.core.tokenizers import BaseTokenizer
from tiny_agent_trainer.config import ConfigManager

logger = logging.getLogger(__name__)


class ModelInference:
    """Main inference class for trained models."""
    
    def __init__(self, model_dir: Union[str, Path], device: Optional[torch.device] = None):
        self.model_dir = Path(model_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load components
        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        logger.info(f"Model loaded: {self.config.task_name} on {self.device}")
    
    def _load_config(self):
        """Load configuration from model directory."""
        config_file = self.model_dir / "config.yaml"
        if not config_file.exists():
            config_file = self.model_dir / "config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration not found in {self.model_dir}")
        
        config_manager = ConfigManager()
        return config_manager.load_config(config_file)
    
    def _load_tokenizer(self):
        """Load tokenizer(s) from model directory."""
        if self.config.task_type == "vsl_translation":
            # Load both tokenizers for VSL tasks
            nl_tokenizer_file = self.model_dir / "nl_tokenizer.json"
            vsl_tokenizer_file = self.model_dir / "vsl_tokenizer.json"
            
            if not nl_tokenizer_file.exists() or not vsl_tokenizer_file.exists():
                raise FileNotFoundError("VSL tokenizer files not found")
            
            nl_tokenizer = BaseTokenizer.load(nl_tokenizer_file)
            vsl_tokenizer = BaseTokenizer.load(vsl_tokenizer_file)
            
            return (nl_tokenizer, vsl_tokenizer)
        else:
            # Single tokenizer for classification tasks
            tokenizer_file = self.model_dir / "tokenizer.json"
            
            if not tokenizer_file.exists():
                raise FileNotFoundError("Tokenizer file not found")
            
            return BaseTokenizer.load(tokenizer_file)
    
    def _load_model(self):
        """Load trained model."""
        model_file = self.model_dir / "best_model.pth"
        if not model_file.exists():
            model_file = self.model_dir / "final_model.pth"
        
        if not model_file.exists():
            raise FileNotFoundError("Model file not found")
        
        # Create model config
        if self.config.task_type == "vsl_translation":
            model_config = {
                'architecture': self.config.model.architecture,
                'nl_vocab_size': self.tokenizer[0].vocab_size,
                'vsl_vocab_size': self.tokenizer[1].vocab_size,
                'd_model': self.config.model.d_model,
                'nhead': self.config.model.nhead,
                'num_layers': self.config.model.num_layers,
                'max_seq_len': self.config.model.max_seq_len
            }
        else:
            # Load class labels from saved metadata if available
            num_classes = self._get_num_classes()
            
            model_config = {
                'architecture': self.config.model.architecture,
                'vocab_size': self.tokenizer.vocab_size,
                'output_dim': num_classes,
                'embedding_dim': self.config.model.embedding_dim,
                'hidden_dim': self.config.model.hidden_dim,
                'd_model': self.config.model.d_model,
                'nhead': self.config.model.nhead,
                'num_layers': self.config.model.num_layers,
                'dropout': self.config.model.dropout,
                'max_seq_len': self.config.model.max_seq_len
            }
        
        # Create and load model
        model = create_model(model_config)
        
        try:
            state_dict = torch.load(model_file, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_num_classes(self) -> int:
        """Get number of classes for classification tasks."""
        if hasattr(self.config.data, 'corpus') and self.config.data.corpus:
            labels = [item[1] for item in self.config.data.corpus]
            return len(set(labels))
        else:
            # Try to infer from model if saved
            return 2  # Default binary classification
    
    def _get_class_labels(self) -> List[str]:
        """Get class labels for classification tasks."""
        if hasattr(self.config.data, 'corpus') and self.config.data.corpus:
            labels = [item[1] for item in self.config.data.corpus]
            return sorted(list(set(labels)))
        else:
            # Default labels
            num_classes = self._get_num_classes()
            return [f"class_{i}" for i in range(num_classes)]
    
    def predict(self, text: str, top_k: int = 1) -> Dict[str, Any]:
        """Make prediction on input text."""
        if self.config.task_type == "vsl_translation":
            return self._predict_vsl(text)
        else:
            return self._predict_classification(text, top_k)
    
    def _predict_classification(self, text: str, top_k: int = 1) -> Dict[str, Any]:
        """Predict class for input text."""
        # Tokenize input
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate if necessary
        max_length = self.config.tokenizer.max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Convert to tensors
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask=attention_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get class labels
        class_labels = self._get_class_labels()
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities[0], min(top_k, len(class_labels)))
        
        top_k_results = []
        all_probabilities = {}
        
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
            label = class_labels[idx.item()]
            prob_value = prob.item()
            
            top_k_results.append({
                'label': label,
                'probability': prob_value,
                'rank': i + 1
            })
            
            all_probabilities[label] = prob_value
        
        return {
            'predicted_label': class_labels[predicted_class],
            'confidence': confidence,
            'probabilities': all_probabilities,
            'top_k_predictions': top_k_results,
            'input_text': text
        }
    
    def _predict_vsl(self, natural_language: str) -> Dict[str, Any]:
        """Predict VSL code for natural language input."""
        nl_tokenizer, vsl_tokenizer = self.tokenizer
        
        # Tokenize natural language input
        nl_tokens = nl_tokenizer.tokenize(natural_language)
        
        # Truncate if necessary
        max_length = self.config.tokenizer.max_length
        if len(nl_tokens) > max_length:
            nl_tokens = nl_tokens[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(nl_tokens)
        
        # Convert to tensors
        input_tensor = torch.tensor([nl_tokens], dtype=torch.long, device=self.device)
        attention_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask=attention_tensor)
            
            # Get most likely tokens for each position
            predicted_tokens = torch.argmax(outputs, dim=-1).squeeze(0)
        
        # Convert back to VSL code
        vsl_tokens = []
        for token_id in predicted_tokens:
            token = vsl_tokenizer.itos.get(token_id.item(), '<unk>')
            if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                vsl_tokens.append(token)
        
        # Join tokens to form VSL code
        predicted_vsl = ''.join(vsl_tokens)
        
        # Clean up the result
        predicted_vsl = self._clean_vsl_output(predicted_vsl)
        
        return {
            'predicted_vsl': predicted_vsl,
            'input_text': natural_language,
            'confidence': 1.0  # TODO: Implement proper confidence for sequence generation
        }
    
    def _clean_vsl_output(self, vsl_code: str) -> str:
        """Clean and validate VSL output."""
        # Remove extra spaces
        vsl_code = vsl_code.strip()
        
        # Basic validation - ensure it looks like valid VSL
        if not vsl_code:
            return "A1"  # Default fallback
        
        return vsl_code
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict on a batch of texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                try:
                    result = self.predict(text)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting text '{text[:50]}...': {e}")
                    results.append({
                        'error': str(e),
                        'input_text': text
                    })
        
        return results
    
    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate model on test data."""
        if self.config.task_type == "vsl_translation":
            return self._evaluate_vsl(test_data)
        else:
            return self._evaluate_classification(test_data)
    
    def _evaluate_classification(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate classification model."""
        correct = 0
        total = len(test_data)
        
        predictions = []
        true_labels = []
        
        for text, true_label in test_data:
            result = self.predict(text)
            predicted_label = result['predicted_label']
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
            if predicted_label == true_label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-class metrics
        class_labels = self._get_class_labels()
        per_class_metrics = {}
        
        for label in class_labels:
            true_positives = sum(1 for p, t in zip(predictions, true_labels) if p == label and t == label)
            false_positives = sum(1 for p, t in zip(predictions, true_labels) if p == label and t != label)
            false_negatives = sum(1 for p, t in zip(predictions, true_labels) if p != label and t == label)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class_metrics': per_class_metrics
        }
    
    def _evaluate_vsl(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate VSL translation model."""
        exact_matches = 0
        total = len(test_data)
        
        for nl_text, true_vsl in test_data:
            result = self.predict(nl_text)
            predicted_vsl = result['predicted_vsl']
            
            if predicted_vsl.strip() == true_vsl.strip():
                exact_matches += 1
        
        exact_match_accuracy = exact_matches / total if total > 0 else 0.0
        
        return {
            'exact_match_accuracy': exact_match_accuracy,
            'exact_matches': exact_matches,
            'total': total
        }
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """Provide explanation for prediction (basic version)."""
        result = self.predict(text)
        
        # Tokenize to show what the model sees
        if self.config.task_type == "vsl_translation":
            nl_tokenizer, _ = self.tokenizer
            tokens = nl_tokenizer.tokenize(text)
            token_strings = [nl_tokenizer.itos.get(token, '<unk>') for token in tokens]
        else:
            tokens = self.tokenizer.tokenize(text)
            token_strings = [self.tokenizer.itos.get(token, '<unk>') for token in tokens]
        
        explanation = {
            'prediction': result,
            'tokenization': {
                'original_text': text,
                'tokens': token_strings,
                'token_count': len(tokens)
            },
            'model_info': {
                'task_type': self.config.task_type,
                'architecture': self.config.model.architecture,
                'vocab_size': self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 'N/A'
            }
        }
        
        return explanation


class BatchInference:
    """Utility for efficient batch inference."""
    
    def __init__(self, model_inference: ModelInference):
        self.inference = model_inference
    
    def predict_file(self, input_file: str, output_file: str, 
                    batch_size: int = 32, text_column: str = 'text'):
        """Predict on text file and save results."""
        import pandas as pd
        
        # Read input file
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input file")
        
        texts = df[text_column].tolist()
        
        # Make predictions
        predictions = self.inference.predict_batch(texts, batch_size)
        
        # Create results dataframe
        results_df = df.copy()
        
        if self.inference.config.task_type == "vsl_translation":
            results_df['predicted_vsl'] = [p.get('predicted_vsl', '') for p in predictions]
            results_df['confidence'] = [p.get('confidence', 0.0) for p in predictions]
        else:
            results_df['predicted_label'] = [p.get('predicted_label', '') for p in predictions]
            results_df['confidence'] = [p.get('confidence', 0.0) for p in predictions]
        
        # Save results
        if output_file.endswith('.csv'):
            results_df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            results_df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError("Unsupported output format. Use CSV or JSON.")
        
        logger.info(f"Batch inference completed. Results saved to {output_file}")
        return results_df


def load_model_for_inference(model_dir: Union[str, Path], 
                           device: Optional[torch.device] = None) -> ModelInference:
    """Convenience function to load model for inference."""
    return ModelInference(model_dir, device)


def predict_single(model_dir: Union[str, Path], text: str) -> Dict[str, Any]:
    """Quick single prediction."""
    inference = load_model_for_inference(model_dir)
    return inference.predict(text)
