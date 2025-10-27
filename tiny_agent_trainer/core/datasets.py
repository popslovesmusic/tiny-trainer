"""
Unified dataset classes with proper handling of different data types and formats.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import csv
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging
import re
from abc import ABC, abstractmethod

from tiny_agent_trainer.core.tokenizers import BaseTokenizer, pad_sequences, create_attention_mask

logger = logging.getLogger(__name__)


class BaseTaskDataset(Dataset, ABC):
    """Base class for all task datasets."""
    
    def __init__(self, tokenizer: BaseTokenizer, max_length: Optional[int] = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
    
    @abstractmethod
    def load_data(self, data_source: Union[str, List, Dict]) -> None:
        """Load data from source."""
        pass
    
    def _build_label_mapping(self, labels: List[str]):
        """Build label to index mapping."""
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        logger.info(f"Built label mapping with {len(unique_labels)} classes: {unique_labels}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item with attention mask."""
        text, label = self.data[idx], self.labels[idx]
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate if necessary
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Convert to tensors
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        label_tensor = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        
        return tokens_tensor, label_tensor, attention_mask_tensor
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_to_idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        from collections import Counter
        
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.label_to_idx)
        
        weights = []
        for idx in range(num_classes):
            label = self.idx_to_label[idx]
            weight = total_samples / (num_classes * label_counts[label])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class ClassificationDataset(BaseTaskDataset):
    """Dataset for classification tasks (sentiment, firewall, etc.)."""
    
    def load_data(self, data_source: Union[str, List[Tuple[str, str]], Dict]) -> None:
        """Load classification data."""
        
        if isinstance(data_source, str):
            # Load from file
            self._load_from_file(data_source)
        elif isinstance(data_source, list):
            # Direct list of (text, label) tuples
            self._load_from_list(data_source)
        elif isinstance(data_source, dict):
            # Configuration dict with corpus
            self._load_from_config(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _load_from_file(self, filepath: str):
        """Load data from file (JSON, CSV)."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'text' in data[0] and 'label' in data[0]:
                    texts = [item['text'] for item in data]
                    labels = [item['label'] for item in data]
                elif isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    texts = [item[0] for item in data]
                    labels = [item[1] for item in data]
                else:
                    raise ValueError("Invalid JSON format")
            else:
                raise ValueError("Empty or invalid JSON file")
        
        elif filepath.suffix == '.csv':
            texts, labels = [], []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'text' in row and 'label' in row:
                        texts.append(row['text'])
                        labels.append(row['label'])
                    else:
                        # Assume first two columns are text and label
                        row_values = list(row.values())
                        if len(row_values) >= 2:
                            texts.append(row_values[0])
                            labels.append(row_values[1])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.data = texts
        self.labels = labels
        self._build_label_mapping(labels)
        
        logger.info(f"Loaded {len(self.data)} samples from {filepath}")
    
    def _load_from_list(self, data_list: List[Tuple[str, str]]):
        """Load from list of (text, label) tuples."""
        self.data = [item[0] for item in data_list]
        self.labels = [item[1] for item in data_list]
        self._build_label_mapping(self.labels)
        
        logger.info(f"Loaded {len(self.data)} samples from list")
    
    def _load_from_config(self, config: Dict):
        """Load from config dictionary."""
        if 'corpus' not in config:
            raise ValueError("Config must contain 'corpus' key")
        
        corpus = config['corpus']
        self._load_from_list(corpus)


class VSLDataset(BaseTaskDataset):
    """Dataset for VSL (Visual Signal Language) tasks."""
    
    def __init__(self, nl_tokenizer: BaseTokenizer, vsl_tokenizer: BaseTokenizer, 
                 max_length: Optional[int] = 256):
        # Don't call super().__init__ since we need different setup
        self.nl_tokenizer = nl_tokenizer
        self.vsl_tokenizer = vsl_tokenizer
        self.max_length = max_length
        self.data = []
        self.vsl_targets = []
    
    def load_data(self, data_source: Union[str, List, Dict]) -> None:
        """Load VSL training data."""
        
        if isinstance(data_source, str):
            self._load_from_file(data_source)
        elif isinstance(data_source, list):
            self._load_from_list(data_source)
        elif isinstance(data_source, dict):
            self._load_from_config(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _load_from_file(self, filepath: str):
        """Load VSL data from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                nl_texts = []
                vsl_codes = []
                
                for item in data:
                    if isinstance(item, dict) and 'nl' in item and 'vsl' in item:
                        nl_texts.append(item['nl'])
                        vsl_codes.append(item['vsl'])
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        nl_texts.append(item[0])
                        vsl_codes.append(item[1])
                
                self.data = nl_texts
                self.vsl_targets = vsl_codes
        
        elif filepath.suffix == '.txt':
            # Parse VSL specification format
            self._load_from_spec_file(filepath)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(self.data)} VSL examples from {filepath}")
    
    def _load_from_spec_file(self, filepath: Path):
        """Load from VSL specification file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract examples using regex
        example_pattern = r"Example:\s+([^\n]+?)\s→\s\"([^\"]+)\""
        matches = re.finditer(example_pattern, content, re.DOTALL)
        
        nl_texts = []
        vsl_codes = []
        
        for match in matches:
            vsl_code = match.group(1).strip()
            nl_text = match.group(2).strip()
            
            nl_texts.append(nl_text)
            vsl_codes.append(vsl_code)
        
        self.data = nl_texts
        self.vsl_targets = vsl_codes
        
        logger.info(f"Extracted {len(self.data)} examples from specification")
    
    def _load_from_list(self, data_list: List[Tuple[str, str]]):
        """Load from list of (natural_language, vsl_code) tuples."""
        self.data = [item[0] for item in data_list]
        self.vsl_targets = [item[1] for item in data_list]
    
    def _load_from_config(self, config: Dict):
        """Load from config dictionary."""
        if 'corpus' not in config:
            raise ValueError("Config must contain 'corpus' key")
        
        corpus = config['corpus']
        if isinstance(corpus, list) and len(corpus) > 0:
            if isinstance(corpus[0], dict) and 'nl' in corpus[0] and 'vsl' in corpus[0]:
                self.data = [item['nl'] for item in corpus]
                self.vsl_targets = [item['vsl'] for item in corpus]
            else:
                self._load_from_list(corpus)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item with both NL and VSL tokens."""
        nl_text = self.data[idx]
        vsl_code = self.vsl_targets[idx]
        
        # Tokenize natural language
        nl_tokens = self.nl_tokenizer.tokenize(nl_text)
        if self.max_length and len(nl_tokens) > self.max_length:
            nl_tokens = nl_tokens[:self.max_length]
        
        # Tokenize VSL code
        vsl_tokens = self.vsl_tokenizer.tokenize(vsl_code)
        if self.max_length and len(vsl_tokens) > self.max_length:
            vsl_tokens = vsl_tokens[:self.max_length]
        
        # Create attention masks
        nl_attention_mask = [1] * len(nl_tokens)
        vsl_attention_mask = [1] * len(vsl_tokens)
        
        # Convert to tensors
        nl_tensor = torch.tensor(nl_tokens, dtype=torch.long)
        vsl_tensor = torch.tensor(vsl_tokens, dtype=torch.long)
        nl_attention_tensor = torch.tensor(nl_attention_mask, dtype=torch.long)
        vsl_attention_tensor = torch.tensor(vsl_attention_mask, dtype=torch.long)
        
        return nl_tensor, vsl_tensor, nl_attention_tensor, vsl_attention_tensor


class MultiTaskDataset(Dataset):
    """Dataset that combines multiple tasks."""
    
    def __init__(self, datasets: Dict[str, BaseTaskDataset]):
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.cumulative_lengths = []
        self.total_length = 0
        
        # Calculate cumulative lengths
        for dataset in datasets.values():
            self.total_length += len(dataset)
            self.cumulative_lengths.append(self.total_length)
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int):
        """Get item from appropriate sub-dataset."""
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_name = self.task_names[i]
                dataset = self.datasets[dataset_name]
                
                # Calculate local index
                local_idx = idx - (self.cumulative_lengths[i-1] if i > 0 else 0)
                
                # Get item from dataset
                item = dataset[local_idx]
                
                # Add task identifier
                return item + (torch.tensor(i, dtype=torch.long),)
        
        raise IndexError("Index out of range")


def collate_fn(batch: List[Tuple[torch.Tensor, ...]], pad_token_id: int = 0):
    """Collate function for batching with proper padding."""
    
    if len(batch[0]) == 3:  # Classification: (tokens, label, attention_mask)
        tokens, labels, attention_masks = zip(*batch)
        
        # Pad sequences
        padded_tokens, padded_attention_masks = pad_sequences(
            [t.tolist() for t in tokens], 
            pad_token_id=pad_token_id
        )
        
        return (
            torch.tensor(padded_tokens, dtype=torch.long),
            torch.stack(labels),
            torch.tensor(padded_attention_masks, dtype=torch.long)
        )
    
    elif len(batch[0]) == 4:  # VSL: (nl_tokens, vsl_tokens, nl_attention, vsl_attention)
        nl_tokens, vsl_tokens, nl_attention, vsl_attention = zip(*batch)
        
        # Pad NL sequences
        padded_nl_tokens, padded_nl_attention = pad_sequences(
            [t.tolist() for t in nl_tokens],
            pad_token_id=pad_token_id
        )
        
        # Pad VSL sequences  
        padded_vsl_tokens, padded_vsl_attention = pad_sequences(
            [t.tolist() for t in vsl_tokens],
            pad_token_id=pad_token_id
        )
        
        return (
            torch.tensor(padded_nl_tokens, dtype=torch.long),
            torch.tensor(padded_vsl_tokens, dtype=torch.long),
            torch.tensor(padded_nl_attention, dtype=torch.long),
            torch.tensor(padded_vsl_attention, dtype=torch.long)
        )
    
    else:
        # Generic handling for other cases
        return torch.utils.data.dataloader.default_collate(batch)


def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, 
                     num_workers: int = 0, **kwargs) -> DataLoader:
    """Create dataloader with appropriate collate function."""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=0),
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/validation/test sets."""
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )


# Factory functions
def create_classification_dataset(data_source: Union[str, List, Dict], 
                                tokenizer: BaseTokenizer, 
                                max_length: Optional[int] = 512) -> ClassificationDataset:
    """Create classification dataset."""
    dataset = ClassificationDataset(tokenizer, max_length)
    dataset.load_data(data_source)
    return dataset


def create_vsl_dataset(data_source: Union[str, List, Dict],
                      nl_tokenizer: BaseTokenizer,
                      vsl_tokenizer: BaseTokenizer,
                      max_length: Optional[int] = 256) -> VSLDataset:
    """Create VSL dataset."""
    dataset = VSLDataset(nl_tokenizer, vsl_tokenizer, max_length)
    dataset.load_data(data_source)
    return dataset
