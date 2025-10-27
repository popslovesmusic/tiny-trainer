import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

# New dataclasses to represent the YAML structure
# This is a key change that makes the configuration objects easier to use and validate
@dataclass
class ModelConfig:
    architecture: str
    d_model: int
    nhead: int
    num_layers: int
    embedding_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: Optional[float] = 0.1
    max_seq_len: Optional[int] = 100

@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "adamw"
    early_stopping: bool = True

@dataclass
class TokenizerConfig:
    type: str
    max_length: int
    lowercase: bool
    min_freq: Optional[int] = 1

@dataclass
class DataConfig:
    corpus: List[Any]
    
@dataclass
class SecurityConfig:
    enable_vsl_execution: bool
    vsl_timeout: int
    validate_inputs: bool

@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    security: SecurityConfig

    def to_dict(self):
        """Converts the dataclass object to a dictionary for saving."""
        data = {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'tokenizer': self.tokenizer.__dict__,
            'data': self.data.__dict__,
            'security': self.security.__dict__
        }
        return data

