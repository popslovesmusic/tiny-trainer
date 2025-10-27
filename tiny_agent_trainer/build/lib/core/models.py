"""
Unified model architectures for the Tiny Agent Trainer.
Supports both RNN and Transformer architectures with proper GPU utilization.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ImprovedRNN(nn.Module):
    """Enhanced RNN with proper architecture and GPU support."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 128, 
        hidden_dim: int = 256, 
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        if attention_mask is not None:
            # Handle variable length sequences
            lengths = attention_mask.sum(dim=1).cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(embedded)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Final layers
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


class ImprovedTransformer(nn.Module):
    """Enhanced transformer with proper attention masks and architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 2,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = ((d_model // nhead) + 1) * nhead
            logger.warning(f"Adjusted d_model to {d_model} to be divisible by nhead={nhead}")
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following transformer best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding with scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create attention mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            # Transformer expects True for positions to ignore
            src_key_padding_mask = ~attention_mask.bool()
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling (better than just taking first token)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Final classification
        x = self.layer_norm(x)
        x = self.dropout(x)
        output = self.classifier(x)
        
        return output


class VSLTransformer(nn.Module):
    """Specialized transformer for VSL code generation."""
    
    def __init__(
        self,
        nl_vocab_size: int,
        vsl_vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        max_seq_len: int = 256
    ):
        super().__init__()
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = ((d_model // nhead) + 1) * nhead
        
        self.d_model = d_model
        self.nl_embedding = nn.Embedding(nl_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vsl_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.nl_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Attention mask for padding
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        
        # Encode
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Project to VSL vocabulary
        output = self.output_projection(encoded)
        
        return output


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create models based on configuration."""
    
    model_type = config.get('architecture', 'transformer').lower()
    vocab_size = config['vocab_size']
    output_dim = config['output_dim']
    
    if model_type == 'rnn':
        return ImprovedRNN(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=output_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            bidirectional=config.get('bidirectional', True)
        )
    
    elif model_type == 'transformer':
        return ImprovedTransformer(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_encoder_layers=config.get('num_layers', 3),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.1),
            output_dim=output_dim,
            max_seq_len=config.get('max_seq_len', 512)
        )
    
    elif model_type == 'vsl_transformer':
        return VSLTransformer(
            nl_vocab_size=config['nl_vocab_size'],
            vsl_vocab_size=config['vsl_vocab_size'],
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 3),
            max_seq_len=config.get('max_seq_len', 256)
        )
    
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")
