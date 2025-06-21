#!/usr/bin/env python3
"""
Justice Cross-Attention module for SCOTUS voting prediction.

This module implements multi-head attention mechanism that allows case embeddings
to query justice embeddings, producing contextualized court representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class JusticeCrossAttention(nn.Module):
    """
    Multi-head attention mechanism for justice embeddings.
    
    Uses the case embedding as query to attend to justice embeddings,
    producing a contextualized court representation.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # Use batch_first for easier handling
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optional feed-forward network for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.ffn_layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, 
                case_emb: torch.Tensor,       # (batch_size, embedding_dim) or (embedding_dim,)
                justice_embs: torch.Tensor,   # (batch_size, max_justices, embedding_dim) or (max_justices, embedding_dim) 
                justice_mask: Optional[torch.Tensor] = None  # (batch_size, max_justices) - True for real justices, False for padding
               ) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.
        
        Args:
            case_emb: Case embedding
            justice_embs: Justice embeddings (may include padding)
            justice_mask: Mask for real vs padded justices
            
        Returns:
            Contextualized court embedding
        """
        # Handle both single sample and batch cases
        single_sample = False
        if case_emb.dim() == 1:
            single_sample = True
            case_emb = case_emb.unsqueeze(0)  # (1, embedding_dim)
            justice_embs = justice_embs.unsqueeze(0)  # (1, max_justices, embedding_dim)
            if justice_mask is not None:
                justice_mask = justice_mask.unsqueeze(0)  # (1, max_justices)
        
        batch_size = case_emb.size(0)
        
        # Prepare query, key, value
        # Query: case embedding expanded to match justice sequence length
        query = case_emb.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        key = justice_embs  # (batch_size, max_justices, embedding_dim)
        value = justice_embs  # (batch_size, max_justices, embedding_dim)
        
        # Create attention mask if not provided
        if justice_mask is None:
            # Assume all justices are real (no padding) - this shouldn't happen in practice
            justice_mask = torch.ones(batch_size, justice_embs.size(1), dtype=torch.bool, device=case_emb.device)
        
        # Convert mask for attention (MultiheadAttention expects inverted mask)
        # True values are ignored in attention, False values are attended to
        attn_mask = ~justice_mask  # Invert: True for padding, False for real justices
        
        # Apply multi-head attention
        attn_output = self.attn(
            query=query,
            key=key, 
            value=value,
            key_padding_mask=attn_mask  # (batch_size, max_justices)
        ) 
        
        # attn_output shape: (batch_size, 1, embedding_dim)
        court_emb = attn_output.squeeze(1)  # (batch_size, embedding_dim)
        
        # Residual connection with case embedding
        court_emb = self.layer_norm(court_emb + case_emb)
        court_emb = self.dropout(court_emb)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(court_emb)
        court_emb = self.ffn_layer_norm(court_emb + ffn_output)
        
        # Handle single sample case
        if single_sample:
            court_emb = court_emb.squeeze(0)  # Remove batch dimension
        
        return court_emb