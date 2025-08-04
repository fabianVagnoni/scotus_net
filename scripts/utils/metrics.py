#!/usr/bin/env python3
"""
Metrics utilities for SCOTUS AI model evaluation.

This module provides various metrics calculation functions for evaluating
model performance, including F1-Score, precision, recall, and other
classification metrics.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List, Dict, Tuple, Optional
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Get logger
logger = logging.getLogger(__name__)


def calculate_f1_macro(predictions: List[int], targets: List[int], 
                      num_classes: int = 3, class_names: Optional[List[str]] = None,
                      verbose: bool = False) -> float:
    """
    Calculate F1-Score Macro for multi-class classification.
    
    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        num_classes: Number of classes (default: 3 for SCOTUS voting)
        class_names: Optional list of class names for logging
        verbose: Whether to log detailed per-class metrics
        
    Returns:
        F1-Score Macro (average of per-class F1 scores)
    """
    # Convert to numpy arrays
    y_pred = np.array(predictions)
    y_true = np.array(targets)
    
    # Default class names for SCOTUS voting
    if class_names is None:
        if num_classes == 3:
            class_names = ["Majority In Favor", "Majority Against", "Majority Absent"]
        else:
            class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate F1-Score for each class
    f1_scores = []
    
    for class_idx in range(num_classes):
        # Create binary classification for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        # Calculate F1 for this class
        if np.sum(y_true_binary) == 0 and np.sum(y_pred_binary) == 0:
            # No true positives and no predicted positives - perfect for this class
            f1_class = 1.0
        elif np.sum(y_true_binary) == 0:
            # No true positives but some predicted positives - precision = 0
            f1_class = 0.0
        elif np.sum(y_pred_binary) == 0:
            # No predicted positives but some true positives - recall = 0
            f1_class = 0.0
        else:
            # Standard F1 calculation
            f1_class = f1_score(y_true_binary, y_pred_binary, zero_division=0.0)
        
        f1_scores.append(f1_class)
        
        if verbose:
            logger.debug(f"F1-Score for {class_names[class_idx]}: {f1_class:.4f}")
    
    # Calculate macro average
    f1_macro = np.mean(f1_scores)
    
    if verbose:
        logger.debug(f"Individual F1 scores: {[f'{score:.4f}' for score in f1_scores]}")
        logger.debug(f"F1-Score Macro: {f1_macro:.4f}")
    
    return f1_macro

def calculate_mrr(all_trunc_embeddings: List[torch.Tensor], all_full_embeddings: List[torch.Tensor]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for contrastive learning evaluation.
    
    Args:
        all_trunc_embeddings: List of truncated bio embeddings tensors
        all_full_embeddings: List of full bio embeddings tensors
        
    Returns:
        float: Mean Reciprocal Rank score
    """
    if not all_trunc_embeddings or not all_full_embeddings:
        logger.warning("Empty embeddings provided to calculate_mrr")
        return 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Concatenate all embeddings
    # Ensure all tensors are on the same device before concatenation
    device = all_trunc_embeddings[0].device if all_trunc_embeddings else device
    all_trunc_embeddings = [emb.to(device) for emb in all_trunc_embeddings]
    all_full_embeddings = [emb.to(device) for emb in all_full_embeddings]
    
    all_trunc_embeddings = torch.cat(all_trunc_embeddings, dim=0)  # (N, D)
    all_full_embeddings = torch.cat(all_full_embeddings, dim=0)    # (N, D)
    
    # Calculate similarity matrix
    similarity_matrix = torch.mm(all_trunc_embeddings, all_full_embeddings.t())  # (N, N)
    
    # Calculate MRR
    reciprocal_ranks = []
    for i in range(similarity_matrix.size(0)):
        # Get similarities for this truncated bio with all full bios
        similarities = similarity_matrix[i]  # (N,)
        
        # Sort in descending order and get ranks
        _, sorted_indices = torch.sort(similarities, descending=True)
        
        # Find the rank of the correct match (index i)
        # Create tensor for comparison to avoid device mismatch
        target_idx = torch.tensor(i, device=similarity_matrix.device)
        correct_rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based ranking
        
        # Add reciprocal rank
        reciprocal_ranks.append(1.0 / correct_rank)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr