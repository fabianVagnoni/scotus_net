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