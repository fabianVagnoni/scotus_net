#!/usr/bin/env python3
"""
Hyperparameter Optimization for SCOTUS Voting Model using Optuna.

This script performs hyperparameter tuning for both model architecture and training parameters
to find the optimal configuration for SCOTUS voting prediction.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import time
import signal
from tqdm import tqdm

from scripts.models.model_trainer import SCOTUSModelTrainer
from scripts.models.scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
from torch.utils.data import DataLoader
from scripts.models.config import ModelConfig
from scripts.utils.logger import get_logger
from scripts.utils.holdout_test_set import HoldoutTestSetManager
from scripts.models.losses import create_scotus_loss_function


class OptunaModelTrainer(SCOTUSModelTrainer):
    """
    Extended model trainer for Optuna optimization with early stopping and validation tracking.
    """
    
    def __init__(self, trial: Trial, base_config: ModelConfig):
        """Initialize trainer with trial-specific configuration."""
        super().__init__()
        self.trial = trial
        self.base_config = base_config
        self.best_val_loss = float('inf')
        self.early_stop_patience = base_config.optuna_early_stop_patience
        self.min_epochs = base_config.optuna_min_epochs
        # Override holdout manager to ensure we exclude holdout cases
        self.holdout_manager = HoldoutTestSetManager()
        
    def train_model_for_optimization(self, dataset_file: str = None) -> float:
        """
        Train model with trial-specific hyperparameters and return combined optimization metric.
        
        Returns:
            Combined metric: (KL Divergence Loss + (1 - F1-Score Macro)) / 2
            This balances probabilistic accuracy (loss) with classification performance (F1)
        """
        start_time = time.time()
        max_trial_time = self.base_config.optuna_max_trial_time
        
        try:
            self.logger.info("Loading dataset for optimization...")
            # Load dataset
            dataset = self.load_case_dataset(dataset_file)
            
            # Get tokenized file paths
            bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths(dataset)
            
            self.logger.info("Splitting dataset...")
            # Split dataset (use smaller validation set for faster optimization)
            train_dataset, val_dataset = self.split_dataset(
                dataset, 
                train_ratio=0.8, 
                val_ratio=0.2
            )
            
            self.logger.info("Suggesting hyperparameters...")
            # Suggest hyperparameters
            hyperparams = self._suggest_hyperparameters()
            self.logger.info(f"Trial hyperparameters: {hyperparams}")
            
            # Log which parameters are tuned vs fixed
            tuned_params = []
            fixed_params = []
            
            for param_name in ['hidden_dim', 'dropout_rate', 'num_attention_heads', 'use_justice_attention', 'learning_rate', 'batch_size', 'weight_decay', 'unfreeze_epoch']:
                tune_flag = getattr(self.base_config, f'tune_{param_name}')
                if tune_flag:
                    tuned_params.append(f"{param_name}={hyperparams[param_name]}")
                else:
                    fixed_params.append(f"{param_name}={hyperparams[param_name]}")
            
            if tuned_params:
                self.logger.info(f"Tuned parameters: {', '.join(tuned_params)}")
            if fixed_params:
                self.logger.info(f"Fixed parameters: {', '.join(fixed_params)}")
            
            self.logger.info("Initializing model...")
            # Initialize model with trial hyperparameters
            model = SCOTUSVotingModel(
                bio_tokenized_file=bio_tokenized_file,
                description_tokenized_file=description_tokenized_file,
                bio_model_name=self.base_config.bio_model_name,
                description_model_name=self.base_config.description_model_name,
                embedding_dim=self.base_config.embedding_dim,  # Keep fixed for compatibility
                hidden_dim=hyperparams['hidden_dim'],
                dropout_rate=hyperparams['dropout_rate'],
                max_justices=self.base_config.max_justices,  # Keep fixed
                num_attention_heads=hyperparams['num_attention_heads'],
                use_justice_attention=hyperparams['use_justice_attention'],
                device=str(self.device)
            )
            
            model.to(self.device)
            self.logger.info("Model moved to device")
            
            # Override the unfreeze epoch for this trial
            trial_unfreeze_epoch = hyperparams['unfreeze_epoch']
            self.logger.info(f"Trial will unfreeze sentence transformers at epoch: {trial_unfreeze_epoch}")
            
            self.logger.info("Preparing datasets...")
            # Prepare datasets
            train_dataset_dict = self.prepare_dataset_dict(train_dataset)
            val_dataset_dict = self.prepare_dataset_dict(val_dataset)
            
            if not train_dataset_dict or not val_dataset_dict:
                raise ValueError("No valid training or validation cases found")
            
            # Limit dataset size for faster optimization
            max_train_samples = len(train_dataset_dict)
            max_val_samples = len(val_dataset_dict)
            
            if self.base_config.optuna_max_train_samples > 0:
                max_train_samples = min(len(train_dataset_dict), self.base_config.optuna_max_train_samples)
            if self.base_config.optuna_max_val_samples > 0:
                max_val_samples = min(len(val_dataset_dict), self.base_config.optuna_max_val_samples)
            
            self.logger.info(f"Using {max_train_samples} training samples and {max_val_samples} validation samples")
            
            train_keys = list(train_dataset_dict.keys())[:max_train_samples]
            val_keys = list(val_dataset_dict.keys())[:max_val_samples]
            
            train_subset = {k: train_dataset_dict[k] for k in train_keys}
            val_subset = {k: val_dataset_dict[k] for k in val_keys}
            
            train_pytorch_dataset = SCOTUSDataset(train_subset)
            val_pytorch_dataset = SCOTUSDataset(val_subset)
            
            self.logger.info("Creating data loaders...")
            # Data loaders with trial batch size
            train_loader = DataLoader(
                train_pytorch_dataset, 
                batch_size=hyperparams['batch_size'], 
                shuffle=True, 
                collate_fn=collate_fn,
                num_workers=0  # Keep 0 for stability
            )
            val_loader = DataLoader(
                val_pytorch_dataset, 
                batch_size=hyperparams['batch_size'], 
                shuffle=False, 
                collate_fn=collate_fn,
                num_workers=0
            )
            
            self.logger.info("Setting up optimizer and loss function...")
            # Training setup with trial hyperparameters
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=hyperparams['learning_rate'], 
                weight_decay=hyperparams['weight_decay']
            )
            
            # Setup sentence transformer fine-tuning if enabled
            sentence_transformer_optimizer = None
            if self.base_config.enable_sentence_transformer_finetuning and trial_unfreeze_epoch >= 0:
                # Create separate optimizer for sentence transformers
                sentence_transformer_params = []
                if self.base_config.unfreeze_bio_model:
                    sentence_transformer_params.extend(model.bio_model.parameters())
                if self.base_config.unfreeze_description_model:
                    sentence_transformer_params.extend(model.description_model.parameters())
                
                if sentence_transformer_params:
                    sentence_transformer_optimizer = torch.optim.AdamW(
                        sentence_transformer_params, 
                        lr=self.base_config.sentence_transformer_learning_rate, 
                        weight_decay=hyperparams['weight_decay']
                    )
            
            # Setup loss function using the new modular system
            criterion = create_scotus_loss_function(self.base_config.loss_function, self.base_config)
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )
            
            self.logger.info("Starting training loop...")
            # Training loop with early stopping
            model.train()
            best_combined_metric = float('inf')
            patience_counter = 0
            
            max_epochs = self.base_config.optuna_max_epochs
            
            for epoch in range(max_epochs):
                # Check timeout (if enabled)
                if max_trial_time > 0 and time.time() - start_time > max_trial_time:
                    self.logger.warning(f"Trial timeout after {max_trial_time} seconds")
                    break
                
                # Check if we should unfreeze sentence transformers at this epoch
                if (self.base_config.enable_sentence_transformer_finetuning and 
                    trial_unfreeze_epoch == epoch and
                    sentence_transformer_optimizer is not None):
                    
                    self.logger.info(f"🔓 Unfreezing sentence transformers at epoch {epoch + 1}")
                    model.unfreeze_models_selectively(
                        unfreeze_bio=self.base_config.unfreeze_bio_model,
                        unfreeze_description=self.base_config.unfreeze_description_model
                    )
                    
                    # Log the status
                    status = model.get_sentence_transformer_status()
                    self.logger.info(f"   Bio model trainable: {status['bio_model_trainable']}")
                    self.logger.info(f"   Description model trainable: {status['description_model_trainable']}")
                
                self.logger.info(f"Starting epoch {epoch+1}/{max_epochs}")
                # Training phase
                model.train()
                train_loss = 0.0
                num_batches = 0
                
                # Training progress bar
                train_pbar = tqdm(
                    enumerate(train_loader), 
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{max_epochs} - Training",
                    leave=False,
                    ncols=100
                )
                
                for batch_idx, batch in train_pbar:
                    try:
                        optimizer.zero_grad()
                        if sentence_transformer_optimizer is not None:
                            sentence_transformer_optimizer.zero_grad()
                        
                        batch_predictions = []
                        batch_targets = batch['targets'].to(self.device)
                        
                        # Process each sample in the batch
                        for i in range(len(batch['case_ids'])):
                            case_description_path = batch['case_description_paths'][i]
                            justice_bio_paths = batch['justice_bio_paths'][i]
                            
                            # Forward pass
                            prediction = model(case_description_path, justice_bio_paths)
                            batch_predictions.append(prediction)
                        
                        # Stack predictions and compute loss
                        predictions_tensor = torch.stack(batch_predictions)
                        
                        # Compute loss using modular loss system (pass epoch for annealing)
                        loss = criterion(predictions_tensor, batch_targets, epoch=epoch)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        if sentence_transformer_optimizer is not None:
                            # Check if sentence transformers are actually unfrozen before stepping
                            status = model.get_sentence_transformer_status()
                            if status['any_trainable']:
                                sentence_transformer_optimizer.step()
                        
                        train_loss += loss.item()
                        num_batches += 1
                        
                        # Update progress bar with current loss
                        current_avg_loss = train_loss / num_batches
                        train_pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
                        
                    except Exception as e:
                        self.logger.warning(f"Error in training batch {batch_idx}: {e}")
                        continue
                
                train_pbar.close()
                
                if num_batches == 0:
                    raise ValueError("No successful training batches")
                    
                avg_train_loss = train_loss / num_batches
                
                self.logger.info(f"Epoch {epoch+1} training completed, avg loss: {avg_train_loss:.4f}")
                
                # Validation phase
                self.logger.info(f"Starting validation for epoch {epoch+1}")
                combined_metric = self._evaluate_model_for_optimization(model, val_loader, criterion)
                
                # Learning rate scheduling (using combined metric)
                scheduler.step(combined_metric)
                
                # Early stopping check
                if combined_metric < best_combined_metric:
                    best_combined_metric = combined_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                self.logger.info(f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.4f}, Combined Metric: {combined_metric:.4f}")
                
                # Report intermediate value to Optuna
                self.trial.report(combined_metric, epoch)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                
                # Early stopping
                if patience_counter >= self.early_stop_patience and epoch >= self.min_epochs:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.logger.info(f"Trial completed with best combined metric: {best_combined_metric:.4f}")
            return best_combined_metric
            
        except Exception as e:
            self.logger.error(f"Error in trial: {e}")
            return float('inf')
    
    def _suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        hyperparams = {}
        
        # Model Architecture - Hidden Dimension
        if self.base_config.tune_hidden_dim:
            hyperparams['hidden_dim'] = self.trial.suggest_categorical('hidden_dim', self.base_config.optuna_hidden_dim_options)
        else:
            hyperparams['hidden_dim'] = self.base_config.hidden_dim
        
        # Regularization - Dropout Rate
        if self.base_config.tune_dropout_rate:
            dropout_min, dropout_max, dropout_step = self.base_config.optuna_dropout_rate_range
            dropout_kwargs = {'step': dropout_step} if dropout_step is not None else {}
            hyperparams['dropout_rate'] = self.trial.suggest_float('dropout_rate', dropout_min, dropout_max, **dropout_kwargs)
        else:
            hyperparams['dropout_rate'] = self.base_config.dropout_rate
        
        # Attention Mechanism - Number of Heads
        if self.base_config.tune_num_attention_heads:
            hyperparams['num_attention_heads'] = self.trial.suggest_categorical('num_attention_heads', self.base_config.optuna_attention_heads_options)
        else:
            hyperparams['num_attention_heads'] = self.base_config.num_attention_heads
        
        # Attention Mechanism - Use Justice Attention
        if self.base_config.tune_use_justice_attention:
            hyperparams['use_justice_attention'] = self.trial.suggest_categorical('use_justice_attention', self.base_config.optuna_justice_attention_options)
        else:
            hyperparams['use_justice_attention'] = self.base_config.use_justice_attention
        
        # Training Parameters - Learning Rate
        if self.base_config.tune_learning_rate:
            lr_min, lr_max, lr_log = self.base_config.optuna_learning_rate_range
            lr_kwargs = {'log': lr_log} if lr_log else {}
            hyperparams['learning_rate'] = self.trial.suggest_float('learning_rate', lr_min, lr_max, **lr_kwargs)
        else:
            hyperparams['learning_rate'] = self.base_config.learning_rate
        
        # Training Parameters - Batch Size
        if self.base_config.tune_batch_size:
            hyperparams['batch_size'] = self.trial.suggest_categorical('batch_size', self.base_config.optuna_batch_size_options)
        else:
            hyperparams['batch_size'] = self.base_config.batch_size
        
        # Regularization - Weight Decay
        if self.base_config.tune_weight_decay:
            wd_min, wd_max, wd_log = self.base_config.optuna_weight_decay_range
            wd_kwargs = {'log': wd_log} if wd_log else {}
            hyperparams['weight_decay'] = self.trial.suggest_float('weight_decay', wd_min, wd_max, **wd_kwargs)
        else:
            hyperparams['weight_decay'] = self.base_config.weight_decay
        
        # Sentence Transformer Fine-tuning - Unfreeze Epoch
        if self.base_config.tune_unfreeze_epoch:
            hyperparams['unfreeze_epoch'] = self.trial.suggest_categorical('unfreeze_epoch', self.base_config.optuna_unfreeze_epoch_options)
        else:
            hyperparams['unfreeze_epoch'] = self.base_config.unfreeze_sentence_transformers_epoch
        
        return hyperparams
    
    def _evaluate_model_for_optimization(self, model: SCOTUSVotingModel, data_loader, criterion) -> float:
        """
        Evaluate model for optimization using combined loss and F1-Score Macro.
        
        Returns:
            Combined metric: (KL Divergence Loss + (1 - F1-Score Macro)) / 2
            This ensures both metrics are in the same direction (lower is better)
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        first_batch_logged = False
        
        # For F1-Score calculation
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    batch_predictions = []
                    batch_targets = batch['targets'].to(self.device)
                    
                    # Process each sample in the batch
                    for i in range(len(batch['case_ids'])):
                        case_description_path = batch['case_description_paths'][i]
                        justice_bio_paths = batch['justice_bio_paths'][i]
                        
                        # Forward pass
                        prediction = model(case_description_path, justice_bio_paths)
                        batch_predictions.append(prediction)
                    
                    # Stack predictions and compute loss
                    predictions_tensor = torch.stack(batch_predictions)
                    
                    # Compute loss using modular loss system
                    loss = criterion(predictions_tensor, batch_targets)
                    
                    # Store predictions and targets for F1-Score calculation
                    # Convert to class predictions (argmax) for F1-Score
                    predicted_classes = torch.argmax(predictions_tensor, dim=1).cpu().numpy()
                    target_classes = torch.argmax(batch_targets, dim=1).cpu().numpy()
                    
                    all_predictions.extend(predicted_classes)
                    all_targets.extend(target_classes)
                    
                    # Log predictions vs targets for first batch only
                    if not first_batch_logged and batch_idx == 0:
                        self.logger.info("=== VALIDATION PREDICTIONS vs TARGETS ===")
                        
                        # Convert log predictions back to probabilities for display
                        display_predictions = torch.softmax(predictions_tensor, dim=1)
                        
                        # Show first few samples in the batch
                        num_samples_to_show = min(3, len(batch['case_ids']))
                        
                        for i in range(num_samples_to_show):
                            case_id = batch['case_ids'][i]
                            pred = display_predictions[i].cpu().numpy()
                            target = batch_targets[i].cpu().numpy()
                            
                            self.logger.info(f"Sample {i+1} (Case ID: {case_id}):")
                            self.logger.info(f"  Predicted:  [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}, {pred[3]:.4f}]")
                            self.logger.info(f"  Target:     [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}, {target[3]:.4f}]")
                            
                            # Calculate and show the difference
                            diff = pred - target
                            self.logger.info(f"  Difference: [{diff[0]:+.4f}, {diff[1]:+.4f}, {diff[2]:+.4f}, {diff[3]:+.4f}]")
                            
                            # Show which class has highest prediction vs target
                            pred_class = pred.argmax()
                            target_class = target.argmax()
                            class_names = ["Majority In Favor", "Majority Against", "Majority Absent", "Other"]
                            self.logger.info(f"  Pred Class: {class_names[pred_class]} ({pred[pred_class]:.4f})")
                            self.logger.info(f"  True Class: {class_names[target_class]} ({target[target_class]:.4f})")
                            self.logger.info("")
                        
                        first_batch_logged = True
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error in evaluation batch: {e}")
                    continue
        
        model.train()
        
        if num_batches == 0:
            return float('inf')
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Calculate F1-Score Macro
        f1_macro = self._calculate_f1_macro(all_predictions, all_targets)
        
        # Combine metrics: (Loss + (1 - F1)) / 2
        # This ensures both metrics are minimized (lower is better)
        combined_metric = (avg_loss + (1.0 - f1_macro)) / 2.0
        
        self.logger.info(f"Evaluation metrics - Loss: {avg_loss:.4f}, F1-Macro: {f1_macro:.4f}, Combined: {combined_metric:.4f}")
        
        return combined_metric
    
    def _calculate_f1_macro(self, predictions: list, targets: list) -> float:
        """
        Calculate F1-Score Macro for 4 classes.
        
        Args:
            predictions: List of predicted class indices
            targets: List of target class indices
            
        Returns:
            F1-Score Macro (average of per-class F1 scores)
        """
        from sklearn.metrics import f1_score
        import numpy as np
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(targets)
        
        # Class names for logging
        class_names = ["Majority In Favor", "Majority Against", "Majority Absent", "Other"]
        
        # Calculate F1-Score for each class
        f1_scores = []
        
        for class_idx in range(4):  # 4 classes: 0, 1, 2, 3
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
            self.logger.debug(f"F1-Score for {class_names[class_idx]}: {f1_class:.4f}")
        
        # Calculate macro average
        f1_macro = np.mean(f1_scores)
        
        self.logger.debug(f"Individual F1 scores: {[f'{score:.4f}' for score in f1_scores]}")
        self.logger.debug(f"F1-Score Macro: {f1_macro:.4f}")
        
        return f1_macro

    def _log_tuning_configuration(self):
        """Log the tuning configuration for debugging purposes."""
        self.logger.info("Tuning configuration:")
        self.logger.info(f"  - Hidden Dimension: {'Tuned' if self.base_config.tune_hidden_dim else 'Fixed'}")
        self.logger.info(f"  - Dropout Rate: {'Tuned' if self.base_config.tune_dropout_rate else 'Fixed'}")
        self.logger.info(f"  - Number of Attention Heads: {'Tuned' if self.base_config.tune_num_attention_heads else 'Fixed'}")
        self.logger.info(f"  - Use Justice Attention: {'Tuned' if self.base_config.tune_use_justice_attention else 'Fixed'}")
        self.logger.info(f"  - Learning Rate: {'Tuned' if self.base_config.tune_learning_rate else 'Fixed'}")
        self.logger.info(f"  - Batch Size: {'Tuned' if self.base_config.tune_batch_size else 'Fixed'}")
        self.logger.info(f"  - Weight Decay: {'Tuned' if self.base_config.tune_weight_decay else 'Fixed'}")
        self.logger.info(f"  - Unfreeze Epoch: {'Tuned' if self.base_config.tune_unfreeze_epoch else 'Fixed'}")


def initialize_results_file(study_name: str, n_trials: int, dataset_file: str):
    """
    Initialize the tuning results file with header information.
    
    Args:
        study_name: Name of the optimization study
        n_trials: Number of trials to run
        dataset_file: Path to dataset file
    """
    results_file = Path("scripts/models/tunning_results.txt")
    
    # Create directory if it doesn't exist
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_lines = [
        f"{'#' * 80}",
        f"# SCOTUS AI Hyperparameter Optimization Results",
        f"# Study: {study_name}",
        f"# Started: {timestamp}",
        f"# Number of trials: {n_trials}",
        f"# Dataset: {dataset_file}",
        f"{'#' * 80}",
        "",
    ]
    
    # Write header to file (overwrite mode to start fresh)
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header_lines) + '\n')
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to initialize results file: {e}")


def append_trial_results(trial: Trial, combined_metric: float, hyperparams: Dict[str, Any] = None, trial_status: str = "COMPLETED"):
    """
    Append trial results to the tuning results file.
    
    Args:
        trial: Optuna trial object
        combined_metric: Combined optimization metric achieved
        hyperparams: Trial hyperparameters (optional, will be extracted from trial if not provided)
        trial_status: Status of the trial (COMPLETED, PRUNED, FAILED)
    """
    # Create the results file path
    results_file = Path("scripts/models/tunning_results.txt")
    
    # Create directory if it doesn't exist
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get hyperparameters if not provided
    if hyperparams is None:
        hyperparams = trial.params
    
    # Prepare trial results text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format results
    result_lines = [
        f"=" * 80,
        f"Trial #{trial.number} - {timestamp}",
        f"Combined Metric (Loss + (1-F1))/2: {combined_metric:.6f}",
        f"Trial Status: {trial_status}",
        f"Parameters:",
    ]
    
    # Add hyperparameters
    for key, value in hyperparams.items():
        result_lines.append(f"  {key}: {value}")
    
    result_lines.extend([
        "",  # Empty line for separation
    ])
    
    # Write to file (append mode)
    try:
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(result_lines) + '\n')
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to append trial results to file: {e}")


def append_optimization_summary(study: optuna.Study):
    """
    Append optimization summary to the results file.
    
    Args:
        study: Completed Optuna study
    """
    results_file = Path("scripts/models/tunning_results.txt")
    
    # Prepare summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_lines = [
        f"{'#' * 80}",
        f"# OPTIMIZATION SUMMARY - {timestamp}",
        f"{'#' * 80}",
        f"Total trials: {len(study.trials)}",
        f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}",
        f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}",
        f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}",
        "",
        f"BEST TRIAL: #{study.best_trial.number}",
        f"Best combined metric (Loss + (1-F1))/2: {study.best_value:.6f}",
        f"Best parameters:",
    ]
    
    # Add best parameters
    for key, value in study.best_params.items():
        summary_lines.append(f"  {key}: {value}")
    
    summary_lines.extend([
        "",
        f"Study completed at: {timestamp}",
        f"{'#' * 80}",
        "",
    ])
    
    # Write to file (append mode)
    try:
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines) + '\n')
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to append optimization summary to file: {e}")


def objective(trial: Trial, base_config: ModelConfig, dataset_file: str) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        dataset_file: Path to dataset file
        
    Returns:
        Combined metric to minimize: (KL Divergence Loss + (1 - F1-Score Macro)) / 2
        This balances probabilistic accuracy with classification performance
    """
    logger = get_logger(__name__)
    logger.info(f"Starting trial {trial.number}")
    
    try:
        # Create trainer for this trial
        trainer = OptunaModelTrainer(trial, base_config)
        
        # Train model and get combined metric
        combined_metric = trainer.train_model_for_optimization(dataset_file)
        
        logger.info(f"Trial {trial.number} completed with combined metric: {combined_metric:.4f}")
        
        # Append trial results to file
        append_trial_results(trial, combined_metric, trial_status="COMPLETED")
        
        return combined_metric
        
    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        # Also log pruned trials
        append_trial_results(trial, float('inf'), trial_status="PRUNED")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        # Log failed trials
        append_trial_results(trial, float('inf'), trial_status="FAILED")
        return float('inf')


def run_hyperparameter_optimization(
    n_trials: int = None,
    study_name: str = None,
    dataset_file: str = None,
    storage: str = None,
    n_jobs: int = 1,
    timeout: Optional[int] = None
) -> optuna.Study:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name for the study (default: auto-generated)
        dataset_file: Path to dataset file
        storage: Storage backend for study persistence
        n_jobs: Number of parallel jobs (1 for sequential)
        timeout: Timeout in seconds
        
    Returns:
        Optuna study object with results
    """
    logger = get_logger(__name__)
    
    # Load base configuration
    base_config = ModelConfig()
    
    if dataset_file is None:
        dataset_file = base_config.dataset_file
    
    if n_trials is None:
        n_trials = base_config.optuna_n_trials
    
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"scotus_optimization_{timestamp}"
    
    logger.info(f"🔍 Starting hyperparameter optimization")
    logger.info(f"   📊 Study name: {study_name}")
    logger.info(f"   🎯 Number of trials: {n_trials}")
    logger.info(f"   📁 Dataset: {dataset_file}")
    logger.info(f"   💾 Storage: {storage or 'in-memory'}")
    
    # Initialize results file with header
    initialize_results_file(study_name, n_trials, dataset_file)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # Minimize combined metric: (Loss + (1-F1))/2
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=base_config.optuna_pruner_startup_trials, 
            n_warmup_steps=base_config.optuna_pruner_warmup_steps
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, dataset_file),
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Print results
    logger.info(f"🎉 Optimization completed!")
    logger.info(f"   🏆 Best trial: {study.best_trial.number}")
    logger.info(f"   📈 Best combined metric: {study.best_value:.4f}")
    logger.info(f"   ⚙️  Best parameters:")
    
    for key, value in study.best_params.items():
        logger.info(f"      {key}: {value}")
    
    # Append final summary to results file
    append_optimization_summary(study)
    
    return study


def save_best_config(study: optuna.Study, output_file: str = "best_config.env"):
    """
    Save the best hyperparameters as a new config file.
    
    Args:
        study: Completed Optuna study
        output_file: Output configuration file path
    """
    logger = get_logger(__name__)
    
    # Load base config to get non-optimized parameters
    base_config = ModelConfig()
    
    # Create new config with best parameters
    best_params = study.best_params
    
    config_lines = [
        "# SCOTUS AI Model Configuration - Optimized with Optuna",
        f"# Study: {study.study_name}",
        f"# Best trial: {study.best_trial.number}",
        f"# Best combined metric (Loss + (1-F1))/2: {study.best_value:.6f}",
        f"# Optimization date: {datetime.now().isoformat()}",
        f"# Note: Optimized using combined KL Divergence Loss and F1-Score Macro",
        "",
        "# Model Architecture",
        f"BIO_MODEL_NAME={base_config.bio_model_name}",
        f"DESCRIPTION_MODEL_NAME={base_config.description_model_name}",
        f"EMBEDDING_DIM={base_config.embedding_dim}",
        f"HIDDEN_DIM={best_params.get('hidden_dim', base_config.hidden_dim)}",
        f"MAX_JUSTICES={base_config.max_justices}",
        "",
        "# Attention Mechanism",
        f"USE_JUSTICE_ATTENTION={str(best_params.get('use_justice_attention', base_config.use_justice_attention)).lower()}",
        f"NUM_ATTENTION_HEADS={best_params.get('num_attention_heads', base_config.num_attention_heads)}",
        "",
        "# Regularization",
        f"DROPOUT_RATE={best_params.get('dropout_rate', base_config.dropout_rate)}",
        f"WEIGHT_DECAY={best_params.get('weight_decay', base_config.weight_decay)}",
        "",
        "# Training Configuration",
        f"LEARNING_RATE={best_params.get('learning_rate', base_config.learning_rate)}",
        f"NUM_EPOCHS={base_config.num_epochs}",
        f"BATCH_SIZE={best_params.get('batch_size', base_config.batch_size)}",
        f"NUM_WORKERS={base_config.num_workers}",
        "",
        "# Loss Function",
        f"LOSS_FUNCTION={base_config.loss_function}",
        f"KL_REDUCTION={base_config.kl_reduction}",
        "",
        "# Other parameters (kept from base config)",
        f"PATIENCE={base_config.patience}",
        f"LR_SCHEDULER_FACTOR={base_config.lr_scheduler_factor}",
        f"LR_SCHEDULER_PATIENCE={base_config.lr_scheduler_patience}",
        f"MAX_GRAD_NORM={base_config.max_grad_norm}",
        f"DEVICE={base_config.device}",
        "",
        "# Data Paths",
        f"DATASET_FILE={base_config.dataset_file}",
        f"BIO_TOKENIZED_FILE={base_config.bio_tokenized_file}",
        f"DESCRIPTION_TOKENIZED_FILE={base_config.description_tokenized_file}",
        "",
        "# Output Paths",
        f"MODEL_OUTPUT_DIR={base_config.model_output_dir}",
        f"BEST_MODEL_NAME={base_config.best_model_name}",
        "",
        "# Dataset Splitting",
        f"TRAIN_RATIO={base_config.train_ratio}",
        f"VAL_RATIO={base_config.val_ratio}",
        f"TEST_RATIO={base_config.test_ratio}",
        f"SPLIT_RANDOM_STATE={base_config.split_random_state}",
        "",
        "# Validation and Evaluation",
        f"VALIDATION_FREQUENCY={base_config.validation_frequency}",
        f"EVALUATE_ON_TEST={str(base_config.evaluate_on_test).lower()}",
        "",
        "# Logging and Progress",
        f"VERBOSE_TRAINING={str(base_config.verbose_training).lower()}",
        f"LOG_FREQUENCY={base_config.log_frequency}",
        "",
        "# Memory Management",
        f"CLEAR_CACHE_ON_OOM={str(base_config.clear_cache_on_oom).lower()}",
        "",
        "# Model Saving",
        f"SAVE_CHECKPOINTS={str(base_config.save_checkpoints).lower()}",
        f"CHECKPOINT_FREQUENCY={base_config.checkpoint_frequency}",
        "",
        "# Cross-Attention Specific",
        f"USE_ATTENTION_FFN={str(base_config.use_attention_ffn).lower()}",
        f"ATTENTION_FFN_MULTIPLIER={base_config.attention_ffn_multiplier}",
        "",
        "# Advanced Training",
        f"USE_MIXED_PRECISION={str(base_config.use_mixed_precision).lower()}",
        f"GRADIENT_ACCUMULATION_STEPS={base_config.gradient_accumulation_steps}",
        "",
        "# Random Seeds",
        f"RANDOM_SEED={base_config.random_seed}",
        f"TORCH_SEED={base_config.torch_seed}",
        f"NUMPY_SEED={base_config.numpy_seed}",
        "",
        "# Model Loading and Caching",
        f"USE_MODEL_CACHE={str(base_config.use_model_cache).lower()}",
        f"DOWNLOAD_MODELS={str(base_config.download_models).lower()}",
        "",
        "# Data Validation",
        f"MIN_JUSTICES_PER_CASE={base_config.min_justices_per_case}",
        f"MAX_JUSTICES_PER_CASE={base_config.max_justices_per_case}",
        f"SKIP_MISSING_DESCRIPTIONS={str(base_config.skip_missing_descriptions).lower()}",
        f"SKIP_MISSING_BIOGRAPHIES={str(base_config.skip_missing_biographies).lower()}"
    ]
    
    # Write config file
    with open(output_file, 'w') as f:
        f.write('\n'.join(config_lines))
    
    logger.info(f" Best configuration saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    
    # Load config to get defaults
    base_config = ModelConfig()
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SCOTUS voting model")
    parser.add_argument("--n-trials", type=int, default=base_config.optuna_n_trials, 
                       help=f"Number of optimization trials (default: {base_config.optuna_n_trials})")
    parser.add_argument("--study-name", type=str, help="Name for the optimization study")
    parser.add_argument("--dataset-file", type=str, default=base_config.dataset_file,
                       help=f"Path to dataset file (default: {base_config.dataset_file})")
    parser.add_argument("--storage", type=str, help="Storage backend (e.g., sqlite:///study.db)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--output-config", type=str, default="optimized_config.env", 
                       help="Output file for best configuration")
    parser.add_argument("--dashboard", action="store_true", help="Start Optuna dashboard after optimization")
    
    args = parser.parse_args()
    
    # Run optimization
    study = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        dataset_file=args.dataset_file,
        storage=args.storage,
        n_jobs=args.n_jobs,
        timeout=args.timeout
    )
    
    # Save best configuration
    save_best_config(study, args.output_config)
    
    # Start dashboard if requested
    if args.dashboard and args.storage:
        print(f"\n🌐 Starting Optuna dashboard...")
        print(f"   Run: optuna-dashboard {args.storage}")
        print(f"   Then open: http://localhost:8080")


if __name__ == "__main__":
    main() 