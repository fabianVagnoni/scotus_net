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

from scripts.models.model_trainer import SCOTUSModelTrainer
from scripts.models.scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
from torch.utils.data import DataLoader
from scripts.models.config import ModelConfig
from scripts.utils.logger import get_logger
from scripts.utils.holdout_test_set import HoldoutTestSetManager


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
        self.early_stop_patience = 3  # Reduced for faster optimization
        self.min_epochs = 2  # Minimum epochs before early stopping
        # Override holdout manager to ensure we exclude holdout cases
        self.holdout_manager = HoldoutTestSetManager()
        
    def train_model_for_optimization(self, dataset_file: str = None) -> float:
        """
        Train model with trial-specific hyperparameters and return validation loss.
        
        Returns:
            Best validation loss achieved during training
        """
        try:
            # Load dataset
            dataset = self.load_case_dataset(dataset_file)
            
            # Get tokenized file paths
            bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths(dataset)
            
            # Split dataset (use smaller validation set for faster optimization)
            train_dataset, val_dataset = self.split_dataset(
                dataset, 
                train_ratio=0.8, 
                val_ratio=0.2
            )
            
            # Suggest hyperparameters
            hyperparams = self._suggest_hyperparameters()
            
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
            
            # Prepare datasets
            train_dataset_dict = self.prepare_dataset_dict(train_dataset)
            val_dataset_dict = self.prepare_dataset_dict(val_dataset)
            
            if not train_dataset_dict or not val_dataset_dict:
                raise ValueError("No valid training or validation cases found")
            
            # Limit dataset size for faster optimization
            max_train_samples = min(len(train_dataset_dict), 1000)  # Limit training samples
            max_val_samples = min(len(val_dataset_dict), 200)      # Limit validation samples
            
            train_keys = list(train_dataset_dict.keys())[:max_train_samples]
            val_keys = list(val_dataset_dict.keys())[:max_val_samples]
            
            train_subset = {k: train_dataset_dict[k] for k in train_keys}
            val_subset = {k: val_dataset_dict[k] for k in val_keys}
            
            train_pytorch_dataset = SCOTUSDataset(train_subset)
            val_pytorch_dataset = SCOTUSDataset(val_subset)
            
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
            
            # Training setup with trial hyperparameters
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=hyperparams['learning_rate'], 
                weight_decay=hyperparams['weight_decay']
            )
            
            # Setup loss function (fixed to KL Divergence for optimization)
            criterion = nn.KLDivLoss(reduction='batchmean')
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )
            
            # Training loop with early stopping
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            
            max_epochs = 10  # Fixed epochs for optimization (was hyperparams['num_epochs'])
            
            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        optimizer.zero_grad()
                        
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
                        
                        # Apply log_softmax for KL divergence loss
                        log_predictions = torch.log_softmax(predictions_tensor, dim=1)
                        loss = criterion(log_predictions, batch_targets)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error in training batch {batch_idx}: {e}")
                        continue
                
                if num_batches == 0:
                    raise ValueError("No successful training batches")
                    
                avg_train_loss = train_loss / num_batches
                
                # Validation phase
                val_loss = self._evaluate_model_for_optimization(model, val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Report intermediate value to Optuna
                self.trial.report(val_loss, epoch)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Early stopping
                if patience_counter >= self.early_stop_patience and epoch >= self.min_epochs:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            return best_val_loss
            
        except Exception as e:
            self.logger.error(f"Error in trial: {e}")
            return float('inf')
    
    def _suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        return {
            # Model Architecture
            'hidden_dim': self.trial.suggest_categorical('hidden_dim', [256, 512, 768, 1024]),
            'dropout_rate': self.trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'num_attention_heads': self.trial.suggest_categorical('num_attention_heads', [2, 4, 6, 8]),
            'use_justice_attention': self.trial.suggest_categorical('use_justice_attention', [True, False]),
            
            # Training Parameters
            'learning_rate': self.trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': self.trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'weight_decay': self.trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        }
    
    def _evaluate_model_for_optimization(self, model: SCOTUSVotingModel, data_loader, criterion) -> float:
        """Evaluate model for optimization (faster version)."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
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
                    
                    # Apply log_softmax for KL divergence loss
                    log_predictions = torch.log_softmax(predictions_tensor, dim=1)
                    loss = criterion(log_predictions, batch_targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error in evaluation batch: {e}")
                    continue
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')


def objective(trial: Trial, base_config: ModelConfig, dataset_file: str) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        dataset_file: Path to dataset file
        
    Returns:
        Validation loss to minimize
    """
    logger = get_logger(__name__)
    logger.info(f"Starting trial {trial.number}")
    
    try:
        # Create trainer for this trial
        trainer = OptunaModelTrainer(trial, base_config)
        
        # Train model and get validation loss
        val_loss = trainer.train_model_for_optimization(dataset_file)
        
        logger.info(f"Trial {trial.number} completed with validation loss: {val_loss:.4f}")
        return val_loss
        
    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return float('inf')


def run_hyperparameter_optimization(
    n_trials: int = 50,
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
    
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"scotus_optimization_{timestamp}"
    
    logger.info(f"🔍 Starting hyperparameter optimization")
    logger.info(f"   📊 Study name: {study_name}")
    logger.info(f"   🎯 Number of trials: {n_trials}")
    logger.info(f"   📁 Dataset: {dataset_file}")
    logger.info(f"   💾 Storage: {storage or 'in-memory'}")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # Minimize validation loss
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
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
    logger.info(f"   📈 Best validation loss: {study.best_value:.4f}")
    logger.info(f"   ⚙️  Best parameters:")
    
    for key, value in study.best_params.items():
        logger.info(f"      {key}: {value}")
    
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
        f"# Best validation loss: {study.best_value:.6f}",
        f"# Optimization date: {datetime.now().isoformat()}",
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
    
    logger.info(f"💾 Best configuration saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SCOTUS voting model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, help="Name for the optimization study")
    parser.add_argument("--dataset-file", type=str, help="Path to dataset file")
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