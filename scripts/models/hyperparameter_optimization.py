#!/usr/bin/env python3
"""
Simplified Hyperparameter Optimization for SCOTUS Voting Model using Optuna.

This simplified version performs hyperparameter tuning for basic model architecture and training parameters
without complex unfreezing strategies.
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
from scripts.utils.config import ModelConfig
from scripts.utils.logger import get_logger
from scripts.utils.holdout_test_set import HoldoutTestSetManager
from scripts.models.losses import create_scotus_loss_function
from scripts.utils.metrics import calculate_f1_macro


class OptunaModelTrainer(SCOTUSModelTrainer):
    """
    Simplified model trainer for Optuna optimization.
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
        """
        start_time = time.time()
        max_trial_time = self.base_config.optuna_max_trial_time
        
        try:
            self.logger.info("Loading dataset for optimization...")
            # Load dataset
            dataset = self.load_case_dataset(dataset_file)
            
            # Get tokenized file paths and load tokenized data
            bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths(dataset)
            bio_data, description_data = self.load_tokenized_data(bio_tokenized_file, description_tokenized_file)
            
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
            
            # Process datasets
            train_processed = self.prepare_processed_data(train_dataset, bio_data, description_data, verbose=False)
            val_processed = self.prepare_processed_data(val_dataset, bio_data, description_data, verbose=False)
            
            # Create datasets and data loaders
            train_dataset_obj = SCOTUSDataset(train_processed)
            val_dataset_obj = SCOTUSDataset(val_processed)
            
            train_loader = DataLoader(
                train_dataset_obj,
                batch_size=hyperparams['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers
            )
            
            val_loader = DataLoader(
                val_dataset_obj,
                batch_size=hyperparams['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers
            )
            
            # Initialize model with trial hyperparameters
            model = SCOTUSVotingModel(
                bio_model_name=self.base_config.bio_model_name,
                description_model_name=self.base_config.description_model_name,
                embedding_dim=self.base_config.embedding_dim,
                hidden_dim=hyperparams['hidden_dim'],
                dropout_rate=hyperparams['dropout_rate'],
                max_justices=self.base_config.max_justices,
                num_attention_heads=hyperparams['num_attention_heads'],
                use_justice_attention=hyperparams['use_justice_attention'],
                use_noise_reg=hyperparams['use_noise_reg'],
                noise_reg_alpha=hyperparams['noise_reg_alpha'],
                device=self.device
            )
            model.to(self.device)
            
            # Create loss function
            criterion = create_scotus_loss_function(self.base_config)
            
            # Create optimizer for non-transformer parameters
            main_optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=hyperparams['learning_rate'],
                weight_decay=hyperparams['weight_decay']
            )
            
            # Initialize sentence transformer optimizer (will be used if models are unfrozen)
            sentence_transformer_optimizer = None
            
            self.logger.info("🚀 Starting optimization training...")
            self.logger.info(f"   Training samples: {len(train_processed)}")
            self.logger.info(f"   Validation samples: {len(val_processed)}")
            
            # Training loop
            num_epochs = self.base_config.optuna_max_epochs
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Check trial timeout
                if time.time() - start_time > max_trial_time:
                    self.logger.warning(f"Trial timeout reached ({max_trial_time}s). Stopping early.")
                    break
                
                # Training phase
                model.train()
                total_train_loss = 0.0
                num_train_batches = 0
                
                for batch in train_loader:
                    main_optimizer.zero_grad()
                    if sentence_transformer_optimizer:
                        sentence_transformer_optimizer.zero_grad()
                    
                    # Move batch to device
                    case_input_ids = batch['case_input_ids'].to(self.device)
                    case_attention_mask = batch['case_attention_mask'].to(self.device)
                    justice_input_ids = [j.to(self.device) for j in batch['justice_input_ids']]
                    justice_attention_mask = [j.to(self.device) for j in batch['justice_attention_mask']]
                    justice_counts = batch['justice_counts']
                    targets = batch['targets'].to(self.device)
                    
                    # Forward pass
                    predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                    
                    # Compute loss
                    loss = criterion(predictions, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer steps
                    main_optimizer.step()
                    if sentence_transformer_optimizer:
                        sentence_transformer_optimizer.step()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                
                # Validation phase
                val_loss, val_f1 = self._evaluate_model_for_optimization(model, val_loader, criterion)
                
                avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
                
                # Combined metric for optimization
                combined_metric = (val_loss + (1 - val_f1)) / 2
                
                self.logger.info(f"Trial Epoch {epoch+1}/{num_epochs} - "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"Val F1: {val_f1:.4f}, "
                               f"Combined Metric: {combined_metric:.4f}")
                
                # Handle unfreezing at specified epoch
                if epoch + 1 == hyperparams.get('unfreeze_at_epoch', float('inf')):
                    self.logger.info("🔓 Unfreezing sentence transformers...")
                    if hyperparams.get('unfreeze_bio_model', False):
                        model.unfreeze_bio_model()
                        self.logger.info("   - Bio model unfrozen")
                    if hyperparams.get('unfreeze_description_model', False):
                        model.unfreeze_description_model()
                        self.logger.info("   - Description model unfrozen")
                    
                    # Create sentence transformer optimizer if models were unfrozen
                    if model.get_sentence_transformer_status()['any_trainable']:
                        st_params = []
                        if hyperparams.get('unfreeze_bio_model', False):
                            st_params.extend(model.bio_model.parameters())
                        if hyperparams.get('unfreeze_description_model', False):
                            st_params.extend(model.description_model.parameters())
                        
                        sentence_transformer_optimizer = torch.optim.Adam(
                            st_params,
                            lr=hyperparams.get('sentence_transformer_learning_rate', hyperparams['learning_rate'] * 0.1),
                            weight_decay=hyperparams['weight_decay']
                        )
                        self.logger.info(f"   - Sentence transformer optimizer created")
                
                # Early stopping based on combined metric
                if combined_metric < self.best_val_loss:
                    self.best_val_loss = combined_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience and epoch >= self.min_epochs:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                # Report intermediate values to Optuna
                self.trial.report(combined_metric, epoch)
                
                # Handle pruning based on the intermediate value
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.exceptions.TrialPruned()
            
            self.logger.info(f"✅ Trial completed with combined metric: {self.best_val_loss:.4f}")
            return self.best_val_loss
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"❌ Trial failed with error: {str(e)}")
            # Return a high penalty value for failed trials
            return 10.0

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

        # Regularization - Use NEFTune
        if self.base_config.tune_use_noise_reg:
            hyperparams['use_noise_reg'] = self.trial.suggest_categorical('use_noise_reg', self.base_config.optuna_use_noise_reg_options)
            # Only tune alpha if noise_reg is enabled
            if hyperparams['use_noise_reg']:
                alpha_min, alpha_max, alpha_log = self.base_config.optuna_noise_reg_alpha_range
                alpha_kwargs = {'log': alpha_log} if alpha_log else {}
                hyperparams['noise_reg_alpha'] = self.trial.suggest_float('noise_reg_alpha', alpha_min, alpha_max, **alpha_kwargs)
            else:
                hyperparams['noise_reg_alpha'] = self.base_config.noise_reg_alpha
        else:
            hyperparams['use_noise_reg'] = self.base_config.use_noise_reg
            hyperparams['noise_reg_alpha'] = self.base_config.noise_reg_alpha
        
        # Simplified unfreezing strategy (optional)
        if self.base_config.tune_unfreezing:
            hyperparams['unfreeze_at_epoch'] = self.trial.suggest_categorical('unfreeze_at_epoch', self.base_config.optuna_unfreeze_at_epoch_options)
            hyperparams['unfreeze_bio_model'] = self.trial.suggest_categorical('unfreeze_bio_model', [True, False])
            hyperparams['unfreeze_description_model'] = self.trial.suggest_categorical('unfreeze_description_model', [True, False])
            
            if hyperparams['unfreeze_bio_model'] or hyperparams['unfreeze_description_model']:
                st_lr_min, st_lr_max, st_lr_log = self.base_config.optuna_sentence_transformer_lr_range
                st_lr_kwargs = {'log': st_lr_log} if st_lr_log else {}
                hyperparams['sentence_transformer_learning_rate'] = self.trial.suggest_float('sentence_transformer_learning_rate', st_lr_min, st_lr_max, **st_lr_kwargs)
        else:
            hyperparams['unfreeze_at_epoch'] = self.base_config.unfreeze_at_epoch
            hyperparams['unfreeze_bio_model'] = self.base_config.unfreeze_bio_model
            hyperparams['unfreeze_description_model'] = self.base_config.unfreeze_description_model
            hyperparams['sentence_transformer_learning_rate'] = self.base_config.sentence_transformer_learning_rate
        
        return hyperparams

    def _evaluate_model_for_optimization(self, model: SCOTUSVotingModel, data_loader: DataLoader, criterion) -> tuple:
        """
        Evaluate model for optimization, returning both loss and F1 score.
        
        Returns:
            Tuple of (val_loss, val_f1_macro)
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                justice_input_ids = [j.to(self.device) for j in batch['justice_input_ids']]
                justice_attention_mask = [j.to(self.device) for j in batch['justice_attention_mask']]
                justice_counts = batch['justice_counts']
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                
                # Compute loss
                loss = criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for F1 calculation
                # Convert to class predictions (highest probability)
                pred_classes = torch.argmax(predictions, dim=1)
                target_classes = torch.argmax(targets, dim=1)
                
                all_predictions.extend(pred_classes.cpu().numpy())
                all_targets.extend(target_classes.cpu().numpy())
        
        val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        val_f1 = calculate_f1_macro(all_targets, all_predictions)
        
        return val_loss, val_f1


def initialize_results_file(study_name: str, n_trials: int, dataset_file: str, experiment_name: str = None):
    """Initialize the results CSV file with headers."""
    results_dir = Path("results/hyperparameter_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        filename = f"{experiment_name}_{study_name}_optimization_results_{timestamp}.csv"
    else:
        filename = f"{study_name}_optimization_results_{timestamp}.csv"
    
    results_file = results_dir / filename
    
    # Create initial CSV with headers
    headers = [
        "trial_number", "trial_status", "trial_start_time", "trial_end_time", "duration_seconds",
        "combined_metric", "val_loss", "val_f1_macro",
        "hidden_dim", "dropout_rate", "num_attention_heads", "use_justice_attention",
        "learning_rate", "batch_size", "weight_decay", "use_noise_reg", "noise_reg_alpha",
        "unfreeze_at_epoch", "unfreeze_bio_model", "unfreeze_description_model", "sentence_transformer_learning_rate",
        "dataset_file", "experiment_name"
    ]
    
    df = pd.DataFrame(columns=headers)
    df.to_csv(results_file, index=False)
    
    return str(results_file)


def append_trial_results(trial: Trial, combined_metric: float, hyperparams: Dict[str, Any] = None, trial_status: str = "COMPLETED", experiment_name: str = None):
    """Append trial results to the CSV file."""
    results_dir = Path("results/hyperparameter_optimization")
    
    # Find the most recent results file
    pattern = f"*_optimization_results_*.csv"
    if experiment_name:
        pattern = f"{experiment_name}_*_optimization_results_*.csv"
    
    results_files = sorted(results_dir.glob(pattern))
    if not results_files:
        return
    
    results_file = results_files[-1]
    
    # Extract values from trial
    trial_data = {
        "trial_number": trial.number,
        "trial_status": trial_status,
        "trial_start_time": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "trial_end_time": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "combined_metric": combined_metric,
        "val_loss": trial.user_attrs.get("val_loss", None),
        "val_f1_macro": trial.user_attrs.get("val_f1_macro", None)
    }
    
    # Add hyperparameters
    if hyperparams:
        for key, value in hyperparams.items():
            trial_data[key] = value
    else:
        # Extract from trial params
        for key in trial.params:
            trial_data[key] = trial.params[key]
    
    trial_data["experiment_name"] = experiment_name
    
    # Append to CSV
    df = pd.DataFrame([trial_data])
    df.to_csv(results_file, mode='a', header=False, index=False)


def objective(trial: Trial, base_config: ModelConfig, dataset_file: str, experiment_name: str = None) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        dataset_file: Path to dataset file
        experiment_name: Optional experiment name for logging
        
    Returns:
        Combined metric to minimize
    """
    trainer = OptunaModelTrainer(trial, base_config)
    
    try:
        combined_metric = trainer.train_model_for_optimization(dataset_file)
        
        # Store additional metrics in trial user attributes
        trial.set_user_attr("val_loss", trainer.best_val_loss)
        
        # Log trial results
        hyperparams = trainer._suggest_hyperparameters()
        append_trial_results(trial, combined_metric, hyperparams, "COMPLETED", experiment_name)
        
        return combined_metric
        
    except optuna.exceptions.TrialPruned:
        append_trial_results(trial, float('inf'), None, "PRUNED", experiment_name)
        raise
    except Exception as e:
        trainer.logger.error(f"Trial {trial.number} failed: {str(e)}")
        append_trial_results(trial, float('inf'), None, "FAILED", experiment_name)
        return 10.0  # High penalty for failed trials


def run_hyperparameter_optimization(
    n_trials: int = None,
    study_name: str = None,
    dataset_file: str = None,
    storage: str = None,
    n_jobs: int = 1,
    timeout: Optional[int] = None,
    experiment_name: str = None
) -> optuna.Study:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
        dataset_file: Path to dataset file
        storage: Storage URL for study persistence
        n_jobs: Number of parallel jobs
        timeout: Timeout in seconds
        experiment_name: Optional experiment name
        
    Returns:
        Completed Optuna study
    """
    logger = get_logger(__name__)
    config = ModelConfig()
    
    # Use config defaults if not provided
    n_trials = n_trials or config.optuna_n_trials
    study_name = study_name or config.optuna_study_name
    dataset_file = dataset_file or config.case_dataset_file
    
    # Initialize results file
    results_file = initialize_results_file(study_name, n_trials, dataset_file, experiment_name)
    logger.info(f"📊 Results will be saved to: {results_file}")
    
    # Create study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    logger.info(f"🎯 Starting hyperparameter optimization...")
    logger.info(f"   Study name: {study_name}")
    logger.info(f"   Dataset: {dataset_file}")
    logger.info(f"   Number of trials: {n_trials}")
    logger.info(f"   Parallel jobs: {n_jobs}")
    logger.info(f"   Timeout: {timeout}s" if timeout else "   Timeout: None")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, dataset_file, experiment_name),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Log results
    logger.info("🏆 Optimization completed!")
    logger.info(f"   Best trial: {study.best_trial.number}")
    logger.info(f"   Best combined metric: {study.best_value:.4f}")
    logger.info(f"   Best parameters: {study.best_params}")
    
    return study


def save_best_config(study: optuna.Study, base_config: ModelConfig, output_file: str = "best_config.env"):
    """Save the best configuration to a file."""
    best_params = study.best_params
    
    config_lines = [
        "# Best hyperparameters from Optuna optimization",
        f"# Best combined metric: {study.best_value:.4f}",
        f"# Trial number: {study.best_trial.number}",
        "",
        "# Model Architecture",
        f"HIDDEN_DIM={best_params.get('hidden_dim', base_config.hidden_dim)}",
        f"DROPOUT_RATE={best_params.get('dropout_rate', base_config.dropout_rate)}",
        f"NUM_ATTENTION_HEADS={best_params.get('num_attention_heads', base_config.num_attention_heads)}",
        f"USE_JUSTICE_ATTENTION={best_params.get('use_justice_attention', base_config.use_justice_attention)}",
        "",
        "# Training Parameters",
        f"LEARNING_RATE={best_params.get('learning_rate', base_config.learning_rate)}",
        f"BATCH_SIZE={best_params.get('batch_size', base_config.batch_size)}",
        f"WEIGHT_DECAY={best_params.get('weight_decay', base_config.weight_decay)}",
        "",
        "# Regularization",
        f"USE_NOISE_REG={best_params.get('use_noise_reg', base_config.use_noise_reg)}",
        f"NOISE_REG_ALPHA={best_params.get('noise_reg_alpha', base_config.noise_reg_alpha)}",
        "",
        "# Unfreezing Strategy",
        f"UNFREEZE_AT_EPOCH={best_params.get('unfreeze_at_epoch', base_config.unfreeze_at_epoch)}",
        f"UNFREEZE_BIO_MODEL={best_params.get('unfreeze_bio_model', base_config.unfreeze_bio_model)}",
        f"UNFREEZE_DESCRIPTION_MODEL={best_params.get('unfreeze_description_model', base_config.unfreeze_description_model)}",
        f"SENTENCE_TRANSFORMER_LEARNING_RATE={best_params.get('sentence_transformer_learning_rate', base_config.sentence_transformer_learning_rate)}"
    ]
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(config_lines))
    
    print(f"💾 Best configuration saved to: {output_file}")


def main():
    """Main function for running hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="SCOTUS Model Hyperparameter Optimization")
    parser.add_argument("--n-trials", type=int, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, help="Name of the optimization study")
    parser.add_argument("--dataset-file", type=str, help="Path to the dataset file")
    parser.add_argument("--storage", type=str, help="Storage URL for study persistence")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--experiment-name", type=str, help="Optional experiment name for logging")
    parser.add_argument("--save-best-config", type=str, help="Save best config to this file")
    
    args = parser.parse_args()
    
    # Run optimization
    study = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        dataset_file=args.dataset_file,
        storage=args.storage,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        experiment_name=args.experiment_name
    )
    
    # Save best config if requested
    if args.save_best_config:
        config = ModelConfig()
        save_best_config(study, config, args.save_best_config)


if __name__ == "__main__":
    main()