#!/usr/bin/env python3
"""
Simplified Hyperparameter Optimization for SCOTUS Voting Model using Optuna.

This simplified version performs hyperparameter tuning for basic model architecture and training parameters
without complex unfreezing strategies.
"""

import warnings
import os
# Suppress specific warnings for cleaner output during hyperparameter optimization
def suppress_warnings():
    """Suppress common warnings that don't affect functionality during optimization."""
    
    # 1. Suppress FutureWarnings about torch.cuda.amp.GradScaler and autocast
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
    
    # 2. Suppress UserWarnings about gradient requirements
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True.*")
    
    # 3. Suppress RoBERTa encoder attention mask deprecation warnings
    warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*deprecated.*")
    
    # 4. Suppress dimension mismatch warnings (if they're expected in your case)
    warnings.filterwarnings("ignore", message=".*model embedding dimension.*doesn't match expected.*")
    
    # 5. Alternative: Set environment variable to reduce HuggingFace warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # 6. Suppress all FutureWarnings (more aggressive approach)
    # warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 7. Suppress all UserWarnings (more aggressive approach)
    # warnings.filterwarnings("ignore", category=UserWarning)
suppress_warnings()

import gc
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

# (deprecated):
# from torch.cuda.amp import autocast, GradScaler
# self.scaler = GradScaler()
# with autocast():
# (recommended):
from torch.amp import autocast, GradScaler

from scripts.models.model_trainer import SCOTUSModelTrainer
from scripts.models.scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
from torch.utils.data import DataLoader
from scripts.models.config import ModelConfig
from scripts.utils.logger import get_logger
from scripts.utils.holdout_test_set import HoldoutTestSetManager, TimeBasedCrossValidator
from scripts.models.losses import create_scotus_loss_function
from scripts.utils.metrics import calculate_f1_macro


class OptunaModelTrainer(SCOTUSModelTrainer):
    """
    Simplified model trainer for Optuna optimization.
    """
    
    def __init__(self, trial: Trial, base_config: ModelConfig, prepared_data: Dict = None):
        """Initialize trainer with trial-specific configuration and pre-loaded data."""
        super().__init__()
        self.trial = trial
        self.base_config = base_config
        self.best_val_loss = float('inf')
        self.early_stop_patience = base_config.optuna_early_stop_patience
        self.min_epochs = base_config.optuna_min_epochs
        # Override holdout manager to ensure we exclude holdout cases
        self.holdout_manager = HoldoutTestSetManager()
        # Store pre-loaded and split data
        self.prepared_data = prepared_data
    
    def _clear_memory_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _enable_gradient_checkpointing(self, model):
        """Enable gradient checkpointing for sentence transformers to save memory."""
        try:
            # Enable gradient checkpointing for bio model if it has the method
            if hasattr(model.bio_model, 'gradient_checkpointing_enable'):
                model.bio_model.gradient_checkpointing_enable()
            elif hasattr(model.bio_model, '_modules') and hasattr(model.bio_model._modules.get('0'), 'gradient_checkpointing_enable'):
                # For SentenceTransformer models, try to access the underlying transformer
                model.bio_model._modules['0'].gradient_checkpointing_enable()
            
            # Enable gradient checkpointing for description model if it has the method
            if hasattr(model.description_model, 'gradient_checkpointing_enable'):
                model.description_model.gradient_checkpointing_enable()
            elif hasattr(model.description_model, '_modules') and hasattr(model.description_model._modules.get('0'), 'gradient_checkpointing_enable'):
                # For SentenceTransformer models, try to access the underlying transformer
                model.description_model._modules['0'].gradient_checkpointing_enable()
                
        except Exception as e:
            # Gradient checkpointing might not be available for all models, so just log and continue
            self.logger.debug(f"Could not enable gradient checkpointing: {e}")
        
    def train_model_for_optimization(self) -> float:
        """
        Train model with trial-specific hyperparameters using pre-loaded data.
        
        Returns:
            Average combined metric across all CV folds: (KL Divergence Loss + (1 - F1-Score Macro)) / 2
        """
        start_time = time.time()
        max_trial_time = self.base_config.optuna_max_trial_time
        
        try:
            if not self.prepared_data:
                raise ValueError("No prepared data available. Data should be pre-loaded.")
            
            self.logger.info("Using pre-loaded data for optimization...")
            
            # Suggest hyperparameters
            hyperparams = self._suggest_hyperparameters()
            self.logger.info(f"Trial hyperparameters: {hyperparams}")
            
            # Use pre-loaded CV splits or single split
            if 'cv_splits' in self.prepared_data:
                cv_splits = self.prepared_data['cv_splits']
                bio_data = self.prepared_data['bio_data']
                description_data = self.prepared_data['description_data']
                
                # Train and evaluate on each fold
                fold_metrics = []
                
                # Progress bar for CV folds
                cv_progress = tqdm(
                    enumerate(cv_splits), 
                    total=len(cv_splits),
                    desc="CV Folds",
                    leave=True
                )
                
                for fold_idx, (train_dataset, val_dataset) in cv_progress:
                    # Check trial timeout
                    if time.time() - start_time > max_trial_time:
                        self.logger.warning(f"Trial timeout reached ({max_trial_time}s). Stopping at fold {fold_idx + 1}.")
                        break
                    
                    self.logger.info(f"🔄 Training fold {fold_idx + 1}/{len(cv_splits)}")
                    
                    # Train model for this fold
                    fold_metric = self._train_single_fold(
                        train_dataset, val_dataset, bio_data, description_data, 
                        hyperparams, fold_idx + 1, start_time, max_trial_time
                    )
                    
                    fold_metrics.append(fold_metric)
                    self.logger.info(f"   Fold {fold_idx + 1} metric: {fold_metric:.4f}")
                    
                    # Clear memory after each fold
                    self._clear_memory_cache()
                    
                    # Update CV progress bar with current fold metric
                    cv_progress.set_postfix({'fold_metric': f'{fold_metric:.4f}'})
                
                # Return average metric across folds
                if fold_metrics:
                    avg_metric = sum(fold_metrics) / len(fold_metrics)
                    self.logger.info(f"✅ Average metric across {len(fold_metrics)} folds: {avg_metric:.4f}")
                    return avg_metric
                else:
                    return 10.0  # High penalty if no folds completed
                    
            else:
                # Single train/val split
                train_dataset = self.prepared_data['train_dataset']
                val_dataset = self.prepared_data['val_dataset']
                bio_data = self.prepared_data['bio_data']
                description_data = self.prepared_data['description_data']
                
                return self._train_single_fold(
                    train_dataset, val_dataset, bio_data, description_data, 
                    hyperparams, 1, start_time, max_trial_time
                )
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"❌ Trial failed with error: {str(e)}")
            # Return a high penalty value for failed trials
            return 10.0
        finally:
            # Always clear memory at the end of each trial
            self._clear_memory_cache()
    
    def _train_single_fold(self, train_dataset: Dict, val_dataset: Dict, bio_data: Dict, description_data: Dict, 
                          hyperparams: Dict, fold_num: int, start_time: float, max_trial_time: float) -> float:
        """
        Train and evaluate a single fold.
        
        Returns:
            Combined metric for this fold: (KL Divergence Loss + (1 - F1-Score Macro)) / 2
        """
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
        
        # Enable gradient checkpointing for memory efficiency
        self._enable_gradient_checkpointing(model)
        
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
        
        self.logger.info(f"🚀 Starting fold {fold_num} training...")
        self.logger.info(f"   Training samples: {len(train_processed)}")
        self.logger.info(f"   Validation samples: {len(val_processed)}")
        
        # Training loop
        num_epochs = self.base_config.optuna_max_epochs
        patience_counter = 0
        best_fold_metric = float('inf')
        
        for epoch in range(num_epochs):
            # Check trial timeout
            if time.time() - start_time > max_trial_time:
                self.logger.warning(f"Trial timeout reached ({max_trial_time}s). Stopping fold {fold_num} early.")
                break
            
            # Training phase
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0
            
            # Training progress bar
            train_progress = tqdm(
                train_loader, 
                desc=f"Fold {fold_num} Epoch {epoch+1}/{num_epochs}",
                leave=False,
                disable=False
            )
            
            for batch in train_progress:
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
                
                with autocast(device_type=self.device):
                    # Forward pass
                    predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                    
                    # Compute loss
                    loss = criterion(predictions, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams['max_grad_norm'])
                
                # Optimizer steps
                self.scaler.step(main_optimizer)
                if sentence_transformer_optimizer:
                    self.scaler.step(sentence_transformer_optimizer)
                self.scaler.update()
                
                total_train_loss += loss.item()
                num_train_batches += 1
                
                # Update progress bar with current loss
                train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear cache periodically during training to prevent memory buildup
                if num_train_batches % 10 == 0:
                    self._clear_memory_cache()
            
            # Validation phase
            val_loss, val_f1 = self._evaluate_model_for_optimization(model, val_loader, criterion)
            
            # Clear memory after validation
            self._clear_memory_cache()
            
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            
            # Combined metric for optimization
            if epoch % 2 == 0:
                combined_metric = (val_loss + (1 - val_f1)) / 2
            
                self.logger.info(f"Fold {fold_num} Epoch {epoch+1}/{num_epochs} - "
                            f"Train Loss: {avg_train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Val F1: {val_f1:.4f}, "
                            f"Combined Metric: {combined_metric:.4f}")
                
            # Handle unfreezing at specified epoch
            if epoch + 1 == hyperparams.get('unfreeze_at_epoch', float('inf')):
                self.logger.info(f"🔓 Unfreezing sentence transformers for fold {fold_num}...")
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
            if epoch % 2 == 0 and combined_metric < best_fold_metric:
                best_fold_metric = combined_metric
                patience_counter = 0
            elif epoch % 2 == 0:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience and epoch >= self.min_epochs:
                    self.logger.info(f"Early stopping triggered for fold {fold_num} after {epoch+1} epochs")
                    break
            
            # Report intermediate values to Optuna (only for first fold to avoid confusion)
            if fold_num == 1 and epoch % 2 == 0:
                self.trial.report(combined_metric, epoch)
                
                # Handle pruning based on the intermediate value
                if self.trial.should_prune():
                    self.logger.info(f"Trial pruned at fold {fold_num}, epoch {epoch+1}")
                    raise optuna.exceptions.TrialPruned()
        
        # Clean up model and free memory
        del model
        del criterion
        del main_optimizer
        if sentence_transformer_optimizer:
            del sentence_transformer_optimizer
        del train_loader
        del val_loader
        del train_dataset_obj
        del val_dataset_obj
        
        # Clear memory cache
        self._clear_memory_cache()
        
        self.logger.info(f"✅ Fold {fold_num} completed with metric: {best_fold_metric:.4f}")
        return best_fold_metric

    def _suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        hyperparams = {}
        
        # Model Architecture - Hidden Dimension
        if self.base_config.tune_hidden_dim:
            #self.logger.info(f"Tuning hidden dimension: {self.base_config.optuna_hidden_dim_options}")
            hyperparams['hidden_dim'] = self.trial.suggest_categorical('hidden_dim', self.base_config.optuna_hidden_dim_options)
        else:
            hyperparams['hidden_dim'] = self.base_config.hidden_dim
        
        # Regularization - Dropout Rate
        if self.base_config.tune_dropout_rate:
            #self.logger.info(f"Tuning dropout rate: {self.base_config.optuna_dropout_rate_range}")
            dropout_min, dropout_max, dropout_step = self.base_config.optuna_dropout_rate_range
            dropout_kwargs = {'step': dropout_step} if dropout_step is not None else {}
            hyperparams['dropout_rate'] = self.trial.suggest_float('dropout_rate', dropout_min, dropout_max, **dropout_kwargs)
        else:
            hyperparams['dropout_rate'] = self.base_config.dropout_rate
        
        # Attention Mechanism - Number of Heads
        if self.base_config.tune_num_attention_heads:
            #self.logger.info(f"Tuning number of attention heads: {self.base_config.optuna_attention_heads_options}")
            hyperparams['num_attention_heads'] = self.trial.suggest_categorical('num_attention_heads', self.base_config.optuna_attention_heads_options)
        else:
            #self.logger.info(f"Using default number of attention heads: {self.base_config.num_attention_heads}")
            hyperparams['num_attention_heads'] = self.base_config.num_attention_heads
        
        # Attention Mechanism - Use Justice Attention
        if self.base_config.tune_use_justice_attention:
            #self.logger.info(f"Tuning use justice attention: {self.base_config.optuna_justice_attention_options}")
            hyperparams['use_justice_attention'] = self.trial.suggest_categorical('use_justice_attention', self.base_config.optuna_justice_attention_options)
        else:
            #self.logger.info(f"Using default use justice attention: {self.base_config.use_justice_attention}")
            hyperparams['use_justice_attention'] = self.base_config.use_justice_attention
        
        # Training Parameters - Learning Rate
        if self.base_config.tune_learning_rate:
            #self.logger.info(f"Tuning learning rate: {self.base_config.optuna_learning_rate_range}")
            lr_min, lr_max, lr_log = self.base_config.optuna_learning_rate_range
            lr_kwargs = {'log': lr_log} if lr_log else {}
            hyperparams['learning_rate'] = self.trial.suggest_float('learning_rate', lr_min, lr_max, **lr_kwargs)
        else:
            hyperparams['learning_rate'] = self.base_config.learning_rate
        
        # Training Parameters - Batch Size
        if self.base_config.tune_batch_size:
            #self.logger.info(f"Tuning batch size: {self.base_config.optuna_batch_size_options}")
            hyperparams['batch_size'] = self.trial.suggest_categorical('batch_size', self.base_config.optuna_batch_size_options)
        else:
            hyperparams['batch_size'] = self.base_config.batch_size
        
        # Regularization - Weight Decay
        if self.base_config.tune_weight_decay:
            #self.logger.info(f"Tuning weight decay: {self.base_config.optuna_weight_decay_range}")
            wd_min, wd_max, wd_log = self.base_config.optuna_weight_decay_range
            wd_kwargs = {'log': wd_log} if wd_log else {}
            hyperparams['weight_decay'] = self.trial.suggest_float('weight_decay', wd_min, wd_max, **wd_kwargs)
        else:
            hyperparams['weight_decay'] = self.base_config.weight_decay

        # Regularization - Use NEFTune
        if self.base_config.tune_use_noise_reg:
            #self.logger.info(f"Tuning use noise regularization: {self.base_config.optuna_use_noise_reg_options}")
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
            #self.logger.info(f"Tuning unfreezing: {self.base_config.optuna_unfreeze_at_epoch_options}")
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
        
        # Training Parameters - Max Grad Norm
        if self.base_config.tune_max_grad_norm:
            #self.logger.info(f"Tuning max grad norm: {self.base_config.optuna_max_grad_norm_range}")
            max_grad_norm_min, max_grad_norm_max, max_grad_norm_log = self.base_config.optuna_max_grad_norm_range
            max_grad_norm_kwargs = {'log': max_grad_norm_log} if max_grad_norm_log else {}
            hyperparams['max_grad_norm'] = self.trial.suggest_float('max_grad_norm', max_grad_norm_min, max_grad_norm_max, **max_grad_norm_kwargs)
        else:
            hyperparams['max_grad_norm'] = self.base_config.max_grad_norm
        
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
            progress_bar = tqdm(data_loader, desc="Evaluating model", leave=False)
            for batch in progress_bar:
                # Move batch to device
                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                justice_input_ids = [j.to(self.device) for j in batch['justice_input_ids']]
                justice_attention_mask = [j.to(self.device) for j in batch['justice_attention_mask']]
                justice_counts = batch['justice_counts']
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                with autocast(device_type=self.device):
                    predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                
                    # Compute loss
                    loss = criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for F1 calculation
                # Convert to class predictions (highest probability)
                pred_classes = torch.argmax(predictions, dim=1)
                target_classes = torch.argmax(targets, dim=1)
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                all_predictions.extend(pred_classes.cpu().numpy())
                all_targets.extend(target_classes.cpu().numpy())
                
                # Clear cache periodically during evaluation
                if num_batches % 10 == 0:
                    self._clear_memory_cache()

        
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
        #"trial_end_time": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        #"duration_seconds": trial.duration.total_seconds() if trial.duration else None,
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


def prepare_data_for_optimization(base_config: ModelConfig, dataset_file: str) -> Dict:
    """
    Prepare and split data once for all optimization trials.
    
    Args:
        base_config: Base configuration
        dataset_file: Path to dataset file
        
    Returns:
        Dictionary containing prepared data splits and tokenized data
    """
    logger = get_logger(__name__)
    
    logger.info("📂 Loading and preparing data for optimization...")
    
    # Create a temporary trainer to use its methods
    temp_trainer = SCOTUSModelTrainer()
    
    # Load dataset
    dataset = temp_trainer.load_case_dataset(dataset_file)
    
    # Get tokenized file paths and load tokenized data
    bio_tokenized_file, description_tokenized_file = temp_trainer.get_tokenized_file_paths()
    bio_data, description_data = temp_trainer.load_tokenized_data(bio_tokenized_file, description_tokenized_file)
    
    # Create holdout manager
    holdout_manager = HoldoutTestSetManager()
    
    prepared_data = {
        'bio_data': bio_data,
        'description_data': description_data
    }
    
    # Create splits based on configuration
    if base_config.use_time_based_cv:
        logger.info("Creating time-based CV splits...")
        cv_validator = TimeBasedCrossValidator(
            n_folds=base_config.time_based_cv_folds,
            train_size=base_config.time_based_cv_train_size,
            val_size=base_config.time_based_cv_val_size
        )
        
        cv_splits = cv_validator.create_time_based_cv_splits(dataset, holdout_manager)
        
        if not cv_splits:
            raise ValueError("No CV splits could be created")
        
        prepared_data['cv_splits'] = cv_splits
        logger.info(f"✅ Created {len(cv_splits)} CV splits")
        
    else:
        logger.info("Creating single train/val split...")
        train_dataset, val_dataset = temp_trainer.split_dataset(
            dataset, 
            train_ratio=0.8, 
            val_ratio=0.2
        )
        
        prepared_data['train_dataset'] = train_dataset
        prepared_data['val_dataset'] = val_dataset
        logger.info("✅ Created single train/val split")
    
    return prepared_data


def objective(trial: Trial, base_config: ModelConfig, prepared_data: Dict, experiment_name: str = None) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        prepared_data: Pre-loaded and split data
        experiment_name: Optional experiment name for logging
        
    Returns:
        Combined metric to minimize
    """
    trainer = OptunaModelTrainer(trial, base_config, prepared_data)
    
    try:
        combined_metric = trainer.train_model_for_optimization()
        
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
    finally:
        # Clean up trainer and clear memory
        if 'trainer' in locals():
            trainer._clear_memory_cache()
            del trainer

def objective_with_progress(trial, config, prepared_data, experiment_name, trial_progress, study):
    result = objective(trial, config, prepared_data, experiment_name)
    trial_progress.update(1)
    try:
        metric = study.best_value
        best_str = f"{metric:.4f}"
    except ValueError:
        best_str = "N/A"
    trial_progress.set_postfix({'best_metric': best_str})
    return result

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
    dataset_file = config.dataset_file
    
    # Initialize results file
    results_file = initialize_results_file(study_name, n_trials, dataset_file, experiment_name)
    logger.info(f"📊 Results will be saved to: {results_file}")
    
    # Prepare data once for all trials
    prepared_data = prepare_data_for_optimization(config, dataset_file)
    
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
    
    # Run optimization with progress tracking
    logger.info("🔄 Starting optimization trials...")
    
    # Create a manual progress bar for trials if n_jobs == 1 (parallel jobs don't work well with custom progress bars)
    if n_jobs == 1:
        trial_progress = tqdm(total=n_trials, desc="Optuna Trials", leave=True)

        
        study.optimize(
            lambda trial: objective_with_progress(trial, config, prepared_data, experiment_name, trial_progress, study),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=False  # Disable built-in progress bar since we have our own
        )
        
        trial_progress.close()
    else:
        # For parallel jobs, use built-in progress bar
        study.optimize(
            lambda trial: objective(trial, config, prepared_data, experiment_name),
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