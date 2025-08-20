#!/usr/bin/env python3
"""
Hyperparameter Tuning for Contrastive Justice Model using Optuna
================================================================

This script performs hyperparameter optimization using Optuna with Mean Reciprocal Rank (MRR) 
as the evaluation metric. It tunes batch size, weight decay, dropout, learning rate, 
temperature, and alpha parameters.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
import json
import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import gc

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from scripts.pretraining.constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from scripts.pretraining.loss import ContrastiveLoss
    from scripts.pretraining.config import ContrastiveJusticeConfig
    from scripts.utils.logger import get_logger
    from scripts.utils.progress import get_progress_bar
    from scripts.utils.metrics import calculate_mrr
except ImportError:
    from constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from loss import ContrastiveLoss
    from config import ContrastiveJusticeConfig
    from ..utils.logger import get_logger
    from ..utils.progress import get_progress_bar
    from ..utils.metrics import calculate_mrr




def initialize_results_file(study_name: str, n_trials: int, experiment_name: str = None):
    """
    Initialize the tuning results file with header information.
    
    Args:
        study_name: Name of the optimization study
        n_trials: Number of trials to run
        experiment_name: Name of the experiment (for file naming)
    """
    if experiment_name:
        results_file = Path(f"logs/hyperparameter_tunning_logs/pretraining_hy_tunning_{experiment_name}.txt")
    else:
        results_file = Path("logs/hyperparameter_tunning_logs/pretraining_hy_tunning_default.txt")
    
    # Create directory if it doesn't exist
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_lines = [
        f"{'#' * 80}",
        f"# SCOTUS AI Pretraining Hyperparameter Optimization Results",
        f"# Study: {study_name}",
        f"# Experiment: {experiment_name or 'default'}",
        f"# Started: {timestamp}",
        f"# Number of trials: {n_trials}",
        f"# Metric: Mean Reciprocal Rank (MRR) - Higher is better",
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


def append_trial_results(trial, mrr_score: float, hyperparams: Dict[str, any] = None, trial_status: str = "COMPLETED", experiment_name: str = None):
    """
    Append trial results to the tuning results file.
    
    Args:
        trial: Optuna trial object
        mrr_score: MRR score achieved
        hyperparams: Trial hyperparameters (optional, will be extracted from trial if not provided)
        trial_status: Status of the trial (COMPLETED, PRUNED, FAILED)
        experiment_name: Name of the experiment (for file naming)
    """
    # Create the results file path
    if experiment_name:
        results_file = Path(f"logs/hyperparameter_tunning_logs/pretraining_hy_tunning_{experiment_name}.txt")
    else:
        results_file = Path("logs/hyperparameter_tunning_logs/pretraining_hy_tunning_default.txt")
    
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
        f"Mean Reciprocal Rank (MRR): {mrr_score:.6f}",
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


def append_optimization_summary(study: optuna.Study, experiment_name: str = None):
    """
    Append optimization summary to the results file.
    
    Args:
        study: Completed Optuna study
        experiment_name: Name of the experiment (for file naming)
    """
    if experiment_name:
        results_file = Path(f"logs/hyperparameter_tunning_logs/pretraining_hy_tunning_{experiment_name}.txt")
    else:
        results_file = Path("logs/hyperparameter_tunning_logs/pretraining_hy_tunning_default.txt")
    
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
        f"Best MRR Score: {study.best_value:.6f}",
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


class HyperparameterTuner:
    def __init__(self, config_file: str = None, experiment_name: str = None):
        """Initialize the hyperparameter tuner."""
        self.logger = get_logger(__name__)
        self.base_config = ContrastiveJusticeConfig(config_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        self.logger.info(f"Using device: {self.device}")
        
        # Load and split the dataset once
        self._prepare_data()
    
    def _prepare_data(self):
        """Load dataset and build time-ordered CV folds (and a final test holdout)."""
        with open(self.base_config.pretraining_dataset_file, 'r', encoding='utf-8') as f:
            pretraining_dataset = json.load(f)

        # Build (justice_id, year) list; default missing years to 0 to keep them earliest
        justice_year_pairs = []
        self.justice_year_map = {}
        for justice_id, data in pretraining_dataset.items():
            try:
                year = int(data[0]) if data[0] is not None else 0
            except Exception:
                year = 0
            justice_year_pairs.append((justice_id, year))
            self.justice_year_map[justice_id] = year

        # Ascending by year gives natural time order compatible with encode_pretraining.py
        self.ordered_justices = [j for j, _ in sorted(justice_year_pairs, key=lambda x: x[1])]

        # Reserve the most recent justices as test holdout
        test_size = int(self.base_config.test_set_size)
        self.test_justices = self.ordered_justices[-test_size:] if test_size > 0 else []

        # Time-based CV folds are carved from the remaining prefix before the test window
        usable_end = len(self.ordered_justices) - len(self.test_justices)
        train_size = int(getattr(self.base_config, 'time_based_cv_train_size', 0))
        val_size = int(getattr(self.base_config, 'time_based_cv_val_size', 0))
        n_folds = int(getattr(self.base_config, 'time_based_cv_folds', 1))

        self.cv_splits = []
        if getattr(self.base_config, 'use_time_based_cv', True) and train_size > 0 and val_size > 0:
            for fold_idx in range(n_folds):
                val_end = usable_end - fold_idx * val_size
                val_start = max(0, val_end - val_size)
                train_end = val_start
                train_start = max(0, train_end - train_size)

                if train_start >= train_end or val_start >= val_end:
                    break

                train_ids = self.ordered_justices[train_start:train_end]
                val_ids = self.ordered_justices[val_start:val_end]
                if len(train_ids) == 0 or len(val_ids) == 0:
                    break
                self.cv_splits.append((train_ids, val_ids))

        # Fallback to single split if CVTT disabled or insufficient data
        if not self.cv_splits:
            val_size_default = int(self.base_config.val_set_size)
            val_ids = self.ordered_justices[usable_end - val_size_default:usable_end]
            train_ids = self.ordered_justices[:usable_end - val_size_default]
            self.cv_splits = [(train_ids, val_ids)]

        self.logger.info(f"Built {len(self.cv_splits)} time-based folds; Test holdout: {len(self.test_justices)} justices")
        # Log explicit time ranges for each fold (printed once at setup)
        for idx, (train_ids, val_ids) in enumerate(self.cv_splits, start=1):
            train_years = [self.justice_year_map[j] for j in train_ids] if train_ids else []
            val_years = [self.justice_year_map[j] for j in val_ids] if val_ids else []
            if train_years and val_years:
                self.logger.info(
                    f"Fold {idx}: Train years [{min(train_years)}-{max(train_years)}] (n={len(train_ids)}), "
                    f"Val years [{min(val_years)}-{max(val_years)}] (n={len(val_ids)})"
                )
    
    def calculate_mrr(self, model: ContrastiveJustice, val_loader: DataLoader) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) for the validation set.
        
        For each truncated bio, we rank all full bio embeddings by similarity
        and find the rank of the correct match. MRR is the average of 1/rank.
        """
        model.eval()
        all_trunc_embeddings = []
        all_full_embeddings = []
        
        with torch.no_grad():
            # Progress bar for validation batches
            val_pbar = get_progress_bar(
                val_loader,
                desc="Calculating MRR",
                total=len(val_loader)
            )
            
            for batch_idx, batch in enumerate(val_pbar):
                try:
                    batch_trunc_bio_data = batch['trunc_bio_data']
                    batch_full_bio_data = batch['full_bio_data']
                    
                    # Get embeddings
                    e_t, e_f = model.forward(batch_trunc_bio_data, batch_full_bio_data)
                    
                    # Normalize embeddings for cosine similarity
                    e_t = F.normalize(e_t, dim=-1)
                    e_f = F.normalize(e_f, dim=-1)
                    
                    all_trunc_embeddings.append(e_t)
                    all_full_embeddings.append(e_f)
                    
                except Exception as e:
                    self.logger.warning(f"Error in MRR batch {batch_idx}: {e}")
                    continue
        
        # Use the centralized calculate_mrr function
        return calculate_mrr(all_trunc_embeddings, all_full_embeddings)
    
    def objective(self, trial):
        """Optuna objective function to maximize MRR."""
        # Sample hyperparameters using config
        if self.base_config.tune_batch_size:
            #print("batch_size options: ", self.base_config.optuna_batch_size_options)
            batch_size = trial.suggest_categorical('batch_size', self.base_config.optuna_batch_size_options)
        else:
            #print("batch_size options from config: ", self.base_config.batch_size)
            batch_size = self.base_config.batch_size
            
        if self.base_config.tune_weight_decay:
            #print("weight_decay options: ", self.base_config.optuna_weight_decay_range)
            wd_min, wd_max, wd_log = self.base_config.optuna_weight_decay_range
            wd_kwargs = {'log': bool(wd_log)} if wd_log else {}
            weight_decay = trial.suggest_float('weight_decay', float(wd_min), float(wd_max), **wd_kwargs)
        else:
            #print("weight_decay options from config: ", self.base_config.weight_decay)
            weight_decay = self.base_config.weight_decay
            
        if self.base_config.tune_dropout_rate:
            #print("dropout_rate options: ", self.base_config.optuna_dropout_rate_range)
            dropout_min, dropout_max, dropout_step = self.base_config.optuna_dropout_rate_range
            dropout_kwargs = {'step': float(dropout_step)} if dropout_step is not None else {}
            dropout_rate = trial.suggest_float('dropout_rate', float(dropout_min), float(dropout_max), **dropout_kwargs)
        else:
            #print("dropout_rate options from config: ", self.base_config.dropout_rate)
            dropout_rate = self.base_config.dropout_rate

        if self.base_config.tune_learning_rate:
            #print("learning_rate options: ", self.base_config.optuna_learning_rate_range)
            lr_min, lr_max, lr_log = self.base_config.optuna_learning_rate_range
            lr_kwargs = {'log': bool(lr_log)} if lr_log else {}
            learning_rate = trial.suggest_float('learning_rate', float(lr_min), float(lr_max), **lr_kwargs)
        else:
            #print("learning_rate options from config: ", self.base_config.learning_rate)
            learning_rate = self.base_config.learning_rate
            
        if self.base_config.tune_temperature:
            #print("temperature options: ", self.base_config.optuna_temperature_range)
            temp_min, temp_max, temp_log = self.base_config.optuna_temperature_range
            temp_kwargs = {'log': bool(temp_log)} if temp_log else {}
            temperature = trial.suggest_float('temperature', float(temp_min), float(temp_max), **temp_kwargs)
        else:
            #print("temperature options from config: ", self.base_config.temperature)
            temperature = self.base_config.temperature
            
        if self.base_config.tune_alpha:
            #print("alpha options: ", self.base_config.optuna_alpha_range)
            alpha_min, alpha_max, alpha_step = self.base_config.optuna_alpha_range
            alpha_kwargs = {'step': float(alpha_step)} if alpha_step is not None else {}
            alpha = trial.suggest_float('alpha', float(alpha_min), float(alpha_max), **alpha_kwargs)
        else:
            #print("alpha options from config: ", self.base_config.alpha)
            alpha = self.base_config.alpha
        
        self.logger.info(f"Trial {trial.number}: batch_size={batch_size}, weight_decay={weight_decay:.2e}, "
                        f"dropout_rate={dropout_rate:.3f}, learning_rate={learning_rate:.2e}, "
                        f"temperature={temperature:.3f}, alpha={alpha:.3f}")
        
        try:
            fold_scores = []
            num_epochs = min(self.base_config.optuna_max_epochs, self.base_config.num_epochs)

            # Iterate CV folds
            for fold_idx, (train_ids, val_ids) in enumerate(self.cv_splits, start=1):
                # Fresh model per fold
                model = ContrastiveJustice(
                    trunc_bio_tokenized_file=self.base_config.trunc_bio_tokenized_file,
                    full_bio_tokenized_file=self.base_config.full_bio_tokenized_file,
                    model_name=self.base_config.model_name,
                    dropout_rate=dropout_rate
                )
                model.to(self.device)

                # Data for this fold
                train_dataset = ContrastiveJusticeDataset(
                    train_ids,
                    self.base_config.trunc_bio_tokenized_file,
                    self.base_config.full_bio_tokenized_file
                )
                val_dataset = ContrastiveJusticeDataset(
                    val_ids,
                    self.base_config.trunc_bio_tokenized_file,
                    self.base_config.full_bio_tokenized_file
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=int(batch_size),
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=int(batch_size),
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0
                )

                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                loss_fn = ContrastiveLoss(temperature=temperature, alpha=alpha)
                loss_fn.to(self.device)

                # Train for this fold
                for epoch in range(num_epochs):
                    model.train()
                    batch_pbar = get_progress_bar(
                        train_loader,
                        desc=f"Trial {trial.number} - Fold {fold_idx} - Epoch {epoch+1}/{num_epochs}",
                        total=len(train_loader)
                    )
                    num_batches = 0
                    for batch in batch_pbar:
                        try:
                            optimizer.zero_grad()
                            e_t, e_f = model.forward(batch['trunc_bio_data'], batch['full_bio_data'])
                            batch_loss = loss_fn(e_t, e_f)
                            batch_loss.backward()
                            optimizer.step()
                            num_batches += 1
                            batch_pbar.set_description(
                                f"Trial {trial.number} - Fold {fold_idx} - Epoch {epoch+1}/{num_epochs} - Loss: {batch_loss.item():.4f}"
                            )
                        except Exception as e:
                            self.logger.warning(f"Error in training batch: {e}")
                            continue

                    if num_batches == 0:
                        append_trial_results(trial, 0.0, trial_status="FAILED", experiment_name=self.experiment_name)
                        return 0.0

                    # Optional pruning based on interim MRR of first fold to keep simple
                    if fold_idx == 1 and epoch < num_epochs - 1:
                        intermediate_mrr = self.calculate_mrr(model, val_loader)
                        trial.report(intermediate_mrr, epoch)
                        if trial.should_prune():
                            append_trial_results(trial, intermediate_mrr, trial_status="PRUNED", experiment_name=self.experiment_name)
                            raise optuna.exceptions.TrialPruned()

                # Final fold score
                fold_mrr = self.calculate_mrr(model, val_loader)
                fold_scores.append(fold_mrr)
                self.logger.info(f"Trial {trial.number} - Fold {fold_idx} MRR: {fold_mrr:.4f}")

                # Cleanup to free memory between folds
                try:
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del optimizer
                    del loss_fn
                    del train_loader
                    del val_loader
                    del train_dataset
                    del val_dataset
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    self.logger.debug(f"Cleanup warning (fold {fold_idx}): {cleanup_err}")

            # Average across folds
            final_mrr = float(sum(fold_scores) / len(fold_scores)) if fold_scores else 0.0
            self.logger.info(f"Trial {trial.number} - Average CV MRR: {final_mrr:.4f}")

            append_trial_results(trial, final_mrr, trial_status="COMPLETED", experiment_name=self.experiment_name)
            return final_mrr
            
        except optuna.exceptions.TrialPruned:
            # Re-raise pruned trials
            raise
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {e}")
            # Log failed trial
            append_trial_results(trial, 0.0, trial_status="FAILED", experiment_name=self.experiment_name)
            return 0.0  # Return poor score for failed trials
    
    def tune_hyperparameters(self, n_trials: int = None, study_name: str = "contrastive_justice_tuning"):
        """Run hyperparameter tuning with Optuna."""
        if n_trials is None:
            n_trials = self.base_config.optuna_n_trials
        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Initialize results file
        initialize_results_file(study_name, n_trials, self.experiment_name)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',  # We want to maximize MRR
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.base_config.optuna_pruner_startup_trials, 
                n_warmup_steps=self.base_config.optuna_pruner_warmup_steps
            )
        )
        
        # Run optimization with progress bar
        print(f"🔍 Starting hyperparameter optimization with {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Print results
        self.logger.info("Hyperparameter tuning completed!")
        self.logger.info(f"Best MRR: {study.best_value:.4f}")
        self.logger.info("Best parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Append final summary to results file
        append_optimization_summary(study, self.experiment_name)
        
        # Save results (keeping existing functionality)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        with open(results_dir / "best_hyperparameters.json", 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save study statistics
        study_stats = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "study_name": study_name
        }
        
        with open(results_dir / "tuning_results.json", 'w') as f:
            json.dump(study_stats, f, indent=2)
        
        self.logger.info(f"Results saved to {results_dir}/")
        
        return study.best_params, study.best_value


def main():
    """Main function to run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Contrastive Justice pretraining model")
    parser.add_argument("--experiment-name", type=str, required=True, 
                       help="Name for the experiment (used in output filenames)")
    parser.add_argument("--n-trials", type=int, default=None, 
                       help="Number of optimization trials (default: from config)")
    parser.add_argument("--study-name", type=str, help="Name for the optimization study")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set default study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"contrastive_justice_tuning_{args.experiment_name}_{timestamp}"
    
    print("🔍 Starting Hyperparameter Tuning for Contrastive Justice Model")
    print("=" * 60)
    print(f"📊 Experiment name: {args.experiment_name}")
    print(f"🧪 Study name: {args.study_name}")
    print(f"🎯 Number of trials: {args.n_trials}")
    print("🚀 Progress bars will show:")
    print("   - Overall trial progress (Optuna)")
    print("   - Per-trial epoch progress")
    print("   - Per-epoch batch progress with loss")
    print("   - MRR calculation progress")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner(config_file=args.config_file, experiment_name=args.experiment_name)
    
    # Run tuning
    best_params, best_mrr = tuner.tune_hyperparameters(n_trials=args.n_trials, study_name=args.study_name)
    
    print("\n🎉 Tuning Complete!")
    print(f"Best MRR: {best_mrr:.4f}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Print summary with experiment name
    print(f"\n🎯 Experiment '{args.experiment_name}' completed!")
    print(f"📊 Results saved to: logs/hyperparameter_tunning_logs/pretraining_hy_tunning_{args.experiment_name}.txt")


if __name__ == "__main__":
    main()