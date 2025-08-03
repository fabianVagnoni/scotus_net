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
except ImportError:
    from constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from loss import ContrastiveLoss
    from config import ContrastiveJusticeConfig
    from ..utils.logger import get_logger
    from ..utils.progress import get_progress_bar




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
        """Load and split the pretraining dataset."""
        with open(self.base_config.pretraining_dataset_file, 'r', encoding='utf-8') as f:
            pretraining_dataset = json.load(f)
        
        # Create a list of (justice_name, year) tuples and sort by year
        justice_year_pairs = []
        for justice_name, data in pretraining_dataset.items():
            year = int(data[0]) if data[0] is not None else 1
            justice_year_pairs.append((justice_name, year))
        
        # Sort by year (ascending) and extract just the justice names
        #self.logger.info(f"Justice year pairs:")
        #self.logger.info(justice_year_pairs)
        ordered_justices = [justice for justice, year in sorted(justice_year_pairs, key=lambda x: x[1])]
        
        # Split the data
        self.test_justices = ordered_justices[-self.base_config.test_set_size:]
        self.val_justices = ordered_justices[-self.base_config.test_set_size - self.base_config.val_set_size:-self.base_config.test_set_size]
        self.train_justices = ordered_justices[:-self.base_config.test_set_size - self.base_config.val_set_size]
        
        # Print val justices names
        self.logger.info(f"Val justices names:")
        self.logger.info(self.val_justices)
        # Print test justices
        self.logger.info(f"Test justices names:")
        self.logger.info(self.test_justices)
        self.logger.info(f"Data split - Train: {len(self.train_justices)}, Val: {len(self.val_justices)}, Test: {len(self.test_justices)}")
    
    def calculate_mrr(self, model: ContrastiveJustice, val_loader: DataLoader) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) for the validation set.
        
        For each truncated bio, we rank all full bio embeddings by similarity
        and find the rank of the correct match. MRR is the average of 1/rank.
        """
        model.eval()
        all_trunc_embeddings = []
        all_full_embeddings = []
        justice_indices = []
        
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
                    
                    # Track which examples these are (for correct matching)
                    batch_size = e_t.size(0)
                    start_idx = batch_idx * val_loader.batch_size
                    justice_indices.extend(range(start_idx, start_idx + batch_size))
                    
                except Exception as e:
                    self.logger.warning(f"Error in MRR batch {batch_idx}: {e}")
                    continue
        
        if not all_trunc_embeddings:
            return 0.0
        
        # Concatenate all embeddings
        # Ensure all tensors are on the same device before concatenation
        device = all_trunc_embeddings[0].device if all_trunc_embeddings else self.device
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
            # Create model with sampled hyperparameters
            model = ContrastiveJustice(
                trunc_bio_tokenized_file=self.base_config.trunc_bio_tokenized_file,
                full_bio_tokenized_file=self.base_config.full_bio_tokenized_file,
                model_name=self.base_config.model_name,
                dropout_rate=dropout_rate
            )
            model.to(self.device)
            
            # Create datasets and data loaders
            train_dataset = ContrastiveJusticeDataset(
                self.train_justices, 
                self.base_config.trunc_bio_tokenized_file, 
                self.base_config.full_bio_tokenized_file
            )
            val_dataset = ContrastiveJusticeDataset(
                self.val_justices, 
                self.base_config.trunc_bio_tokenized_file, 
                self.base_config.full_bio_tokenized_file
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(batch_size),
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0  # Set to 0 to avoid device issues with multiprocessing
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(batch_size),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0  # Set to 0 to avoid device issues with multiprocessing
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            loss_fn = ContrastiveLoss(temperature=temperature, alpha=alpha)
            loss_fn.to(self.device)
            
            # Use configurable number of epochs for hyperparameter tuning
            num_epochs = min(self.base_config.optuna_max_epochs, self.base_config.num_epochs)
            
            # Training loop with progress bar
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                # Progress bar for batches within each epoch
                batch_pbar = get_progress_bar(
                    train_loader, 
                    desc=f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs}",
                    total=len(train_loader)
                )
                
                for batch in batch_pbar:
                    try:
                        optimizer.zero_grad()
                        batch_trunc_bio_data = batch['trunc_bio_data']
                        batch_full_bio_data = batch['full_bio_data']
                        
                        e_t, e_f = model.forward(batch_trunc_bio_data, batch_full_bio_data)
                        batch_loss = loss_fn(e_t, e_f)
                        
                        batch_loss.backward()
                        optimizer.step()
                        
                        epoch_loss += batch_loss.item()
                        num_batches += 1
                        
                        # Update progress bar with current loss
                        batch_pbar.set_description(
                            f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} - Loss: {batch_loss.item():.4f}"
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"Error in training batch: {e}")
                        continue
                
                if num_batches == 0:
                    # Log failed trial
                    append_trial_results(trial, 0.0, trial_status="FAILED", experiment_name=self.experiment_name)
                    return 0.0  # Return poor score if no successful batches
                
                # Report intermediate values for pruning
                if epoch < num_epochs - 1:  # Don't report on the last epoch
                    intermediate_mrr = self.calculate_mrr(model, val_loader)
                    trial.report(intermediate_mrr, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        # Log pruned trial
                        append_trial_results(trial, intermediate_mrr, trial_status="PRUNED", experiment_name=self.experiment_name)
                        raise optuna.exceptions.TrialPruned()
            
            # Calculate final MRR
            final_mrr = self.calculate_mrr(model, val_loader)
            self.logger.info(f"Trial {trial.number} final MRR: {final_mrr:.4f}")
            
            # Log successful trial
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