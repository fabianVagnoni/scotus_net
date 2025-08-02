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

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from scripts.pretraining.constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from scripts.pretraining.loss import ContrastiveLoss
    from scripts.pretraining.config import ContrastiveJusticeConfig
    from scripts.utils.logger import get_logger
except ImportError:
    from constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from loss import ContrastiveLoss
    from config import ContrastiveJusticeConfig
    from ..utils.logger import get_logger


class HyperparameterTuner:
    def __init__(self, config_file: str = None):
        """Initialize the hyperparameter tuner."""
        self.logger = get_logger(__name__)
        self.base_config = ContrastiveJusticeConfig(config_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            year = int(data[0]) if data[0] is not None else 0
            justice_year_pairs.append((justice_name, year))
        
        # Sort by year and extract just the justice names
        ordered_justices = [justice for justice, year in sorted(justice_year_pairs, key=lambda x: x[1])]
        
        # Split the data
        self.test_justices = ordered_justices[:self.base_config.test_set_size]
        self.val_justices = ordered_justices[self.base_config.test_set_size:self.base_config.test_set_size + self.base_config.val_set_size]
        self.train_justices = ordered_justices[self.base_config.test_set_size + self.base_config.val_set_size:]
        
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
            for batch_idx, batch in enumerate(val_loader):
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
            correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based ranking
            
            # Add reciprocal rank
            reciprocal_ranks.append(1.0 / correct_rank)
        
        mrr = np.mean(reciprocal_ranks)
        return mrr
    
    def objective(self, trial):
        """Optuna objective function to maximize MRR."""
        # Sample hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        temperature = trial.suggest_float('temperature', 0.01, 1.0, log=True)
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        
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
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.base_config.num_workers
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.base_config.num_workers
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            loss_fn = ContrastiveLoss(temperature=temperature, alpha=alpha)
            loss_fn.to(self.device)
            
            # Reduced number of epochs for hyperparameter tuning
            num_epochs = min(5, self.base_config.num_epochs)  # Use fewer epochs for tuning
            
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
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
                        
                    except Exception as e:
                        self.logger.warning(f"Error in training batch: {e}")
                        continue
                
                if num_batches == 0:
                    return 0.0  # Return poor score if no successful batches
                
                # Report intermediate values for pruning
                if epoch < num_epochs - 1:  # Don't report on the last epoch
                    intermediate_mrr = self.calculate_mrr(model, val_loader)
                    trial.report(intermediate_mrr, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            
            # Calculate final MRR
            final_mrr = self.calculate_mrr(model, val_loader)
            self.logger.info(f"Trial {trial.number} final MRR: {final_mrr:.4f}")
            
            return final_mrr
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0  # Return poor score for failed trials
    
    def tune_hyperparameters(self, n_trials: int = 50, study_name: str = "contrastive_justice_tuning"):
        """Run hyperparameter tuning with Optuna."""
        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',  # We want to maximize MRR
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Print results
        self.logger.info("Hyperparameter tuning completed!")
        self.logger.info(f"Best MRR: {study.best_value:.4f}")
        self.logger.info("Best parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save results
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
    print("🔍 Starting Hyperparameter Tuning for Contrastive Justice Model")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Run tuning
    best_params, best_mrr = tuner.tune_hyperparameters(n_trials=50)
    
    print("\n🎉 Tuning Complete!")
    print(f"Best MRR: {best_mrr:.4f}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()