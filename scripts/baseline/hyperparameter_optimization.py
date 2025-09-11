#!/usr/bin/env python3
"""
Minimal Hyperparameter Optimization for the Baseline SCOTUS Model using Optuna.

Keeps parity with scripts/models/hyperparameter_optimization.py structure,
but only tunes a few essentials to preserve simplicity.
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*deprecated.*")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*deprecated.*")

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
from optuna.trial import Trial
import torch
from torch.utils.data import DataLoader

from .baseline_trainer import BaselineTrainer
from scripts.utils.holdout_test_set import HoldoutTestSetManager, TimeBasedCrossValidator


def _clear_memory_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def objective(trial: Trial, base_params: Dict[str, Any]) -> float:
    # Suggest minimal set of hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 384, 512, 768])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Optional: if prepared CV split is provided in base_params, use it
    if 'train_dataset' in base_params and 'val_dataset' in base_params:
        trainer = BaselineTrainer(
            dataset_file=base_params['dataset_file'],
            description_tokenized_file=base_params['description_tokenized_file'],
            description_model_name=base_params['description_model_name'],
            embedding_dim=base_params['embedding_dim'],
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=base_params['num_epochs'],
            patience=base_params['patience'],
            device=base_params['device'],
            train_dataset_json=base_params['train_dataset'],
            val_dataset_json=base_params['val_dataset'],
            year_start=2010,
            year_end=2016
        )
    else:
        trainer = BaselineTrainer(
            dataset_file=base_params['dataset_file'],
            description_tokenized_file=base_params['description_tokenized_file'],
            description_model_name=base_params['description_model_name'],
            embedding_dim=base_params['embedding_dim'],
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=base_params['num_epochs'],
            patience=base_params['patience'],
            device=base_params['device']
        )

    # Run training and return validation loss
    try:
        results = trainer.train()
        val_loss = float(results.get('best_val_loss', 10.0))
    finally:
        _clear_memory_cache()

    return val_loss


def run_hyperparameter_optimization(
    n_trials: int = 25,
    study_name: str = "baseline_opt",
    dataset_file: str = 'data/processed/case_dataset.json',
    description_tokenized_file: str = 'data/processed/encoded_descriptions.pkl',
    description_model_name: str = 'Stern5497/sbert-legal-xlm-roberta-base',
    embedding_dim: int = 384,
    num_epochs: int = 8,
    patience: int = 3,
    storage: Optional[str] = None,
    n_jobs: int = 1,
    timeout: Optional[int] = None,
    device: Optional[str] = None,
    use_time_based_cv: bool = True,
    cv_folds: int = 3,
    cv_train_size: int = 800,
    cv_val_size: int = 100
) -> optuna.Study:
    # Base parameters passed to each trial
    base_params = {
        'dataset_file': dataset_file,
        'description_tokenized_file': description_tokenized_file,
        'description_model_name': description_model_name,
        'embedding_dim': embedding_dim,
        'num_epochs': num_epochs,
        'patience': patience,
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Prepare CVTT splits excluding holdout
    if use_time_based_cv:
        from scripts.models.model_trainer import _range_str as _rs  # reuse util if needed
        holdout_manager = HoldoutTestSetManager(dataset_file=dataset_file)
        full_dataset = holdout_manager.load_dataset()
        available_dataset = holdout_manager.filter_dataset_exclude_holdout(full_dataset)
        cv_validator = TimeBasedCrossValidator(n_folds=cv_folds, train_size=cv_train_size, val_size=cv_val_size)
        cv_splits = cv_validator.create_time_based_cv_splits(available_dataset, holdout_manager)

        # Flatten CV into sequential trials by injecting split per trial via study user_attrs pattern
        # For simplicity, cycle through splits if n_trials > len(cv_splits)
        split_pool = cv_splits if cv_splits else []
    else:
        split_pool = []

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

    print(f"Starting baseline HPO: trials={n_trials}, dataset={dataset_file}")

    # Wrap objective to inject CV split per trial when available
    def _obj_with_cv(trial: Trial) -> float:
        if split_pool:
            idx = trial.number % len(split_pool)
            train_ds, val_ds = split_pool[idx]
            base_params['train_dataset'] = train_ds
            base_params['val_dataset'] = val_ds
        else:
            base_params.pop('train_dataset', None)
            base_params.pop('val_dataset', None)
        return objective(trial, base_params)

    study.optimize(
        _obj_with_cv,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    print("Done. Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value:  {study.best_value:.4f}")
    print(f"  params: {study.best_params}")
    return study


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Model Hyperparameter Optimization")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--study-name", type=str, default="baseline_opt")
    parser.add_argument("--dataset-file", type=str, default='data/processed/case_dataset.json')
    parser.add_argument("--desc-tokenized", type=str, default='data/processed/encoded_descriptions.pkl')
    parser.add_argument("--desc-model", type=str, default='Stern5497/sbert-legal-xlm-roberta-base')
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--storage", type=str)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()

    print(f"Starting baseline HPO: trials={args.n_trials}, dataset={args.dataset_file}")
    run_hyperparameter_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        dataset_file=args.dataset_file,
        description_tokenized_file=args.desc_tokenized,
        description_model_name=args.desc_model,
        embedding_dim=args.embedding_dim,
        num_epochs=args.epochs,
        patience=args.patience,
        storage=args.storage,
        n_jobs=args.n_jobs,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()


