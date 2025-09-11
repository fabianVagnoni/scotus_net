#!/usr/bin/env python3
"""
Baseline configuration loader for SCOTUS AI baseline components.

Loads configuration from config.env and exposes typed attributes.
Kept minimal to avoid overengineering.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class BaselineConfig:
    def __init__(self, config_file: str = None):
        if config_file is None:
            config_file = Path(__file__).parent / "config.env"
        load_dotenv(config_file, override=True)
        self.config_file = str(config_file)
        self._load()

    def _load(self) -> None:
        # Model
        self.description_model_name = os.getenv('BASELINE_DESCRIPTION_MODEL_NAME', 'Stern5497/sbert-legal-xlm-roberta-base')
        self.embedding_dim = int(os.getenv('BASELINE_EMBEDDING_DIM', '384'))
        self.hidden_dim = int(os.getenv('BASELINE_HIDDEN_DIM', '512'))
        self.dropout_rate = float(os.getenv('BASELINE_DROPOUT_RATE', '0.1'))

        # Training
        self.learning_rate = float(os.getenv('BASELINE_LEARNING_RATE', '1e-4'))
        self.weight_decay = float(os.getenv('BASELINE_WEIGHT_DECAY', '1e-2'))
        self.num_epochs = int(os.getenv('BASELINE_NUM_EPOCHS', '10'))
        self.batch_size = int(os.getenv('BASELINE_BATCH_SIZE', '16'))
        self.num_workers = int(os.getenv('BASELINE_NUM_WORKERS', '0'))
        self.patience = int(os.getenv('BASELINE_PATIENCE', '5'))

        # Device
        device = os.getenv('BASELINE_DEVICE', 'auto').lower()
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Data
        self.dataset_file = os.getenv('BASELINE_DATASET_FILE', 'data/processed/case_dataset.json')
        self.description_tokenized_file = os.getenv('BASELINE_DESCRIPTION_TOKENIZED_FILE', 'data/processed/encoded_descriptions.pkl')

        # CVTT / Holdout
        self.use_time_based_cv = os.getenv('BASELINE_USE_TIME_BASED_CV', 'true').lower() == 'true'
        self.cv_folds = int(os.getenv('BASELINE_CV_FOLDS', '3'))
        self.cv_train_size = int(os.getenv('BASELINE_CV_TRAIN_SIZE', '800'))
        self.cv_val_size = int(os.getenv('BASELINE_CV_VAL_SIZE', '100'))

        # Optuna
        self.optuna_n_trials = int(os.getenv('BASELINE_OPTUNA_N_TRIALS', '25'))
        self.optuna_study_name = os.getenv('BASELINE_OPTUNA_STUDY_NAME', 'baseline_opt')
        self.optuna_storage = os.getenv('BASELINE_OPTUNA_STORAGE', '')
        self.optuna_n_jobs = int(os.getenv('BASELINE_OPTUNA_N_JOBS', '1'))
        self.optuna_timeout = int(os.getenv('BASELINE_OPTUNA_TIMEOUT', '0'))

        # Year filter for fixed composition
        self.year_start = int(os.getenv('BASELINE_YEAR_START', '2010'))
        self.year_end = int(os.getenv('BASELINE_YEAR_END', '2016'))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'description_model_name': self.description_model_name,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'patience': self.patience,
            'device': self.device,
            'dataset_file': self.dataset_file,
            'description_tokenized_file': self.description_tokenized_file,
            'use_time_based_cv': self.use_time_based_cv,
            'cv_folds': self.cv_folds,
            'cv_train_size': self.cv_train_size,
            'cv_val_size': self.cv_val_size,
            'optuna_n_trials': self.optuna_n_trials,
            'optuna_study_name': self.optuna_study_name,
            'optuna_storage': self.optuna_storage,
            'optuna_n_jobs': self.optuna_n_jobs,
            'optuna_timeout': self.optuna_timeout,
            'year_start': self.year_start,
            'year_end': self.year_end,
        }

    def print_config(self) -> None:
        print("📋 Baseline Configuration:")
        print(f"   Config file: {self.config_file}")
        print(f"   Description model: {self.description_model_name}")
        print(f"   Embedding dim: {self.embedding_dim}")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   Dropout: {self.dropout_rate}")
        print(f"   LR: {self.learning_rate}")
        print(f"   Weight decay: {self.weight_decay}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Patience: {self.patience}")
        print(f"   Device: {self.device}")
        print(f"   Dataset: {self.dataset_file}")
        print(f"   Tokenized: {self.description_tokenized_file}")
        print(f"   CVTT: folds={self.cv_folds}, train={self.cv_train_size}, val={self.cv_val_size}, enabled={self.use_time_based_cv}")
        print(f"   Optuna: trials={self.optuna_n_trials}, study={self.optuna_study_name}")


config = BaselineConfig()


def get_config(config_file: str = None) -> BaselineConfig:
    return BaselineConfig(config_file) if config_file else config


