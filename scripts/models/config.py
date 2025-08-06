#!/usr/bin/env python3
"""
Configuration loader for SCOTUS AI model components.

Loads configuration from config.env file and provides typed access to parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
from dotenv import load_dotenv


class ModelConfig:
    """Configuration class for SCOTUS AI model components."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to config file. If None, uses config.env in same directory.
        """
        if config_file is None:
            config_file = Path(__file__).parent / "config.env"
        
        # Load environment variables from config file
        # Use override=True to ensure config.env values take precedence over existing env vars
        load_dotenv(config_file, override=True)
        
        self.config_file = str(config_file)
        self._load_config()
    
    def _load_config(self):
        """Load all configuration parameters."""
        
        # Model Architecture
        self.bio_model_name = os.getenv('BIO_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.description_model_name = os.getenv('DESCRIPTION_MODEL_NAME', 'Stern5497/sbert-legal-xlm-roberta-base')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '384'))
        self.hidden_dim = int(os.getenv('HIDDEN_DIM', '512'))
        self.max_justices = int(os.getenv('MAX_JUSTICES', '11'))
        
        # Attention Mechanism
        self.use_justice_attention = os.getenv('USE_JUSTICE_ATTENTION', 'true').lower() == 'true'
        self.num_attention_heads = int(os.getenv('NUM_ATTENTION_HEADS', '4'))
        
        # Regularization
        self.dropout_rate = float(os.getenv('DROPOUT_RATE', '0.1'))
        self.weight_decay = float(os.getenv('WEIGHT_DECAY', '0.01'))
        self.use_noise_reg = os.getenv('USE_NOISE_REG', 'true').lower() == 'true'
        self.noise_reg_alpha = float(os.getenv('NOISE_REG_ALPHA', '5.0'))
        self.max_grad_norm = float(os.getenv('MAX_GRAD_NORM', '1.0'))
        
        # Training Configuration
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.0001'))
        self.num_epochs = int(os.getenv('NUM_EPOCHS', '10'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '16'))
        self.num_workers = int(os.getenv('NUM_WORKERS', '4'))
        
        # Early Stopping and Scheduling
        self.patience = int(os.getenv('PATIENCE', '5'))
        
        # Device Configuration
        device = os.getenv('DEVICE', 'auto').lower()
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Data Paths
        self.dataset_file = os.getenv('DATASET_FILE', 'data/processed/case_dataset.json')
        self.bio_tokenized_file = os.getenv('BIO_TOKENIZED_FILE', 'data/processed/encoded_bios.pkl')
        self.description_tokenized_file = os.getenv('DESCRIPTION_TOKENIZED_FILE', 'data/processed/encoded_descriptions.pkl')
        
        # Output Paths
        self.model_output_dir = os.getenv('MODEL_OUTPUT_DIR', 'models_output')
        self.best_model_name = os.getenv('BEST_MODEL_NAME', 'best_model.pth')
        
        # Dataset Splitting
        self.train_ratio = float(os.getenv('TRAIN_RATIO', '0.85'))
        self.val_ratio = float(os.getenv('VAL_RATIO', '0.15'))
        self.split_random_state = int(os.getenv('SPLIT_RANDOM_STATE', '42'))
        
        # Sentence Transformer Fine-tuning Strategy
        self.unfreeze_bio_model = os.getenv('UNFREEZE_BIO_MODEL', 'true').lower() == 'true'
        self.unfreeze_description_model = os.getenv('UNFREEZE_DESCRIPTION_MODEL', 'true').lower() == 'true'
        self.unfreeze_at_epoch = int(os.getenv('UNFREEZE_AT_EPOCH', '3'))
        self.sentence_transformer_learning_rate = float(os.getenv('SENTENCE_TRANSFORMER_LEARNING_RATE', '1e-5'))
        
        # Pretrained Model Path
        self.pretrained_bio_model = os.getenv('PRETRAINED_BIO_MODEL', '')
        
        # Legacy support for unfreeze_transformers (maps to unfreeze_at_epoch)
        self.unfreeze_transformers = self.unfreeze_at_epoch > 0
        
        # Hyperparameter Optimization Configuration
        self.optuna_n_trials = int(os.getenv('OPTUNA_N_TRIALS', '50'))
        self.optuna_study_name = os.getenv('OPTUNA_STUDY_NAME', 'scotus_voting_model_optimization')
        self.optuna_max_trial_time = int(os.getenv('OPTUNA_MAX_TRIAL_TIME', '1200'))
        self.optuna_max_epochs = int(os.getenv('OPTUNA_MAX_EPOCHS', '5'))
        self.optuna_min_epochs = int(os.getenv('OPTUNA_MIN_EPOCHS', '3'))
        self.optuna_early_stop_patience = int(os.getenv('OPTUNA_EARLY_STOP_PATIENCE', '3'))
        
        # Hyperparameter Tuning Control
        self.tune_hidden_dim = os.getenv('TUNE_HIDDEN_DIM', 'true').lower() == 'true'
        self.tune_num_attention_heads = os.getenv('TUNE_NUM_ATTENTION_HEADS', 'true').lower() == 'true'
        self.tune_use_justice_attention = os.getenv('TUNE_USE_JUSTICE_ATTENTION', 'true').lower() == 'true'
        self.tune_learning_rate = os.getenv('TUNE_LEARNING_RATE', 'true').lower() == 'true'
        self.tune_batch_size = os.getenv('TUNE_BATCH_SIZE', 'true').lower() == 'true'
        self.tune_weight_decay = os.getenv('TUNE_WEIGHT_DECAY', 'true').lower() == 'true'
        self.tune_dropout_rate = os.getenv('TUNE_DROPOUT_RATE', 'true').lower() == 'true'
        self.tune_use_noise_reg = os.getenv('TUNE_USE_NOISE_REG', 'true').lower() == 'true'
        self.tune_unfreezing = os.getenv('TUNE_UNFREEZING', 'true').lower() == 'true'
        self.tune_max_grad_norm = os.getenv('TUNE_MAX_GRAD_NORM', 'true').lower() == 'true'
        
        # Hyperparameter Search Spaces
        self.optuna_hidden_dim_options = self._parse_list_option(os.getenv('OPTUNA_HIDDEN_DIM_OPTIONS', '256,512,768,1024'), int)
        self.optuna_attention_heads_options = self._parse_list_option(os.getenv('OPTUNA_ATTENTION_HEADS_OPTIONS', '2,4,6,8'), int)
        self.optuna_justice_attention_options = self._parse_bool_list_option(os.getenv('OPTUNA_JUSTICE_ATTENTION_OPTIONS', 'true,false'))
        self.optuna_batch_size_options = self._parse_list_option(os.getenv('OPTUNA_BATCH_SIZE_OPTIONS', '8,16,32'), int)
        self.optuna_learning_rate_range = self._parse_range_option(os.getenv('OPTUNA_LEARNING_RATE_RANGE', '1e-5,1e-3,true'))
        self.optuna_weight_decay_range = self._parse_range_option(os.getenv('OPTUNA_WEIGHT_DECAY_RANGE', '1e-5,1e-2,true'))
        self.optuna_dropout_rate_range = self._parse_range_option(os.getenv('OPTUNA_DROPOUT_RATE_RANGE', '0.0,0.5,0.1'))
        self.optuna_use_noise_reg_options = self._parse_bool_list_option(os.getenv('OPTUNA_USE_NOISE_REG_OPTIONS', 'true,false'))
        self.optuna_noise_reg_alpha_range = self._parse_range_option(os.getenv('OPTUNA_NOISE_REG_ALPHA_RANGE', '1.0,10.0,false'))
        self.optuna_unfreeze_at_epoch_options = self._parse_list_option(os.getenv('OPTUNA_UNFREEZE_AT_EPOCH_OPTIONS', '2,3,4,5'), int)
        self.optuna_sentence_transformer_lr_range = self._parse_range_option(os.getenv('OPTUNA_SENTENCE_TRANSFORMER_LR_RANGE', '1e-6,1e-4,true'))
        self.optuna_max_grad_norm_range = self._parse_range_option(os.getenv('OPTUNA_MAX_GRAD_NORM_RANGE', '1.0,10.0,true'))

    def _parse_list_option(self, value: str, convert_type: type = str) -> list:
        """Parse comma-separated list option."""
        try:
            return [convert_type(x.strip()) for x in value.split(',')]
        except (ValueError, TypeError):
            return []
    
    def _parse_bool_list_option(self, value: str) -> list:
        """Parse comma-separated boolean list option."""
        try:
            return [x.strip().lower() == 'true' for x in value.split(',')]
        except (ValueError, TypeError):
            return [True, False]
    
    def _parse_range_option(self, value: str) -> tuple:
        """Parse range option (min,max,log_scale or min,max,step)."""
        try:
            parts = [x.strip() for x in value.split(',')]
            if len(parts) == 3:
                min_val = float(parts[0])
                max_val = float(parts[1])
                third_param = parts[2]
                if third_param.lower() in ('true', 'false'):
                    third_param = third_param.lower() == 'true'
                else:
                    third_param = float(third_param)
                return (min_val, max_val, third_param)
            elif len(parts) == 2:
                return (float(parts[0]), float(parts[1]), None)
            else:
                return (0.0, 1.0, None)
        except (ValueError, TypeError):
            return (0.0, 1.0, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.embedding_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Convert dot notation to attribute access
        if '.' in key:
            parts = key.split('.')
            obj = self
            for part in parts:
                obj = getattr(obj, part, default)
                if obj is default:
                    return default
            return obj
        else:
            return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            # Model Architecture
            'bio_model_name': self.bio_model_name,
            'description_model_name': self.description_model_name,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_justices': self.max_justices,
            
            # Attention Mechanism
            'use_justice_attention': self.use_justice_attention,
            'num_attention_heads': self.num_attention_heads,
            
            # Regularization
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'use_noise_reg': self.use_noise_reg,
            'noise_reg_alpha': self.noise_reg_alpha,
            'max_grad_norm': self.max_grad_norm,
            
            # Training Configuration
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            
            # Early Stopping and Scheduling
            'patience': self.patience,
            
            # Device Configuration
            'device': self.device,
            
            # Data Paths
            'dataset_file': self.dataset_file,
            'bio_tokenized_file': self.bio_tokenized_file,
            'description_tokenized_file': self.description_tokenized_file,
            
            # Output Paths
            'model_output_dir': self.model_output_dir,
            'best_model_name': self.best_model_name,
            
            # Dataset Splitting
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'split_random_state': self.split_random_state,
            
            # Sentence Transformer Fine-tuning Strategy
            'unfreeze_bio_model': self.unfreeze_bio_model,
            'unfreeze_description_model': self.unfreeze_description_model,
            'unfreeze_at_epoch': self.unfreeze_at_epoch,
            'sentence_transformer_learning_rate': self.sentence_transformer_learning_rate,
            
            # Pretrained Model Path
            'pretrained_bio_model': self.pretrained_bio_model,
        }
    
    def print_config(self):
        """Print current configuration."""
        print("📋 SCOTUS AI Model Configuration:")
        print(f"   📁 Config file: {self.config_file}")
        print(f"   🤖 Bio model: {self.bio_model_name}")
        print(f"   📖 Description model: {self.description_model_name}")
        print(f"   🔢 Embedding dim: {self.embedding_dim}")
        print(f"   🧠 Hidden dim: {self.hidden_dim}")
        print(f"   👥 Max justices: {self.max_justices}")
        print(f"   🎯 Use attention: {self.use_justice_attention}")
        print(f"   🔄 Attention heads: {self.num_attention_heads}")
        print(f"   📊 Batch size: {self.batch_size}")
        print(f"   🎓 Learning rate: {self.learning_rate}")
        print(f"   🔁 Epochs: {self.num_epochs}")
        print(f"   💾 Device: {self.device}")
        print(f"   🔄 Max grad norm: {self.max_grad_norm}")

# Global configuration instance
config = ModelConfig()


def get_config(config_file: str = None) -> ModelConfig:
    """
    Get configuration instance.
    
    Args:
        config_file: Path to config file. If None, uses default.
        
    Returns:
        ModelConfig instance
    """
    if config_file is None:
        return config
    else:
        return ModelConfig(config_file)


def get_model_config() -> Dict[str, Any]:
    """Get model configuration as dictionary."""
    return {
        'bio_model_name': config.bio_model_name,
        'description_model_name': config.description_model_name,
        'embedding_dim': config.embedding_dim,
        'hidden_dim': config.hidden_dim,
        'dropout_rate': config.dropout_rate,
        'max_justices': config.max_justices,
        'num_attention_heads': config.num_attention_heads,
        'use_justice_attention': config.use_justice_attention,
        'use_noise_reg': config.use_noise_reg,
        'noise_reg_alpha': config.noise_reg_alpha,
        'pretrained_bio_model': config.pretrained_bio_model,
        'device': config.device
    }


def get_training_config() -> Dict[str, Any]:
    """Get training configuration as dictionary."""
    return {
        'learning_rate': config.learning_rate,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'weight_decay': config.weight_decay,
        'patience': config.patience,
        'unfreeze_bio_model': config.unfreeze_bio_model,
        'unfreeze_description_model': config.unfreeze_description_model,
        'unfreeze_at_epoch': config.unfreeze_at_epoch,
        'sentence_transformer_learning_rate': config.sentence_transformer_learning_rate,
        'use_noise_reg': config.use_noise_reg,
        'noise_reg_alpha': config.noise_reg_alpha,
        'max_grad_norm': config.max_grad_norm
    }


def get_data_config() -> Dict[str, Any]:
    """Get data configuration as dictionary."""
    return {
        'dataset_file': config.dataset_file,
        'bio_tokenized_file': config.bio_tokenized_file,
        'description_tokenized_file': config.description_tokenized_file,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'split_random_state': config.split_random_state
    }


if __name__ == "__main__":
    # Print configuration when run directly
    config.print_config() 