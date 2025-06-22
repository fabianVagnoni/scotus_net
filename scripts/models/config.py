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
        load_dotenv(config_file)
        
        self.config_file = str(config_file)
        self._load_config()
    
    def _load_config(self):
        """Load all configuration parameters."""
        
        # Model Architecture
        self.bio_model_name = os.getenv('BIO_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.description_model_name = os.getenv('DESCRIPTION_MODEL_NAME', 'Stern5497/sbert-legal-xlm-roberta-base')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '384'))
        self.hidden_dim = int(os.getenv('HIDDEN_DIM', '512'))
        self.max_justices = int(os.getenv('MAX_JUSTICES', '15'))
        
        # Attention Mechanism
        self.use_justice_attention = os.getenv('USE_JUSTICE_ATTENTION', 'true').lower() == 'true'
        self.num_attention_heads = int(os.getenv('NUM_ATTENTION_HEADS', '4'))
        
        # Regularization
        self.dropout_rate = float(os.getenv('DROPOUT_RATE', '0.1'))
        self.weight_decay = float(os.getenv('WEIGHT_DECAY', '0.01'))
        
        # Training Configuration
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.0001'))
        self.num_epochs = int(os.getenv('NUM_EPOCHS', '10'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '4'))
        self.num_workers = int(os.getenv('NUM_WORKERS', '0'))
        
        # Early Stopping and Scheduling
        self.patience = int(os.getenv('PATIENCE', '5'))
        self.lr_scheduler_factor = float(os.getenv('LR_SCHEDULER_FACTOR', '0.5'))
        self.lr_scheduler_patience = int(os.getenv('LR_SCHEDULER_PATIENCE', '3'))
        
        # Gradient Clipping
        self.max_grad_norm = float(os.getenv('MAX_GRAD_NORM', '1.0'))
        
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
        self.train_ratio = float(os.getenv('TRAIN_RATIO', '0.7'))
        self.val_ratio = float(os.getenv('VAL_RATIO', '0.15'))
        self.test_ratio = float(os.getenv('TEST_RATIO', '0.15'))
        self.split_random_state = int(os.getenv('SPLIT_RANDOM_STATE', '42'))
        
        # Loss Function
        self.loss_function = os.getenv('LOSS_FUNCTION', 'kl_div')
        self.kl_reduction = os.getenv('KL_REDUCTION', 'batchmean')
        
        # Validation and Evaluation
        self.validation_frequency = int(os.getenv('VALIDATION_FREQUENCY', '10'))
        self.evaluate_on_test = os.getenv('EVALUATE_ON_TEST', 'true').lower() == 'true'
        
        # Logging and Progress
        self.verbose_training = os.getenv('VERBOSE_TRAINING', 'true').lower() == 'true'
        self.log_frequency = int(os.getenv('LOG_FREQUENCY', '10'))
        
        # Memory Management
        self.clear_cache_on_oom = os.getenv('CLEAR_CACHE_ON_OOM', 'true').lower() == 'true'
        
        # Model Saving
        self.save_checkpoints = os.getenv('SAVE_CHECKPOINTS', 'true').lower() == 'true'
        self.checkpoint_frequency = int(os.getenv('CHECKPOINT_FREQUENCY', '5'))
        
        # Cross-Attention Specific
        self.use_attention_ffn = os.getenv('USE_ATTENTION_FFN', 'true').lower() == 'true'
        self.attention_ffn_multiplier = int(os.getenv('ATTENTION_FFN_MULTIPLIER', '2'))
        
        # Advanced Training
        self.use_mixed_precision = os.getenv('USE_MIXED_PRECISION', 'false').lower() == 'true'
        self.gradient_accumulation_steps = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '1'))
        
        # Random Seeds
        self.random_seed = int(os.getenv('RANDOM_SEED', '42'))
        self.torch_seed = int(os.getenv('TORCH_SEED', '42'))
        self.numpy_seed = int(os.getenv('NUMPY_SEED', '42'))
        
        # Model Loading and Caching
        self.use_model_cache = os.getenv('USE_MODEL_CACHE', 'true').lower() == 'true'
        self.download_models = os.getenv('DOWNLOAD_MODELS', 'true').lower() == 'true'
        
        # Data Validation
        self.min_justices_per_case = int(os.getenv('MIN_JUSTICES_PER_CASE', '1'))
        self.max_justices_per_case = int(os.getenv('MAX_JUSTICES_PER_CASE', '15'))
        self.skip_missing_descriptions = os.getenv('SKIP_MISSING_DESCRIPTIONS', 'true').lower() == 'true'
        self.skip_missing_biographies = os.getenv('SKIP_MISSING_BIOGRAPHIES', 'true').lower() == 'true'
    
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
            
            # Training Configuration
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            
            # Early Stopping and Scheduling
            'patience': self.patience,
            'lr_scheduler_factor': self.lr_scheduler_factor,
            'lr_scheduler_patience': self.lr_scheduler_patience,
            
            # Gradient Clipping
            'max_grad_norm': self.max_grad_norm,
            
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
            'test_ratio': self.test_ratio,
            'split_random_state': self.split_random_state,
            
            # Loss Function
            'loss_function': self.loss_function,
            'kl_reduction': self.kl_reduction,
            
            # Validation and Evaluation
            'validation_frequency': self.validation_frequency,
            'evaluate_on_test': self.evaluate_on_test,
            
            # Logging and Progress
            'verbose_training': self.verbose_training,
            'log_frequency': self.log_frequency,
            
            # Memory Management
            'clear_cache_on_oom': self.clear_cache_on_oom,
            
            # Model Saving
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_frequency': self.checkpoint_frequency,
            
            # Cross-Attention Specific
            'use_attention_ffn': self.use_attention_ffn,
            'attention_ffn_multiplier': self.attention_ffn_multiplier,
            
            # Advanced Training
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            
            # Random Seeds
            'random_seed': self.random_seed,
            'torch_seed': self.torch_seed,
            'numpy_seed': self.numpy_seed,
            
            # Model Loading and Caching
            'use_model_cache': self.use_model_cache,
            'download_models': self.download_models,
            
            # Data Validation
            'min_justices_per_case': self.min_justices_per_case,
            'max_justices_per_case': self.max_justices_per_case,
            'skip_missing_descriptions': self.skip_missing_descriptions,
            'skip_missing_biographies': self.skip_missing_biographies,
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
        'lr_scheduler_factor': config.lr_scheduler_factor,
        'lr_scheduler_patience': config.lr_scheduler_patience,
        'max_grad_norm': config.max_grad_norm,
        'loss_function': config.loss_function,
        'kl_reduction': config.kl_reduction,
        'validation_frequency': config.validation_frequency,
        'log_frequency': config.log_frequency,
        'verbose_training': config.verbose_training
    }


def get_data_config() -> Dict[str, Any]:
    """Get data configuration as dictionary."""
    return {
        'dataset_file': config.dataset_file,
        'bio_tokenized_file': config.bio_tokenized_file,
        'description_tokenized_file': config.description_tokenized_file,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'split_random_state': config.split_random_state,
        'min_justices_per_case': config.min_justices_per_case,
        'max_justices_per_case': config.max_justices_per_case,
        'skip_missing_descriptions': config.skip_missing_descriptions,
        'skip_missing_biographies': config.skip_missing_biographies
    }


if __name__ == "__main__":
    # Print configuration when run directly
    config.print_config() 