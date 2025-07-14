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
        
        # Focal Loss Parameters
        self.focal_gamma_initial = float(os.getenv('FOCAL_GAMMA_INITIAL', '2.0'))
        self.focal_gamma_final = float(os.getenv('FOCAL_GAMMA_FINAL', '1.0'))
        self.focal_gamma_decay_rate = float(os.getenv('FOCAL_GAMMA_DECAY_RATE', '0.1'))
        self.focal_weight_power = float(os.getenv('FOCAL_WEIGHT_POWER', '1.0'))
        
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
        
        # Sentence Transformer Fine-tuning
        self.enable_sentence_transformer_finetuning = os.getenv('ENABLE_SENTENCE_TRANSFORMER_FINETUNING', 'true').lower() == 'true'
        self.sentence_transformer_learning_rate = float(os.getenv('SENTENCE_TRANSFORMER_LEARNING_RATE', '1e-5'))
        self.unfreeze_bio_model = os.getenv('UNFREEZE_BIO_MODEL', 'true').lower() == 'true'
        self.unfreeze_description_model = os.getenv('UNFREEZE_DESCRIPTION_MODEL', 'true').lower() == 'true'
        
        # Three-Step Fine-tuning Strategy
        self.first_unfreeze_epoch = int(os.getenv('FIRST_UNFREEZE_EPOCH', '3'))
        self.second_unfreeze_epoch = int(os.getenv('SECOND_UNFREEZE_EPOCH', '5'))
        self.initial_layers_to_unfreeze = int(os.getenv('INITIAL_LAYERS_TO_UNFREEZE', '3'))
        self.lr_reduction_factor = float(os.getenv('LR_REDUCTION_FACTOR', '0.5'))
        self.reduce_main_lr_on_unfreeze = os.getenv('REDUCE_MAIN_LR_ON_UNFREEZE', 'true').lower() == 'true'
        
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
        
        # Hyperparameter Optimization
        self.optuna_n_trials = int(os.getenv('OPTUNA_N_TRIALS', '50'))
        self.optuna_max_trial_time = int(os.getenv('OPTUNA_MAX_TRIAL_TIME', '300'))
        self.optuna_max_epochs = int(os.getenv('OPTUNA_MAX_EPOCHS', '10'))
        self.optuna_min_epochs = int(os.getenv('OPTUNA_MIN_EPOCHS', '2'))
        self.optuna_early_stop_patience = int(os.getenv('OPTUNA_EARLY_STOP_PATIENCE', '3'))
        self.optuna_max_train_samples = int(os.getenv('OPTUNA_MAX_TRAIN_SAMPLES', '500'))
        self.optuna_max_val_samples = int(os.getenv('OPTUNA_MAX_VAL_SAMPLES', '100'))
        self.optuna_pruner_startup_trials = int(os.getenv('OPTUNA_PRUNER_STARTUP_TRIALS', '5'))
        self.optuna_pruner_warmup_steps = int(os.getenv('OPTUNA_PRUNER_WARMUP_STEPS', '3'))
        
        # Hyperparameter Search Spaces
        self.optuna_hidden_dim_options = self._parse_int_list(os.getenv('OPTUNA_HIDDEN_DIM_OPTIONS', '256,512,768,1024'))
        self.optuna_dropout_rate_range = self._parse_float_range(os.getenv('OPTUNA_DROPOUT_RATE_RANGE', '0.1,0.5,0.1'))
        self.optuna_attention_heads_options = self._parse_int_list(os.getenv('OPTUNA_ATTENTION_HEADS_OPTIONS', '2,4,6,8,12,16'))
        self.optuna_justice_attention_options = self._parse_bool_list(os.getenv('OPTUNA_JUSTICE_ATTENTION_OPTIONS', 'true,false'))
        self.optuna_learning_rate_range = self._parse_float_range_with_log(os.getenv('OPTUNA_LEARNING_RATE_RANGE', '1e-5,1e-3,true'))
        self.optuna_batch_size_options = self._parse_int_list(os.getenv('OPTUNA_BATCH_SIZE_OPTIONS', '8,16,32'))
        self.optuna_weight_decay_range = self._parse_float_range_with_log(os.getenv('OPTUNA_WEIGHT_DECAY_RANGE', '1e-4,1e-1,true'))
        
        # Fine-tuning Strategy Search Spaces
        self.optuna_first_unfreeze_epoch_options = self._parse_int_list(os.getenv('OPTUNA_FIRST_UNFREEZE_EPOCH_OPTIONS', '-1,2,3,4'))
        self.optuna_second_unfreeze_epoch_options = self._parse_int_list(os.getenv('OPTUNA_SECOND_UNFREEZE_EPOCH_OPTIONS', '-1,4,5,6'))
        self.optuna_initial_layers_to_unfreeze_options = self._parse_int_list(os.getenv('OPTUNA_INITIAL_LAYERS_TO_UNFREEZE_OPTIONS', '2,3,4'))
        self.optuna_lr_reduction_factor_range = self._parse_float_range(os.getenv('OPTUNA_LR_REDUCTION_FACTOR_RANGE', '0.1,0.8,0.1'))
        
        # Hyperparameter Tuning Control
        self.tune_hidden_dim = os.getenv('TUNE_HIDDEN_DIM', 'true').lower() == 'true'
        self.tune_dropout_rate = os.getenv('TUNE_DROPOUT_RATE', 'true').lower() == 'true'
        self.tune_num_attention_heads = os.getenv('TUNE_NUM_ATTENTION_HEADS', 'true').lower() == 'true'
        self.tune_use_justice_attention = os.getenv('TUNE_USE_JUSTICE_ATTENTION', 'true').lower() == 'true'
        self.tune_learning_rate = os.getenv('TUNE_LEARNING_RATE', 'true').lower() == 'true'
        self.tune_batch_size = os.getenv('TUNE_BATCH_SIZE', 'true').lower() == 'true'
        self.tune_weight_decay = os.getenv('TUNE_WEIGHT_DECAY', 'true').lower() == 'true'
        
        # Fine-tuning Strategy Tuning Control
        self.tune_fine_tuning_strategy = os.getenv('TUNE_FINE_TUNING_STRATEGY', 'true').lower() == 'true'
    
    def _parse_int_list(self, value: str) -> list:
        """Parse comma-separated integers."""
        return [int(x.strip()) for x in value.split(',')]
    
    def _parse_bool_list(self, value: str) -> list:
        """Parse comma-separated booleans."""
        return [x.strip().lower() == 'true' for x in value.split(',')]
    
    def _parse_float_range(self, value: str) -> tuple:
        """Parse float range as min,max,step."""
        parts = [float(x.strip()) for x in value.split(',')]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        elif len(parts) == 2:
            return (parts[0], parts[1], None)
        else:
            raise ValueError(f"Invalid float range format: {value}")
    
    def _parse_float_range_with_log(self, value: str) -> tuple:
        """Parse float range as min,max,log_scale."""
        parts = value.split(',')
        if len(parts) == 3:
            return (float(parts[0].strip()), float(parts[1].strip()), parts[2].strip().lower() == 'true')
        else:
            raise ValueError(f"Invalid float range with log format: {value}")
    
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
            
            # Focal Loss Parameters
            'focal_gamma_initial': self.focal_gamma_initial,
            'focal_gamma_final': self.focal_gamma_final,
            'focal_gamma_decay_rate': self.focal_gamma_decay_rate,
            'focal_weight_power': self.focal_weight_power,
            
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
            
            # Sentence Transformer Fine-tuning
            'enable_sentence_transformer_finetuning': self.enable_sentence_transformer_finetuning,
            'sentence_transformer_learning_rate': self.sentence_transformer_learning_rate,
            'unfreeze_bio_model': self.unfreeze_bio_model,
            'unfreeze_description_model': self.unfreeze_description_model,
            
            # Three-Step Fine-tuning Strategy
            'first_unfreeze_epoch': self.first_unfreeze_epoch,
            'second_unfreeze_epoch': self.second_unfreeze_epoch,
            'initial_layers_to_unfreeze': self.initial_layers_to_unfreeze,
            'lr_reduction_factor': self.lr_reduction_factor,
            'reduce_main_lr_on_unfreeze': self.reduce_main_lr_on_unfreeze,
            
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
            
            # Hyperparameter Optimization
            'optuna_n_trials': self.optuna_n_trials,
            'optuna_max_trial_time': self.optuna_max_trial_time,
            'optuna_max_epochs': self.optuna_max_epochs,
            'optuna_min_epochs': self.optuna_min_epochs,
            'optuna_early_stop_patience': self.optuna_early_stop_patience,
            'optuna_max_train_samples': self.optuna_max_train_samples,
            'optuna_max_val_samples': self.optuna_max_val_samples,
            'optuna_pruner_startup_trials': self.optuna_pruner_startup_trials,
            'optuna_pruner_warmup_steps': self.optuna_pruner_warmup_steps,
            
            # Hyperparameter Search Spaces
            'optuna_hidden_dim_options': self.optuna_hidden_dim_options,
            'optuna_dropout_rate_range': self.optuna_dropout_rate_range,
            'optuna_attention_heads_options': self.optuna_attention_heads_options,
            'optuna_justice_attention_options': self.optuna_justice_attention_options,
            'optuna_learning_rate_range': self.optuna_learning_rate_range,
            'optuna_batch_size_options': self.optuna_batch_size_options,
            'optuna_weight_decay_range': self.optuna_weight_decay_range,
            
            # Fine-tuning Strategy Search Spaces
            'optuna_first_unfreeze_epoch_options': self.optuna_first_unfreeze_epoch_options,
            'optuna_second_unfreeze_epoch_options': self.optuna_second_unfreeze_epoch_options,
            'optuna_initial_layers_to_unfreeze_options': self.optuna_initial_layers_to_unfreeze_options,
            'optuna_lr_reduction_factor_range': self.optuna_lr_reduction_factor_range,
            
            # Hyperparameter Tuning Control
            'tune_hidden_dim': self.tune_hidden_dim,
            'tune_dropout_rate': self.tune_dropout_rate,
            'tune_num_attention_heads': self.tune_num_attention_heads,
            'tune_use_justice_attention': self.tune_use_justice_attention,
            'tune_learning_rate': self.tune_learning_rate,
            'tune_batch_size': self.tune_batch_size,
            'tune_weight_decay': self.tune_weight_decay,
            
            # Fine-tuning Strategy Tuning Control
            'tune_fine_tuning_strategy': self.tune_fine_tuning_strategy,
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