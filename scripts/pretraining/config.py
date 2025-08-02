"""
Configuration for Contrastive Justice Pretraining
"""

import os
from pathlib import Path
from typing import Any

class ContrastiveJusticeConfig:
    def __init__(self, config_file: str = None):
        """
        Initialize configuration from environment file.
        
        Args:
            config_file: Path to config.env file. If None, uses default location.
        """
        if config_file is None:
            # Look for config.env in the current directory
            current_dir = Path(__file__).parent
            config_file = current_dir / "config.env"
        
        self._load_config(config_file)
    
    def _load_config(self, config_file: str):
        """Load configuration from .env file."""
        if os.path.exists(config_file):
            print(f"📥 Loading configuration from: {config_file}")
            self._load_from_env_file(config_file)
        else:
            print(f"⚠️  Config file not found: {config_file}")
            print("📝 Using default configuration values")
            self._set_defaults()
    
    def _load_from_env_file(self, config_file: str):
        """Load configuration values from .env file."""
        # First set all defaults so attributes exist
        self._set_defaults()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Map uppercase keys to lowercase attributes
                    key_mapping = {
                        'MODEL_NAME': 'model_name',
                        'DROPOUT_RATE': 'dropout_rate',
                        'BATCH_SIZE': 'batch_size',
                        'LEARNING_RATE': 'learning_rate',
                        'NUM_EPOCHS': 'num_epochs',
                        'WEIGHT_DECAY': 'weight_decay',
                        'NUM_WORKERS': 'num_workers',
                        'TEMPERATURE': 'temperature',
                        'ALPHA': 'alpha',
                        'LR_SCHEDULER_FACTOR': 'lr_scheduler_factor',
                        'LR_SCHEDULER_PATIENCE': 'lr_scheduler_patience',
                        'MAX_PATIENCE': 'max_patience',
                        'VAL_SPLIT': 'val_split',
                        'JUSTICES_FILE': 'justices_file',
                        'TRUNC_BIO_TOKENIZED_FILE': 'trunc_bio_tokenized_file',
                        'FULL_BIO_TOKENIZED_FILE': 'full_bio_tokenized_file',
                        'PRETRAINING_DATASET_FILE': 'pretraining_dataset_file',
                        'TEST_SET_SIZE': 'test_set_size',
                        'VAL_SET_SIZE': 'val_set_size',
                        'MODEL_OUTPUT_DIR': 'model_output_dir'
                    }
                    
                    # Get the corresponding attribute name
                    attr_name = key_mapping.get(key, key)
                    
                    # Set the attribute based on the key
                    if hasattr(self, attr_name):
                        # Convert value to appropriate type
                        setattr(self, attr_name, self._convert_value(key, value))
                    else:
                        print(f"⚠️  Unknown config key: {key}")
    
    def _convert_value(self, key: str, value: str) -> Any:
        """Convert string value to appropriate type based on key."""
        # Define type mappings for different configuration keys
        int_keys = {
            'BATCH_SIZE', 'NUM_EPOCHS', 'NUM_WORKERS', 'LR_SCHEDULER_PATIENCE', 
            'MAX_PATIENCE', 'TEST_SET_SIZE', 'VAL_SET_SIZE'
        }
        float_keys = {
            'DROPOUT_RATE', 'LEARNING_RATE', 'WEIGHT_DECAY', 'TEMPERATURE', 
            'ALPHA', 'LR_SCHEDULER_FACTOR', 'VAL_SPLIT'
        }
        bool_keys = set()  # No boolean keys in current config
        
        if key in int_keys:
            try:
                return int(value)
            except ValueError:
                print(f"⚠️  Invalid integer value for {key}: {value}, using default")
                return self._get_default_value(key)
        elif key in float_keys:
            try:
                return float(value)
            except ValueError:
                print(f"⚠️  Invalid float value for {key}: {value}, using default")
                return self._get_default_value(key)
        elif key in bool_keys:
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        else:
            return value
    
    def _get_default_value(self, key: str) -> Any:
        """Get default value for a configuration key."""
        defaults = {
            # Model configuration
            'MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
            'DROPOUT_RATE': 0.1,
            
            # Training configuration
            'BATCH_SIZE': 8,
            'LEARNING_RATE': 1e-5,
            'NUM_EPOCHS': 10,
            'WEIGHT_DECAY': 0.01,
            'NUM_WORKERS': 2,
            
            # Loss configuration
            'TEMPERATURE': 0.1,
            'ALPHA': 0.5,
            
            # Learning rate scheduler
            'LR_SCHEDULER_FACTOR': 0.5,
            'LR_SCHEDULER_PATIENCE': 3,
            'MAX_PATIENCE': 10,
            
            # Data configuration
            'VAL_SPLIT': 0.2,
            'JUSTICES_FILE': 'data/raw/justices.json',
            'TRUNC_BIO_TOKENIZED_FILE': 'data/processed/encoded_pre_trunc_bios.pkl',
            'FULL_BIO_TOKENIZED_FILE': 'data/processed/encoded_pre_full_bios.pkl',
            'PRETRAINING_DATASET_FILE': 'data/processed/pretraining_dataset.json',
            'TEST_SET_SIZE': 9,
            'VAL_SET_SIZE': 9,
            
            # Output configuration
            'MODEL_OUTPUT_DIR': 'models/contrastive_justice'
        }
        return defaults.get(key, None)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Model configuration
        self.model_name = self._get_default_value('MODEL_NAME')
        self.dropout_rate = self._get_default_value('DROPOUT_RATE')
        
        # Training configuration
        self.batch_size = self._get_default_value('BATCH_SIZE')
        self.learning_rate = self._get_default_value('LEARNING_RATE')
        self.num_epochs = self._get_default_value('NUM_EPOCHS')
        self.weight_decay = self._get_default_value('WEIGHT_DECAY')
        self.num_workers = self._get_default_value('NUM_WORKERS')
        
        # Loss configuration
        self.temperature = self._get_default_value('TEMPERATURE')
        self.alpha = self._get_default_value('ALPHA')
        
        # Learning rate scheduler
        self.lr_scheduler_factor = self._get_default_value('LR_SCHEDULER_FACTOR')
        self.lr_scheduler_patience = self._get_default_value('LR_SCHEDULER_PATIENCE')
        self.max_patience = self._get_default_value('MAX_PATIENCE')
        
        # Data configuration
        self.val_split = self._get_default_value('VAL_SPLIT')
        self.justices_file = self._get_default_value('JUSTICES_FILE')
        self.trunc_bio_tokenized_file = self._get_default_value('TRUNC_BIO_TOKENIZED_FILE')
        self.full_bio_tokenized_file = self._get_default_value('FULL_BIO_TOKENIZED_FILE')
        self.pretraining_dataset_file = self._get_default_value('PRETRAINING_DATASET_FILE')
        self.test_set_size = self._get_default_value('TEST_SET_SIZE')
        self.val_set_size = self._get_default_value('VAL_SET_SIZE')
        
        # Output configuration
        self.model_output_dir = self._get_default_value('MODEL_OUTPUT_DIR')
    
    def print_config(self):
        """Print current configuration."""
        print("📊 Contrastive Justice Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Dropout Rate: {self.dropout_rate}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Weight Decay: {self.weight_decay}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Alpha: {self.alpha}")
        print(f"   Validation Split: {self.val_split}")
        print(f"   Justices File: {self.justices_file}")
        print(f"   Output Dir: {self.model_output_dir}") 