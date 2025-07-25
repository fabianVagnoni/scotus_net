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
                    
                    # Set the attribute based on the key
                    if hasattr(self, key):
                        # Convert value to appropriate type
                        setattr(self, key, self._convert_value(key, value))
                    else:
                        print(f"⚠️  Unknown config key: {key}")
    
    def _convert_value(self, key: str, value: str) -> Any:
        """Convert string value to appropriate type based on key."""
        # Define type mappings for different configuration keys
        int_keys = {
            'batch_size', 'num_epochs', 'num_workers', 'lr_scheduler_patience', 
            'max_patience', 'test_set_size', 'val_set_size'
        }
        float_keys = {
            'dropout_rate', 'learning_rate', 'weight_decay', 'temperature', 
            'alpha', 'lr_scheduler_factor', 'val_split'
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
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dropout_rate': 0.1,
            
            # Training configuration
            'batch_size': 8,
            'learning_rate': 1e-5,
            'num_epochs': 10,
            'weight_decay': 0.01,
            'num_workers': 2,
            
            # Loss configuration
            'temperature': 0.1,
            'alpha': 0.5,
            
            # Learning rate scheduler
            'lr_scheduler_factor': 0.5,
            'lr_scheduler_patience': 3,
            'max_patience': 10,
            
            # Data configuration
            'val_split': 0.2,
            'justices_file': 'data/raw/justices.json',
            'trunc_bio_tokenized_file': 'data/processed/encoded_bios.pkl',
            'full_bio_tokenized_file': 'data/raw/encoded_bios.pkl',
            'pretraining_dataset_file': 'data/processed/pretraining_dataset.json',
            'test_set_size': 10,
            'val_set_size': 20,
            
            # Output configuration
            'model_output_dir': 'models/contrastive_justice'
        }
        return defaults.get(key, None)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Model configuration
        self.model_name = self._get_default_value('model_name')
        self.dropout_rate = self._get_default_value('dropout_rate')
        
        # Training configuration
        self.batch_size = self._get_default_value('batch_size')
        self.learning_rate = self._get_default_value('learning_rate')
        self.num_epochs = self._get_default_value('num_epochs')
        self.weight_decay = self._get_default_value('weight_decay')
        self.num_workers = self._get_default_value('num_workers')
        
        # Loss configuration
        self.temperature = self._get_default_value('temperature')
        self.alpha = self._get_default_value('alpha')
        
        # Learning rate scheduler
        self.lr_scheduler_factor = self._get_default_value('lr_scheduler_factor')
        self.lr_scheduler_patience = self._get_default_value('lr_scheduler_patience')
        self.max_patience = self._get_default_value('max_patience')
        
        # Data configuration
        self.val_split = self._get_default_value('val_split')
        self.justices_file = self._get_default_value('justices_file')
        self.trunc_bio_tokenized_file = self._get_default_value('trunc_bio_tokenized_file')
        self.full_bio_tokenized_file = self._get_default_value('full_bio_tokenized_file')
        self.pretraining_dataset_file = self._get_default_value('pretraining_dataset_file')
        self.test_set_size = self._get_default_value('test_set_size')
        self.val_set_size = self._get_default_value('val_set_size')
        
        # Output configuration
        self.model_output_dir = self._get_default_value('model_output_dir')
    
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