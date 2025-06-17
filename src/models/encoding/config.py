#!/usr/bin/env python3
"""
Configuration loader for SCOTUS AI encoding pipeline.
Loads hyperparameters from config.env file.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union

class EncodingConfig:
    """Configuration class for encoding hyperparameters."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration from environment file.
        
        Args:
            config_file: Path to config file. If None, looks for config.env in same directory.
        """
        if config_file is None:
            # Look for config.env in the same directory as this file
            current_dir = Path(__file__).parent
            config_file = current_dir / "config.env"
        
        self.config_file = Path(config_file)
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from the .env file."""
        if not self.config_file.exists():
            print(f"⚠️  Config file not found: {self.config_file}")
            print("Using default values...")
            self._set_defaults()
            return
        
        print(f"📥 Loading encoding config from: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                self._config[key] = value
        
        print(f"✅ Loaded {len(self._config)} configuration parameters")
    
    def _set_defaults(self):
        """Set default configuration values."""
        self._config = {
            'BIO_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
            'DESCRIPTION_MODEL_NAME': 'Stern5497/sbert-legal-xlm-roberta-base',
            'EMBEDDING_DIM': '384',
            'MAX_SEQUENCE_LENGTH': '512',
            'BIO_BATCH_SIZE': '16',
            'DESCRIPTION_BATCH_SIZE': '8',
            'DEVICE': 'auto',
            'BIO_OUTPUT_FILE': 'data/processed/encoded_bios.pkl',
            'DESCRIPTION_OUTPUT_FILE': 'data/processed/encoded_descriptions.pkl',
            'DATASET_FILE': 'data/processed/case_dataset.json',
            'BIO_INPUT_DIR': 'data/processed/bios',
            'DESCRIPTION_INPUT_DIR': 'data/processed/case_descriptions',
            'MAX_DESCRIPTION_WORDS': '10000',
            'SHOW_PROGRESS': 'true',
            'CLEAR_CACHE_ON_OOM': 'true',
            'USE_MODEL_CACHE': 'true',
            'TEMPERATURE': '1.0',
            'RANDOM_SEED': '42',
            'NUM_WORKERS': '4'
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return self._config.get(key, default)
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get a string configuration value."""
        return str(self._config.get(key, default))
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        value = self._config.get(key, str(default))
        try:
            return int(value)
        except (ValueError, TypeError):
            print(f"⚠️  Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        value = self._config.get(key, str(default))
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"⚠️  Invalid float value for {key}: {value}, using default: {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = self._config.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_path(self, key: str, default: str = "") -> Path:
        """Get a path configuration value."""
        path_str = self.get_str(key, default)
        return Path(path_str) if path_str else Path(default)
    
    # Convenience properties for commonly used values
    @property
    def bio_model_name(self) -> str:
        return self.get_str('BIO_MODEL_NAME')
    
    @property
    def description_model_name(self) -> str:
        return self.get_str('DESCRIPTION_MODEL_NAME')
    
    @property
    def embedding_dim(self) -> int:
        return self.get_int('EMBEDDING_DIM', 384)
    
    @property
    def max_sequence_length(self) -> int:
        return self.get_int('MAX_SEQUENCE_LENGTH', 512)
    
    @property
    def bio_batch_size(self) -> int:
        return self.get_int('BIO_BATCH_SIZE', 16)
    
    @property
    def description_batch_size(self) -> int:
        return self.get_int('DESCRIPTION_BATCH_SIZE', 8)
    
    @property
    def device(self) -> str:
        return self.get_str('DEVICE', 'auto')
    
    @property
    def bio_output_file(self) -> str:
        return self.get_str('BIO_OUTPUT_FILE')
    
    @property
    def description_output_file(self) -> str:
        return self.get_str('DESCRIPTION_OUTPUT_FILE')
    
    @property
    def dataset_file(self) -> str:
        return self.get_str('DATASET_FILE')
    
    @property
    def bio_input_dir(self) -> str:
        return self.get_str('BIO_INPUT_DIR')
    
    @property
    def description_input_dir(self) -> str:
        return self.get_str('DESCRIPTION_INPUT_DIR')
    
    @property
    def max_description_words(self) -> int:
        return self.get_int('MAX_DESCRIPTION_WORDS', 10000)
    
    @property
    def show_progress(self) -> bool:
        return self.get_bool('SHOW_PROGRESS', True)
    
    @property
    def clear_cache_on_oom(self) -> bool:
        return self.get_bool('CLEAR_CACHE_ON_OOM', True)
    
    @property
    def use_model_cache(self) -> bool:
        return self.get_bool('USE_MODEL_CACHE', True)
    
    @property
    def temperature(self) -> float:
        return self.get_float('TEMPERATURE', 1.0)
    
    @property
    def random_seed(self) -> int:
        return self.get_int('RANDOM_SEED', 42)
    
    @property
    def num_workers(self) -> int:
        return self.get_int('NUM_WORKERS', 4)
    
    def print_config(self):
        """Print the current configuration."""
        print("\n📋 ENCODING CONFIGURATION:")
        print("=" * 50)
        
        sections = {
            "Model Settings": [
                ('BIO_MODEL_NAME', self.bio_model_name),
                ('DESCRIPTION_MODEL_NAME', self.description_model_name),
                ('EMBEDDING_DIM', self.embedding_dim),
                ('MAX_SEQUENCE_LENGTH', self.max_sequence_length),
            ],
            "Batch Settings": [
                ('BIO_BATCH_SIZE', self.bio_batch_size),
                ('DESCRIPTION_BATCH_SIZE', self.description_batch_size),
                ('DEVICE', self.device),
            ],
            "File Paths": [
                ('DATASET_FILE', self.dataset_file),
                ('BIO_INPUT_DIR', self.bio_input_dir),
                ('DESCRIPTION_INPUT_DIR', self.description_input_dir),
                ('BIO_OUTPUT_FILE', self.bio_output_file),
                ('DESCRIPTION_OUTPUT_FILE', self.description_output_file),
            ],
            "Processing": [
                ('MAX_DESCRIPTION_WORDS', self.max_description_words),
                ('SHOW_PROGRESS', self.show_progress),
                ('CLEAR_CACHE_ON_OOM', self.clear_cache_on_oom),
                ('RANDOM_SEED', self.random_seed),
            ]
        }
        
        for section_name, settings in sections.items():
            print(f"\n{section_name}:")
            for key, value in settings:
                print(f"  {key}: {value}")

# Global configuration instance
_config = None

def get_config(config_file: str = None) -> EncodingConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None or config_file is not None:
        _config = EncodingConfig(config_file)
    return _config

def load_config(config_file: str = None) -> EncodingConfig:
    """Load and return configuration (alias for get_config)."""
    return get_config(config_file)

# Convenience functions for direct access
def get_bio_config() -> Dict[str, Any]:
    """Get biography encoding configuration."""
    config = get_config()
    return {
        'model_name': config.bio_model_name,
        'embedding_dim': config.embedding_dim,
        'max_sequence_length': config.max_sequence_length,
        'batch_size': config.bio_batch_size,
        'device': config.device,
        'output_file': config.bio_output_file,
        'input_dir': config.bio_input_dir,
    }

def get_description_config() -> Dict[str, Any]:
    """Get case description encoding configuration."""
    config = get_config()
    return {
        'model_name': config.description_model_name,
        'embedding_dim': config.embedding_dim,
        'max_sequence_length': config.max_sequence_length,
        'batch_size': config.description_batch_size,
        'device': config.device,
        'output_file': config.description_output_file,
        'input_dir': config.description_input_dir,
        'max_words': config.max_description_words,
    }

if __name__ == "__main__":
    # Test the configuration loader
    config = get_config()
    config.print_config() 