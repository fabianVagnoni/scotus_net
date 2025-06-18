"""Configuration management utilities for SCOTUS AI project."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """Initialize configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env_variables()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _load_env_variables(self):
        """Load environment variables from .env file."""
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.name').
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable.
        
        Args:
            key: Environment variable key.
            default: Default value if not found.
            
        Returns:
            Environment variable value.
        """
        return os.getenv(key, default)
    
    def update(self, key: str, value: Any):
        """Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            value: New value.
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)


# Global configuration instance
config = Config() 