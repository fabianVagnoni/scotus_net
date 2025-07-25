#!/usr/bin/env python3
"""
Example script to run contrastive justice pretraining.
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import ContrastiveJusticeConfig
from contrastive_trainer import ContrastiveJusticeTrainer

def main():
    """Run contrastive justice pretraining."""
    
    print("🚀 Starting Contrastive Justice Pretraining")
    
    # Create configuration
    config = ContrastiveJusticeConfig()
    
    # Print configuration
    config.print_config()
    
    # Check if required files exist
    required_files = [
        config.justices_file,
        config.trunc_bio_tokenized_file,
        config.full_bio_tokenized_file
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            print("Please run the data pipeline and tokenization first.")
            return
    
    print("✅ All required files found")
    
    # Create trainer
    trainer = ContrastiveJusticeTrainer(config)
    
    # Run training
    try:
        trained_model = trainer.train_model(config.justices_file)
        print("🎉 Training completed successfully!")
        
        # Model is already saved by the trainer
        print(f"💾 Models saved to: {config.model_output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 