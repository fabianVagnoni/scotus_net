#!/usr/bin/env python3
"""
Simplified training runner for SCOTUS voting model.

This script provides a simple interface to train the SCOTUS voting model
with the simplified architecture and training process.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import random
import numpy as np

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.models.model_trainer import SCOTUSModelTrainer
from scripts.models.config import ModelConfig
from scripts.utils.logger import get_logger


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(experiment_name: str):
    """Set up logging for the training run."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set up file logging
    log_file = logs_dir / f"training_{experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SCOTUS Voting Model")
    parser.add_argument("--dataset-file", type=str, help="Path to the case dataset file")
    parser.add_argument("--experiment-name", type=str, default="scotus_training", help="Name for this training experiment")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device to use for training")
    parser.add_argument("--eval-holdout", action='store_true', help="Evaluate on holdout test set after training")
    parser.add_argument("--eval-test", action='store_true', help="Alias: evaluate on holdout test set after training")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.experiment_name)
    logger = get_logger(__name__)
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    logger.info(f"🎲 Set random seed to {args.seed}")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"🖥️ Using device: {device}")
    
    # Load configuration
    config = ModelConfig()
    if args.config_file:
        logger.info(f"📋 Loading configuration from: {args.config_file}")
        config.load_from_file(args.config_file)
    
    # Override device in config
    config.device = device
    
    logger.info("🚀 Starting SCOTUS Voting Model Training")
    logger.info(f"   Experiment: {args.experiment_name}")
    logger.info(f"   Dataset: {args.dataset_file or config.dataset_file}")
    logger.info(f"   Bio Model: {config.bio_model_name}")
    logger.info(f"   Description Model: {config.description_model_name}")
    logger.info(f"   Embedding Dim: {config.embedding_dim}")
    logger.info(f"   Hidden Dim: {config.hidden_dim}")
    logger.info(f"   Max Justices: {config.max_justices}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Epochs: {config.num_epochs}")
    logger.info(f"   Use Justice Attention: {config.use_justice_attention}")
    logger.info(f"   Use Noise Regularization: {config.use_noise_reg}")
    
    try:
        # Initialize trainer
        trainer = SCOTUSModelTrainer()
        trainer.config = config  # Override with our config
        trainer.device = torch.device(device)
        
        # Train model
        # Ensure dataset path is consistently used across training and holdout evaluation
        if args.dataset_file:
            trainer.config.dataset_file = args.dataset_file
        trainer.train_model(dataset_file=args.dataset_file)
        
        logger.info("✅ Training completed successfully!")
        
        # Evaluate on holdout test set if requested
        if args.eval_holdout or args.eval_test:
            logger.info("🧪 Evaluating on holdout test set...")
            try:
                results = trainer.evaluate_on_holdout_test_set()
                logger.info(f"   Holdout Test Results: {results}")
            except Exception as e:
                logger.warning(f"   Holdout evaluation failed: {str(e)}")
        
        logger.info(f"💾 Model saved to: {config.model_output_dir}/best_model.pth")
        logger.info(f"📋 Training logs saved to: logs/training_{args.experiment_name}.log")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()