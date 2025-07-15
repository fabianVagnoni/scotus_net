#!/usr/bin/env python3
"""
Run SCOTUS AI model training with the optimized configuration.

This script runs the full training using the optimized hyperparameters
from the config.env file. It saves logs and the final model to the 
logs/training_logs/ directory.

Usage:
    python run_training.py --experiment-name my_experiment

The script will:
1. Create a log file: logs/training_logs/training_my_experiment_TIMESTAMP.txt
2. Save the best model: logs/training_logs/final_model_my_experiment_TIMESTAMP.pth
3. Use combined metric (Val Loss + (1-F1))/2 for model selection (same as optimization)
4. Log all training progress (epochs, losses, metrics) to a single txt file
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import shutil

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_trainer import SCOTUSModelTrainer
from models.config import config
from utils.logger import get_logger

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(experiment_name: str):
    """Set up logging to file and console using loguru."""
    # Create logs directory
    logs_dir = Path("logs/training_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"training_{experiment_name}_{timestamp}.txt"
    
    # Set the LOG_FILE environment variable so get_logger() uses this file
    os.environ["LOG_FILE"] = str(log_file)
    
    return log_file

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SCOTUS AI model with optimized configuration")
    parser.add_argument("--experiment-name", type=str, required=True,
                       help="Name for the experiment (used in log and model filenames)")
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.experiment_name)
    
    # Create logger (will use the LOG_FILE environment variable set by setup_logging)
    logger = get_logger(__name__)
    
    # Set random seeds
    set_seeds(config.random_seed)
    
    # Print configuration
    logger.info("🚀 Starting SCOTUS AI Model Training")
    logger.info(f"📝 Experiment: {args.experiment_name}")
    logger.info(f"📋 Log file: {log_file}")
    logger.info("=" * 80)
    config.print_config()
    logger.info("=" * 80)
    
    # Create trainer
    trainer = SCOTUSModelTrainer()
    
    # Check if required files exist
    required_files = [
        config.dataset_file,
        config.bio_tokenized_file,
        config.description_tokenized_file
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file_path in missing_files:
            logger.error(f"   - {file_path}")
        logger.error("Please ensure all required data files are generated before training.")
        sys.exit(1)
    
    # Create output directory in training_logs
    output_dir = Path("logs/training_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also ensure the config's model output directory exists
    config_output_dir = Path(config.model_output_dir)
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Train model
        logger.info("🎓 Starting model training...")
        model = trainer.train_model()
        
        # Save final model (best model is already saved by trainer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = output_dir / f"final_model_{args.experiment_name}_{timestamp}.pth"
        
        # Copy the best model to the training_logs directory
        best_model_path = Path(config.model_output_dir) / config.best_model_name
        if best_model_path.exists():
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"💾 Final model (best validation) saved to: {final_model_path}")
        else:
            # Fallback: save current model state
            model.save_model(str(final_model_path))
            logger.info(f"💾 Final model saved to: {final_model_path}")
        
        logger.info("✅ Training completed successfully!")
        logger.info("📊 Best model was saved based on combined metric (Val Loss + (1-F1))/2")
        logger.info("📈 Combined metric balances probabilistic accuracy with classification performance")
        logger.info("🔄 Learning rate was automatically reduced on combined metric plateau")
        logger.info(f"   - LR reduction factor: {config.lr_scheduler_factor}")
        logger.info(f"   - LR reduction patience: {config.lr_scheduler_patience} epochs")
        logger.info(f"   - Metric: Combined metric (Val Loss + (1-F1))/2 - same as model selection")
        
        # Optionally evaluate on holdout test set
        logger.info("🔍 Evaluating on holdout test set...")
        try:
            holdout_results = trainer.evaluate_on_holdout_test_set()
            logger.info("🎯 Holdout test set evaluation results:")
            logger.info(f"   - Test Loss: {holdout_results['holdout_loss']:.4f}")
            logger.info(f"   - Test Combined Metric (Val Loss + (1-F1))/2: {holdout_results['holdout_combined_metric']:.4f}")
            logger.info(f"   - Test Cases: {holdout_results['num_holdout_cases']}")
            logger.info(f"   - Loss Function: {holdout_results['loss_function']}")
        except Exception as e:
            logger.warning(f"⚠️ Could not evaluate on holdout test set: {e}")
            logger.info("This is normal if no holdout test set is configured.")
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎯 TRAINING SUMMARY")
        logger.info(f"📝 Experiment: {args.experiment_name}")
        logger.info(f"📋 Log file: {log_file}")
        logger.info(f"💾 Model file: {final_model_path}")
        logger.info("📊 Model selection: Best combined metric (Val Loss + (1-F1))/2")
        logger.info("🔄 Learning rate scheduling: ReduceLROnPlateau (automatic reduction)")
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 