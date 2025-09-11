#!/usr/bin/env python3
"""
Simplified training runner for the Baseline SCOTUS model.

Draws inspiration from scripts/models/run_training.py but targets the
single-pipeline baseline that consumes only case descriptions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import random
import numpy as np

# Add the project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.baseline.baseline_trainer import BaselineTrainer
from scripts.baseline.config import BaselineConfig
from scripts.utils.logger import get_logger


def set_seeds(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(experiment_name: str) -> None:
    """Set up logging for the training run."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"training_baseline_{experiment_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline SCOTUS Model")
    parser.add_argument("--dataset-file", type=str, help="Path to the case dataset JSON file")
    parser.add_argument("--config-file", type=str, help="Path to baseline config.env file")
    parser.add_argument("--experiment-name", type=str, default="baseline_training", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device")
    parser.add_argument("--eval-holdout", action='store_true', help="Evaluate on holdout test set after training")
    args = parser.parse_args()

    # Logging and seeds
    setup_logging(args.experiment_name)
    logger = get_logger(__name__)
    set_seeds(args.seed)
    logger.info(f"🎲 Set random seed to {args.seed}")

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"🖥️ Using device: {device}")

    # Load configuration (env-based)
    config = BaselineConfig(config_file=args.config_file) if args.config_file else BaselineConfig()
    logger.info("🚀 Starting Baseline SCOTUS Model Training")
    logger.info(f"   Experiment: {args.experiment_name}")
    logger.info(f"   Dataset: {args.dataset_file or config.dataset_file}")
    logger.info(f"   Description Model: {config.description_model_name}")
    logger.info(f"   Embedding Dim: {config.embedding_dim}")
    logger.info(f"   Hidden Dim: {config.hidden_dim}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Epochs: {config.num_epochs}")
    logger.info(f"   Year filter: [{config.year_start}, {config.year_end}]")

    try:
        # Initialize trainer with config values, allowing dataset override
        trainer = BaselineTrainer(
            dataset_file=(args.dataset_file or config.dataset_file),
            description_tokenized_file=config.description_tokenized_file,
            description_model_name=config.description_model_name,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_epochs=config.num_epochs,
            patience=config.patience,
            device=device,
            year_start=config.year_start,
            year_end=config.year_end,
        )

        # Train
        results = trainer.train()
        logger.info("✅ Training completed successfully!")
        logger.info(f"   Best val loss: {float(results.get('best_val_loss', float('inf'))):.4f}")

        # Optional holdout evaluation
        if args.eval_holdout:
            logger.info("🧪 Evaluating on holdout test set...")
            try:
                test_results = trainer.evaluate_on_holdout_test_set()
                logger.info(f"   Holdout Test Results: {test_results}")
            except Exception as e:
                logger.warning(f"   Holdout evaluation failed: {str(e)}")

        logger.info("💾 Baseline model saved to: models_output/baseline/best_model.pth")
        logger.info(f"📋 Training logs saved to: logs/training_baseline_{args.experiment_name}.log")

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


