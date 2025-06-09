"""Model training module for SCOTUS outcome prediction."""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset

from ..utils.logger import get_logger
from ..utils.config import config


class SCOTUSDataset:
    """Custom dataset class for SCOTUS cases."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """Initialize dataset.
        
        Args:
            df: DataFrame with case data.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine title and opinion text
        text = f"{row['clean_title']} {row['clean_opinion']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['outcome_encoded'], dtype=torch.long)
        }


class SCOTUSTrainer:
    """SCOTUS model trainer."""
    
    def __init__(self, config_path: str = None):
        """Initialize trainer.
        
        Args:
            config_path: Path to configuration file.
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Initialize wandb if configured
        if self.config.get('logging.use_wandb', False):
            wandb.init(
                project=self.config.get('logging.wandb_project', 'scotus-ai'),
                config=self.config.config
            )
    
    def load_data(self) -> tuple:
        """Load processed training data.
        
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        data_path = Path(self.config.get('data.processed_data_path'))
        
        self.logger.info(f"Loading processed data from {data_path}")
        
        try:
            train_df = pd.read_parquet(data_path / 'train.parquet')
            val_df = pd.read_parquet(data_path / 'val.parquet')
            test_df = pd.read_parquet(data_path / 'test.parquet')
            
            self.logger.info(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        model_name = self.config.get('model.name', 'distilbert-base-uncased')
        num_labels = self.config.get('model.num_labels', 2)
        
        self.logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
        """Create HuggingFace datasets.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        max_length = self.config.get('model.max_length', 512)
        
        self.logger.info("Creating datasets")
        
        # Create datasets
        train_texts = []
        train_labels = []
        
        for _, row in train_df.iterrows():
            text = f"{row['clean_title']} {row['clean_opinion']}"
            train_texts.append(text)
            train_labels.append(row['outcome_encoded'])
        
        val_texts = []
        val_labels = []
        
        for _, row in val_df.iterrows():
            text = f"{row['clean_title']} {row['clean_opinion']}"
            val_texts.append(text)
            val_labels.append(row['outcome_encoded'])
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        
        val_encodings = self.tokenizer(
            val_texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions.
            
        Returns:
            Dictionary of metrics.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments.
        
        Returns:
            TrainingArguments object.
        """
        output_dir = Path(self.config.get('paths.model_output', 'models_output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.get('training.num_epochs', 10),
            per_device_train_batch_size=self.config.get('training.batch_size', 16),
            per_device_eval_batch_size=self.config.get('training.batch_size', 16),
            warmup_steps=self.config.get('training.warmup_steps', 500),
            weight_decay=self.config.get('model.weight_decay', 0.01),
            learning_rate=self.config.get('model.learning_rate', 2e-5),
            logging_dir=str(Path(self.config.get('paths.logs', 'logs'))),
            logging_steps=self.config.get('training.logging_steps', 100),
            evaluation_strategy=self.config.get('training.evaluation_strategy', 'steps'),
            eval_steps=self.config.get('training.eval_steps', 500),
            save_steps=self.config.get('training.save_steps', 1000),
            load_best_model_at_end=self.config.get('training.load_best_model_at_end', True),
            metric_for_best_model=self.config.get('training.metric_for_best_model', 'f1'),
            greater_is_better=self.config.get('training.greater_is_better', True),
            save_total_limit=3,
            report_to=['wandb'] if self.config.get('logging.use_wandb', False) else [],
            run_name=f"scotus-{self.config.get('model.name', 'model')}"
        )
    
    def train(self):
        """Train the model."""
        self.logger.info("Starting model training")
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(train_df, val_df)
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        self.logger.info("Starting training")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Evaluate on test set
        self.evaluate_model(trainer, test_df)
        
        self.logger.info("Training completed")
    
    def evaluate_model(self, trainer: Trainer, test_df: pd.DataFrame):
        """Evaluate model on test set.
        
        Args:
            trainer: Trained model trainer.
            test_df: Test DataFrame.
        """
        self.logger.info("Evaluating on test set")
        
        # Create test dataset
        test_texts = []
        test_labels = []
        
        for _, row in test_df.iterrows():
            text = f"{row['clean_title']} {row['clean_opinion']}"
            test_texts.append(text)
            test_labels.append(row['outcome_encoded'])
        
        # Tokenize
        test_encodings = self.tokenizer(
            test_texts, 
            truncation=True, 
            padding=True, 
            max_length=self.config.get('model.max_length', 512),
            return_tensors='pt'
        )
        
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_labels
        })
        
        # Evaluate
        eval_results = trainer.evaluate(test_dataset)
        
        # Log results
        self.logger.info("=== Test Set Results ===")
        for key, value in eval_results.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        # Generate predictions for confusion matrix
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, pred_labels)
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        self.logger.info(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        # Log to wandb if enabled
        if self.config.get('logging.use_wandb', False):
            wandb.log({
                'test_accuracy': eval_results['eval_accuracy'],
                'test_f1': eval_results['eval_f1'],
                'test_precision': eval_results['eval_precision'],
                'test_recall': eval_results['eval_recall']
            })
    
    def save_model_info(self):
        """Save model information and configuration."""
        output_dir = Path(self.config.get('paths.model_output', 'models_output'))
        
        model_info = {
            'model_name': self.config.get('model.name'),
            'max_length': self.config.get('model.max_length'),
            'num_labels': self.config.get('model.num_labels'),
            'training_config': {
                'batch_size': self.config.get('training.batch_size'),
                'num_epochs': self.config.get('training.num_epochs'),
                'learning_rate': self.config.get('model.learning_rate'),
                'weight_decay': self.config.get('model.weight_decay')
            }
        }
        
        import json
        with open(output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SCOTUS outcome prediction model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SCOTUSTrainer(args.config)
    
    # Train model
    trainer.train()
    
    # Save model info
    trainer.save_model_info()


if __name__ == "__main__":
    main() 