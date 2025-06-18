"""Model training for SCOTUS outcome prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

from ..utils.logger import get_logger
from ..utils.config import config


class SCOTUSModelTrainer:
    """Train SCOTUS prediction models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = config
        
    def load_data(self):
        """Load processed training data."""
        data_path = Path(self.config.get('data.processed_data_path', 'data/processed'))
        
        train_df = pd.read_parquet(data_path / 'train.parquet')
        val_df = pd.read_parquet(data_path / 'val.parquet') 
        test_df = pd.read_parquet(data_path / 'test.parquet')
        
        return train_df, val_df, test_df
        
    def train_model(self):
        """Train the SCOTUS prediction model."""
        self.logger.info("Starting model training")
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Setup model
        model_name = self.config.get('model.name', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Prepare training data
        train_texts = [f"{row['clean_title']} {row['clean_opinion']}" for _, row in train_df.iterrows()]
        train_labels = train_df['outcome_encoded'].tolist()
        
        # Tokenize
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'], 
            'labels': train_labels
        })
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models_output',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained('./models_output')
        
        self.logger.info("Training completed")


if __name__ == "__main__":
    trainer = SCOTUSModelTrainer()
    trainer.train_model() 