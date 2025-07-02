"""Model training for SCOTUS outcome prediction."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from .scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
from ..utils.logger import get_logger
from ..utils.holdout_test_set import HoldoutTestSetManager
from .config import config
from .losses import create_scotus_loss_function


class SCOTUSModelTrainer:
    """Train SCOTUS prediction models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.holdout_manager = HoldoutTestSetManager()
        
    def load_case_dataset(self, dataset_file: str = None) -> Dict:
        """Load the case dataset JSON file."""
        if dataset_file is None:
            dataset_file = self.config.dataset_file
        
        if not Path(dataset_file).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        self.logger.info(f"Loading case dataset from: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded dataset with {len(dataset)} cases")
        
        # Filter out holdout test cases
        dataset = self.holdout_manager.filter_dataset_exclude_holdout(dataset)
        
        return dataset
    
    def split_dataset(self, dataset: Dict, train_ratio: float = 0.85, 
                     val_ratio: float = 0.15, random_state: int = 42) -> Tuple[Dict, Dict]:
        """Split the dataset into train and validation sets only (holdout test set is separate)."""
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        case_ids = list(dataset.keys())
        
        # Split into train and validation only
        train_ids, val_ids = train_test_split(
            case_ids, 
            test_size=val_ratio, 
            random_state=random_state
        )
        
        # Create dataset splits
        train_dataset = {case_id: dataset[case_id] for case_id in train_ids}
        val_dataset = {case_id: dataset[case_id] for case_id in val_ids}
        
        self.logger.info(f"Dataset split - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)} (Holdout test set managed separately)")
        
        return train_dataset, val_dataset
    
    def get_tokenized_file_paths(self, dataset: Dict) -> Tuple[str, str]:
        """Extract tokenized file paths from the dataset."""
        
        # Look for tokenized file paths in the dataset
        bio_tokenized_file = None
        description_tokenized_file = None
        
        # Check if dataset has been updated with tokenized locations (4th element)
        for case_id, case_data in dataset.items():
            if len(case_data) == 4:
                tokenized_locations = case_data[3]
                if isinstance(tokenized_locations, dict):
                    bio_tokenized_file = tokenized_locations.get('bio_tokenized_file')
                    description_tokenized_file = tokenized_locations.get('description_tokenized_file')
                    break
        
        # Fall back to default paths if not found in dataset
        if not bio_tokenized_file:
            bio_tokenized_file = self.config.bio_tokenized_file
            
        if not description_tokenized_file:
            description_tokenized_file = self.config.description_tokenized_file
        
        # Verify files exist
        if not Path(bio_tokenized_file).exists():
            raise FileNotFoundError(f"Biography tokenized file not found: {bio_tokenized_file}")
        if not Path(description_tokenized_file).exists():
            raise FileNotFoundError(f"Description tokenized file not found: {description_tokenized_file}")
        
        self.logger.info(f"Using tokenized files - Bios: {bio_tokenized_file}, "
                        f"Descriptions: {description_tokenized_file}")
        
        return bio_tokenized_file, description_tokenized_file
    
    def prepare_dataset_dict(self, dataset: Dict) -> Dict[str, List]:
        """
        Convert dataset to the format expected by SCOTUSDataset, skipping invalid entries.
        
        Args:
            dataset: Dataset with case_id as keys and case data as values
            
        Returns:
            Dictionary with case_id as keys and [justice_bio_paths, case_description_path, voting_percentages] as values
        """
        prepared_dataset = {}
        skipped_missing_desc = 0
        skipped_missing_bio = 0
        skipped_bad_format = 0
        skipped_bad_target = 0

        for case_id, case_data in dataset.items():
            # Handle both old format (3 elements) and new format (4 elements with tokenized_locations)
            if len(case_data) == 3:
                justice_bio_paths, case_description_path, voting_percentages = case_data
            elif len(case_data) == 4:
                justice_bio_paths, case_description_path, voting_percentages, _ = case_data
            else:
                self.logger.warning(f"Skipping case {case_id}: Unexpected case data format with {len(case_data)} elements.")
                skipped_bad_format += 1
                continue
            
            # Ensure voting percentages is a list of 3 floats
            if not isinstance(voting_percentages, list) or len(voting_percentages) != 3:
                self.logger.warning(f"Skipping case {case_id}: Invalid voting percentages format: {voting_percentages}")
                skipped_bad_target += 1
                continue

            # Skip cases without case description path
            if not case_description_path or not case_description_path.strip():
                self.logger.warning(f"Skipping case {case_id}: Missing case description path.")
                skipped_missing_desc += 1
                continue
            
            # Filter out empty/None justice bio paths and skip if none are left
            filtered_bio_paths = [path for path in justice_bio_paths if path and path.strip()]
            if not filtered_bio_paths:
                self.logger.warning(f"Skipping case {case_id}: No valid justice biography paths provided.")
                skipped_missing_bio += 1
                continue
            
            prepared_dataset[case_id] = [filtered_bio_paths, case_description_path, voting_percentages]
        
        total_cases = len(dataset)
        total_skipped = skipped_missing_desc + skipped_missing_bio + skipped_bad_format + skipped_bad_target
        self.logger.info(f"Dataset preparation summary for {total_cases} initial cases:")
        self.logger.info(f"  - Prepared {len(prepared_dataset)} cases for use.")
        self.logger.info(f"  - Skipped {total_skipped} cases in total:")
        self.logger.info(f"    - {skipped_missing_desc} with missing descriptions.")
        self.logger.info(f"    - {skipped_missing_bio} with no valid justice bios.")
        self.logger.info(f"    - {skipped_bad_target} with invalid target format.")
        self.logger.info(f"    - {skipped_bad_format} with unexpected data format.")
        
        return prepared_dataset
        
    def train_model(self, dataset_file: str = None):
        """Train the SCOTUS prediction model."""
        self.logger.info("Starting model training")
        
        # Load dataset
        dataset = self.load_case_dataset(dataset_file)
        
        # Get tokenized file paths
        bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths(dataset)
        
        # Split dataset (test set is managed separately as holdout)
        train_dataset, val_dataset = self.split_dataset(dataset)
        
        # Model configuration
        bio_model_name = self.config.bio_model_name
        description_model_name = self.config.description_model_name
        embedding_dim = self.config.embedding_dim
        hidden_dim = self.config.hidden_dim
        dropout_rate = self.config.dropout_rate
        max_justices = self.config.max_justices
        num_attention_heads = self.config.num_attention_heads
        use_justice_attention = self.config.use_justice_attention
        
        # Initialize model
        model = SCOTUSVotingModel(
            bio_tokenized_file=bio_tokenized_file,
            description_tokenized_file=description_tokenized_file,
            bio_model_name=bio_model_name,
            description_model_name=description_model_name,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_justices=max_justices,
            num_attention_heads=num_attention_heads,
            use_justice_attention=use_justice_attention,
            device=str(self.device)
        )
        
        model.to(self.device)
        
        # Log model statistics
        stats = model.get_tokenized_stats()
        self.logger.info(f"Model loaded with {stats['bio_tokenized_loaded']} biography "
                        f"and {stats['description_tokenized_loaded']} description tokenizations")
        
        # Prepare datasets
        train_dataset_dict = self.prepare_dataset_dict(train_dataset)
        val_dataset_dict = self.prepare_dataset_dict(val_dataset)
        
        if not train_dataset_dict:
            raise ValueError("No valid training cases found")
        if not val_dataset_dict:
            raise ValueError("No valid validation cases found")
        
        train_pytorch_dataset = SCOTUSDataset(train_dataset_dict)
        val_pytorch_dataset = SCOTUSDataset(val_dataset_dict)
        
        # Data loaders
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers
        
        train_loader = DataLoader(
            train_pytorch_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_pytorch_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        # Training setup
        learning_rate = self.config.learning_rate
        num_epochs = self.config.num_epochs
        weight_decay = self.config.weight_decay
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup sentence transformer fine-tuning if enabled
        sentence_transformer_optimizer = None
        if self.config.enable_sentence_transformer_finetuning:
            self.logger.info(f"Sentence transformer fine-tuning enabled. Will unfreeze at epoch {self.config.unfreeze_sentence_transformers_epoch}")
            # Create separate optimizer for sentence transformers (will be activated when unfrozen)
            sentence_transformer_params = []
            if self.config.unfreeze_bio_model:
                sentence_transformer_params.extend(model.bio_model.parameters())
            if self.config.unfreeze_description_model:
                sentence_transformer_params.extend(model.description_model.parameters())
            
            if sentence_transformer_params:
                sentence_transformer_optimizer = torch.optim.AdamW(
                    sentence_transformer_params, 
                    lr=self.config.sentence_transformer_learning_rate, 
                    weight_decay=weight_decay
                )
        
        # Setup loss function using the new modular system
        loss_function = self.config.loss_function
        criterion = create_scotus_loss_function(loss_function, self.config)
        
        self.logger.info(f"Using loss function: {loss_function}")
        loss_config = criterion.get_config()
        self.logger.info(f"Loss configuration: {loss_config}")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config.lr_scheduler_factor, 
            patience=self.config.lr_scheduler_patience
        )
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.patience
        
        for epoch in range(num_epochs):
            # Check if we should unfreeze sentence transformers at this epoch
            if (self.config.enable_sentence_transformer_finetuning and 
                self.config.unfreeze_sentence_transformers_epoch == epoch and
                sentence_transformer_optimizer is not None):
                
                self.logger.info(f"🔓 Unfreezing sentence transformers at epoch {epoch + 1}")
                model.unfreeze_models_selectively(
                    unfreeze_bio=self.config.unfreeze_bio_model,
                    unfreeze_description=self.config.unfreeze_description_model
                )
                
                # Log the status
                status = model.get_sentence_transformer_status()
                self.logger.info(f"   Bio model trainable: {status['bio_model_trainable']}")
                self.logger.info(f"   Description model trainable: {status['description_model_trainable']}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    if sentence_transformer_optimizer is not None:
                        sentence_transformer_optimizer.zero_grad()
                    
                    batch_predictions = []
                    batch_targets = batch['targets'].to(self.device)
                    
                    # Process each sample in the batch
                    for i in range(len(batch['case_ids'])):
                        case_description_path = batch['case_description_paths'][i]
                        justice_bio_paths = batch['justice_bio_paths'][i]
                        
                        # Forward pass
                        prediction = model(case_description_path, justice_bio_paths)
                        batch_predictions.append(prediction)
                    
                    # Stack predictions and compute loss
                    predictions_tensor = torch.stack(batch_predictions)
                    
                    # Compute loss using modular loss system (pass epoch for annealing)
                    loss = criterion(predictions_tensor, batch_targets, epoch=epoch)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
                    
                    optimizer.step()
                    if sentence_transformer_optimizer is not None:
                        # Check if sentence transformers are actually unfrozen before stepping
                        status = model.get_sentence_transformer_status()
                        if status['any_trainable']:
                            sentence_transformer_optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                    
                    # Log progress based on config frequency
                    if (batch_idx + 1) % self.config.log_frequency == 0:
                        self.logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                                       f"Batch {batch_idx+1}/{len(train_loader)}, "
                                       f"Loss: {loss.item():.4f}")
                
                except Exception as e:
                    self.logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            if num_batches == 0:
                self.logger.error("No successful training batches in this epoch")
                break
                
            avg_train_loss = train_loss / num_batches
            
            # Validation phase
            val_loss = self.evaluate_model(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_output_dir = Path(self.config.model_output_dir)
                model_output_dir.mkdir(exist_ok=True)
                model.save_model(str(model_output_dir / 'best_model.pth'))
                self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info("Training completed")
        self.logger.info("Use evaluate_on_holdout_test_set() method for final evaluation on holdout test set")
        
        return model
    
    def evaluate_model(self, model: SCOTUSVotingModel, data_loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate the model on validation/test data."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch_predictions = []
                    batch_targets = batch['targets'].to(self.device)
                    
                    # Process each sample in the batch
                    for i in range(len(batch['case_ids'])):
                        case_description_path = batch['case_description_paths'][i]
                        justice_bio_paths = batch['justice_bio_paths'][i]
                        
                        # Forward pass
                        prediction = model(case_description_path, justice_bio_paths)
                        batch_predictions.append(prediction)
                    
                    # Stack predictions and compute loss
                    predictions_tensor = torch.stack(batch_predictions)
                    
                    # Compute loss using modular loss system
                    loss = criterion(predictions_tensor, batch_targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def evaluate_on_holdout_test_set(self, model_path: str = None) -> Dict:
        """
        Evaluate a trained model on the holdout test set.
        
        Args:
            model_path: Path to saved model (defaults to best model)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_path is None:
            model_output_dir = Path(self.config.model_output_dir)
            model_path = str(model_output_dir / 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Evaluating model on holdout test set: {model_path}")
        
        # Load full dataset (without filtering holdout cases)
        dataset_file = self.config.dataset_file
        with open(dataset_file, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        
        # Get holdout test cases
        holdout_dataset = self.holdout_manager.get_holdout_dataset(full_dataset)
        
        if not holdout_dataset:
            raise ValueError("No holdout test cases found")
        
        # Get tokenized file paths from the full dataset
        bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths(full_dataset)
        
        # Load model
        model = SCOTUSVotingModel(
            bio_tokenized_file=bio_tokenized_file,
            description_tokenized_file=description_tokenized_file,
            bio_model_name=self.config.bio_model_name,
            description_model_name=self.config.description_model_name,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate,
            max_justices=self.config.max_justices,
            num_attention_heads=self.config.num_attention_heads,
            use_justice_attention=self.config.use_justice_attention,
            device=str(self.device)
        )
        
        # Load trained weights
        model.load_model(model_path)
        model.to(self.device)
        model.eval()
        
        # Prepare holdout dataset
        holdout_dataset_dict = self.prepare_dataset_dict(holdout_dataset)
        
        if not holdout_dataset_dict:
            raise ValueError("No valid holdout test cases found")
        
        holdout_pytorch_dataset = SCOTUSDataset(holdout_dataset_dict)
        
        # Create data loader
        holdout_loader = DataLoader(
            holdout_pytorch_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        
        # Setup loss function using the new modular system
        loss_function = self.config.loss_function
        criterion = create_scotus_loss_function(loss_function, self.config)
        
        # Evaluate
        holdout_loss = self.evaluate_model(model, holdout_loader, criterion)
        
        self.logger.info(f"Holdout test set evaluation completed")
        self.logger.info(f"Holdout test loss: {holdout_loss:.4f}")
        
        # Additional metrics could be computed here
        results = {
            'holdout_loss': holdout_loss,
            'num_holdout_cases': len(holdout_dataset_dict),
            'loss_function': loss_function,
            'model_path': model_path
        }
        
        return results


if __name__ == "__main__":
    trainer = SCOTUSModelTrainer()
    trainer.train_model() 