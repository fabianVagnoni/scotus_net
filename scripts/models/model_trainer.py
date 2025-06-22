"""Model training for SCOTUS outcome prediction."""

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
from .config import config


class SCOTUSModelTrainer:
    """Train SCOTUS prediction models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        return dataset
    
    def split_dataset(self, dataset: Dict, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15, 
                     random_state: int = 42) -> Tuple[Dict, Dict, Dict]:
        """Split the dataset into train, validation, and test sets."""
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        case_ids = list(dataset.keys())
        
        # First split: separate train from (val + test)
        train_ids, temp_ids = train_test_split(
            case_ids, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state
        )
        
        # Second split: separate val from test
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Create dataset splits
        train_dataset = {case_id: dataset[case_id] for case_id in train_ids}
        val_dataset = {case_id: dataset[case_id] for case_id in val_ids}
        test_dataset = {case_id: dataset[case_id] for case_id in test_ids}
        
        self.logger.info(f"Dataset split - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
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
            
            # Ensure voting percentages is a list of 4 floats
            if not isinstance(voting_percentages, list) or len(voting_percentages) != 4:
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
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
        
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
        
        # Setup loss function based on configuration
        loss_function = self.config.loss_function
        if loss_function == 'kl_div':
            kl_reduction = self.config.kl_reduction
            criterion = nn.KLDivLoss(reduction=kl_reduction)
        elif loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        self.logger.info(f"Using loss function: {loss_function}")
        if loss_function == 'kl_div':
            self.logger.info(f"KL divergence reduction: {kl_reduction}")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config.lr_scheduler_factor, 
            patience=self.config.lr_scheduler_patience, verbose=True
        )
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.patience
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    
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
                    
                    # Apply appropriate transformations based on loss function
                    if loss_function == 'kl_div':
                        # Apply log_softmax to predictions for KL divergence
                        log_predictions = F.log_softmax(predictions_tensor, dim=1)
                        loss = criterion(log_predictions, batch_targets)
                    elif loss_function == 'mse':
                        # For MSE, use raw predictions and targets
                        loss = criterion(predictions_tensor, batch_targets)
                    elif loss_function == 'cross_entropy':
                        # For cross-entropy, convert targets to class indices (argmax)
                        target_classes = torch.argmax(batch_targets, dim=1)
                        loss = criterion(predictions_tensor, target_classes)
                    else:
                        raise ValueError(f"Unsupported loss function: {loss_function}")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
                    
                    optimizer.step()
                    
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
        
        # Final evaluation on test set if available
        if test_dataset:
            test_dataset_dict = self.prepare_dataset_dict(test_dataset)
            if test_dataset_dict:
                test_pytorch_dataset = SCOTUSDataset(test_dataset_dict)
                test_loader = DataLoader(
                    test_pytorch_dataset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    collate_fn=collate_fn,
                    num_workers=num_workers
                )
                
                test_loss = self.evaluate_model(model, test_loader, criterion)
                self.logger.info(f"Final test loss: {test_loss:.4f}")
        
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
                    
                    # Apply appropriate transformations based on loss function (same as training)
                    loss_function = self.config.loss_function
                    if loss_function == 'kl_div':
                        # Apply log_softmax to predictions for KL divergence
                        log_predictions = F.log_softmax(predictions_tensor, dim=1)
                        loss = criterion(log_predictions, batch_targets)
                    elif loss_function == 'mse':
                        # For MSE, use raw predictions and targets
                        loss = criterion(predictions_tensor, batch_targets)
                    elif loss_function == 'cross_entropy':
                        # For cross-entropy, convert targets to class indices (argmax)
                        target_classes = torch.argmax(batch_targets, dim=1)
                        loss = criterion(predictions_tensor, target_classes)
                    else:
                        raise ValueError(f"Unsupported loss function: {loss_function}")
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')


if __name__ == "__main__":
    trainer = SCOTUSModelTrainer()
    trainer.train_model() 