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
from ..utils.config import config


class SCOTUSModelTrainer:
    """Train SCOTUS prediction models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_case_dataset(self, dataset_file: str = None) -> Dict:
        """Load the case dataset JSON file."""
        if dataset_file is None:
            dataset_file = self.config.get('data.dataset_file', 'data/processed/case_dataset.json')
        
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
            data_path = Path(self.config.get('data.processed_data_path', 'data/processed'))
            bio_tokenized_file = str(data_path / 'encoded_bios.pkl')
            
        if not description_tokenized_file:
            data_path = Path(self.config.get('data.processed_data_path', 'data/processed'))
            description_tokenized_file = str(data_path / 'encoded_descriptions.pkl')
        
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
        Convert dataset to the format expected by SCOTUSDataset.
        
        Args:
            dataset: Dataset with case_id as keys and case data as values
            
        Returns:
            Dictionary with case_id as keys and [justice_bio_paths, case_description_path, voting_percentages] as values
        """
        prepared_dataset = {}
        
        for case_id, case_data in dataset.items():
            # Handle both old format (3 elements) and new format (4 elements with tokenized_locations)
            if len(case_data) == 3:
                justice_bio_paths, case_description_path, voting_percentages = case_data
            elif len(case_data) == 4:
                justice_bio_paths, case_description_path, voting_percentages, _ = case_data
            else:
                self.logger.warning(f"Unexpected case data format for case {case_id}: {len(case_data)} elements")
                continue
            
            # Ensure voting percentages is a list of 4 floats
            if not isinstance(voting_percentages, list) or len(voting_percentages) != 4:
                self.logger.warning(f"Invalid voting percentages for case {case_id}: {voting_percentages}")
                continue
            
            # Filter out empty/None justice bio paths
            filtered_bio_paths = [path for path in justice_bio_paths if path and path.strip()]
            
            # Skip cases without case description
            if not case_description_path or not case_description_path.strip():
                self.logger.warning(f"No case description for case {case_id}")
                continue
            
            prepared_dataset[case_id] = [filtered_bio_paths, case_description_path, voting_percentages]
        
        self.logger.info(f"Prepared {len(prepared_dataset)} cases for training")
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
        bio_model_name = self.config.get('model.bio_model_name', 'all-MiniLM-L6-v2')
        description_model_name = self.config.get('model.description_model_name', 'Stern5497/sbert-legal-xlm-roberta-base')
        embedding_dim = self.config.get('model.embedding_dim', 384)
        hidden_dim = self.config.get('model.hidden_dim', 512)
        dropout_rate = self.config.get('model.dropout_rate', 0.1)
        max_justices = self.config.get('model.max_justices', 15)
        num_attention_heads = self.config.get('model.num_attention_heads', 4)
        use_justice_attention = self.config.get('model.use_justice_attention', True)
        
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
        batch_size = self.config.get('training.batch_size', 4)  # Smaller batch size for memory
        num_workers = self.config.get('training.num_workers', 0)  # Start with 0 to avoid multiprocessing issues
        
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
        learning_rate = self.config.get('training.learning_rate', 1e-4)
        num_epochs = self.config.get('training.num_epochs', 10)
        weight_decay = self.config.get('training.weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.KLDivLoss(reduction='batchmean')  # Using KL divergence for voting distribution
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('training.patience', 5)
        
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
                    predictions_tensor = F.log_softmax(predictions_tensor, dim=1)
                    loss = criterion(predictions_tensor, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                    
                    # Log progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
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
                
                model_output_dir = Path(self.config.get('model.output_dir', './models_output'))
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
                    loss = criterion(predictions_tensor, batch_targets)
                    
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