import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Any
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    from scripts.pretraining.loss import ContrastiveLoss
    from scripts.pretraining.constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from scripts.utils.logger import get_logger
    from scripts.utils.holdout_test_set import HoldoutTestSetManager
except ImportError:
    # Fallback for when running as module from root
    from .loss import ContrastiveLoss
    from .constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from ..utils.logger import get_logger
    from ..utils.holdout_test_set import HoldoutTestSetManager


class ContrastiveJusticeTrainer:
    def __init__(self, config):
        self.logger = get_logger(__name__)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_pretraining_dataset(self, pretraining_dataset_file: str) -> Dict:
        """Load the pretraining dataset."""
        with open(pretraining_dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def split_pretraining_dataset(self, pretraining_dataset: Dict) -> Tuple[Dict, Dict]:
        """Split the pretraining dataset into train and validation sets."""
        # Create a list of (justice_name, year) tuples and sort by year
        justice_year_pairs = []
        for justice_name, data in pretraining_dataset.items():
            year = int(data[0]) if data[0] is not None else 0
            justice_year_pairs.append((justice_name, year))
        
        # Sort by year and extract just the justice names
        ordered_justices = [justice for justice, year in sorted(justice_year_pairs, key=lambda x: x[1])]
        
        test_set = ordered_justices[:self.config.test_set_size]
        val_set = ordered_justices[self.config.test_set_size:self.config.test_set_size + self.config.val_set_size]
        train_set = ordered_justices[self.config.test_set_size + self.config.val_set_size:]
        return train_set, val_set, test_set

    def evaluate_model(self, model: ContrastiveJustice, val_loader: DataLoader, loss_fn: ContrastiveLoss) -> float:
        """Evaluate the model on validation data."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch_trunc_bio_data = batch['trunc_bio_data']
                    batch_full_bio_data = batch['full_bio_data']
                    
                    e_t, e_f = model.forward(batch_trunc_bio_data, batch_full_bio_data)
                    batch_loss = loss_fn(e_t, e_f)
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        if num_batches == 0:
            return float('inf')
        
        return total_loss / num_batches

    def train_model(self, justices_file: str = None):
        """Train the contrastive justice model."""
        
        # Load pretraining dataset
        pretraining_dataset = self.load_pretraining_dataset(self.config.pretraining_dataset_file)
        train_justices, val_justices, test_justices = self.split_pretraining_dataset(pretraining_dataset)

        # Load justices data & Save Directory
        model_output_dir = Path(self.config.model_output_dir)

        # Model configuration
        model_name = self.config.model_name
        dropout_rate = self.config.dropout_rate

        # Initialize model
        model = ContrastiveJustice(
            trunc_bio_tokenized_file=self.config.trunc_bio_tokenized_file,
            full_bio_tokenized_file=self.config.full_bio_tokenized_file,
            model_name=model_name,
            dropout_rate=dropout_rate
        )
        model.to(self.device)

        # Prepare datasets
        training_dataset = ContrastiveJusticeDataset(train_justices, self.config.trunc_bio_tokenized_file, self.config.full_bio_tokenized_file)
        validation_dataset = ContrastiveJusticeDataset(val_justices, self.config.trunc_bio_tokenized_file, self.config.full_bio_tokenized_file)
        test_dataset = ContrastiveJusticeDataset(test_justices, self.config.trunc_bio_tokenized_file, self.config.full_bio_tokenized_file)

        # Data loaders
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers
        train_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        # Training setup
        learning_rate = self.config.learning_rate
        num_epochs = self.config.num_epochs
        weight_decay = self.config.weight_decay
        temperature = self.config.temperature   
        alpha = self.config.alpha

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = ContrastiveLoss(temperature=temperature, alpha=alpha)
        loss_fn.to(self.device)
        self.logger.info(f"Created contrastive loss function with temperature {self.config.temperature} and alpha {self.config.alpha}")

        # Learning rate scheduler
        lr_scheduler_factor = self.config.lr_scheduler_factor
        lr_scheduler_patience = self.config.lr_scheduler_patience   
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience
        )

        # Training loop
        self.logger.info(f"Starting training for {num_epochs} epochs")
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.max_patience
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    batch_trunc_bio_data = batch['trunc_bio_data']
                    batch_full_bio_data = batch['full_bio_data']
                    
                    # Use batch processing for efficiency
                    e_t, e_f = model.forward(batch_trunc_bio_data, batch_full_bio_data)
                    batch_loss = loss_fn(e_t, e_f)
                    train_loss += batch_loss.item()
                    num_batches += 1
                    batch_loss.backward()
                    optimizer.step()
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            if num_batches == 0:
                self.logger.error("No successful training batches in this epoch")
                break
            
            avg_train_loss = train_loss / num_batches
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            self.logger.info(f"Training Loss: {avg_train_loss:.4f}")
            val_loss = self.evaluate_model(model, val_loader, loss_fn)
            self.logger.info(f"Validation Loss: {val_loss:.4f}")

            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # Check if learning rate was reduced and log it
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                self.logger.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                patience_counter = 0
                best_val_loss = val_loss
                model_output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(model_output_dir / 'best_model.pth'))
                self.logger.info(f"New best model saved with validation loss {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs with no improvement")
                    break

        # Load best model if available
        best_model_path = model_output_dir / 'best_model.pth'
        if best_model_path.exists():
            model.load_state_dict(torch.load(str(best_model_path)))
            self.logger.info(f"Loaded best model with validation loss {best_val_loss:.4f}")
        
        self.logger.info("Training completed successfully!")
        return model
        