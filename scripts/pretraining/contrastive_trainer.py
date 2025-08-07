import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Any
import sys
import torch.nn.functional as F
from transformers import AutoModel

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    from scripts.pretraining.loss import ContrastiveLoss
    from scripts.pretraining.constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from scripts.utils.logger import get_logger
    from scripts.utils.holdout_test_set import HoldoutTestSetManager
    from scripts.utils.metrics import calculate_mrr
    from scripts.utils.progress import get_progress_bar
except ImportError:
    # Fallback for when running as module from root
    from .loss import ContrastiveLoss
    from .constrastive_justice import ContrastiveJustice, ContrastiveJusticeDataset, collate_fn
    from ..utils.logger import get_logger
    from ..utils.holdout_test_set import HoldoutTestSetManager
    from ..utils.metrics import calculate_mrr
    from ..utils.progress import get_progress_bar


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
        
        test_set = ordered_justices[-self.config.test_set_size:]
        val_set = ordered_justices[-self.config.test_set_size - self.config.val_set_size:-self.config.test_set_size]
        train_set = ordered_justices[:-self.config.test_set_size - self.config.val_set_size]

        # Logging Split
        self.logger.info(f"Val Justices: {val_set}")
        self.logger.info(f"Test Justices: {test_set}")

        return train_set, val_set, test_set

    def evaluate_model(self, model: ContrastiveJustice, val_loader: DataLoader, loss_fn: ContrastiveLoss) -> Tuple[float, float]:
        """Evaluate the model on validation data."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_trunc_embeddings = []
        all_full_embeddings = []
        justice_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    batch_trunc_bio_data = batch['trunc_bio_data']
                    batch_full_bio_data = batch['full_bio_data']
                    
                    e_t, e_f = model.forward(batch_trunc_bio_data, batch_full_bio_data)
                    batch_loss = loss_fn(e_t, e_f)

                    # Normalize embeddings for cosine similarity
                    e_t = F.normalize(e_t, dim=-1)
                    e_f = F.normalize(e_f, dim=-1)
                    all_trunc_embeddings.append(e_t)
                    all_full_embeddings.append(e_f)
                    # Track which examples these are (for correct matching)
                    batch_size = e_t.size(0)
                    start_idx = batch_idx * val_loader.batch_size
                    justice_indices.extend(range(start_idx, start_idx + batch_size))
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        mrr = calculate_mrr(all_trunc_embeddings, all_full_embeddings)
        
        if num_batches == 0:
            return float('inf'), 0.0
        
        return total_loss / num_batches, mrr

    def train_model(self, justices_file: str = None):
        """Train the contrastive justice model."""
        
        # Load pretraining dataset
        pretraining_dataset = self.load_pretraining_dataset(self.config.pretraining_dataset_file)
        train_justices, val_justices, test_justices = self.split_pretraining_dataset(pretraining_dataset)

        # Load justices data & Save Directory
        # Use MODEL_OUTPUT_DIR from configuration
        model_output_dir = Path(self.config.model_output_dir)
        
        # Create the directory structure if it doesn't exist
        model_output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(model_output_dir, os.W_OK):
            self.logger.error(f"Cannot write to {model_output_dir}")
            raise PermissionError(f"Cannot write to {model_output_dir}")
        self.logger.info(f"Model will be saved to: {model_output_dir.absolute()}")

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
        
        # Progress bar for epochs
        epoch_pbar = get_progress_bar(
            range(num_epochs),
            desc="Training Progress",
            total=num_epochs
        )
        
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Progress bar for batches within each epoch
            batch_pbar = get_progress_bar(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                total=len(train_loader)
            )
            
            for batch in batch_pbar:
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
                    
                    # Update batch progress bar with current loss
                    batch_pbar.set_description(
                        f"Epoch {epoch+1}/{num_epochs} - Loss: {batch_loss.item():.4f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error in training batch: {e}")
                    continue
            
            if num_batches == 0:
                self.logger.error("No successful training batches in this epoch")
                break
            
            avg_train_loss = train_loss / num_batches
            
            # Update epoch progress bar with training loss
            epoch_pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}"
            )
            
            # Validation
            val_loss, val_mrr = self.evaluate_model(model, val_loader, loss_fn)
            
            # Update epoch progress bar with validation metrics
            epoch_pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, MRR: {val_mrr:.4f}"
            )
            
            # Log detailed information
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            self.logger.info(f"Training Loss: {avg_train_loss:.4f}")
            self.logger.info(f"Validation Loss: {val_loss:.4f}")
            self.logger.info(f"Validation MRR: {val_mrr:.4f}")

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
                model.truncated_bio_model.save_pretrained(str(model_output_dir / 'best_model.pth'))
                self.logger.info(f"New best model saved with validation loss {val_loss:.4f} in location {(model_output_dir / 'best_model.pth').absolute()}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs with no improvement")
                    break

        # Load best model if available
        best_model_path = model_output_dir / 'best_model.pth'
        if best_model_path.exists():
            model.truncated_bio_model = AutoModel.from_pretrained(str(best_model_path))
            self.logger.info(f"Loaded best model with validation loss {best_val_loss:.4f}")
        
        self.logger.info("Training completed successfully!")
        return model
        