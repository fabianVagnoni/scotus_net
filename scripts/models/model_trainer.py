"""
Simplified SCOTUS Model Trainer.

This trainer handles:
1. Data loading and preprocessing from tokenized files
2. Training with only complete unfreezing options
3. Only KL-Div loss
4. Simplified training loop
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import os
import pickle
from typing import Dict, List, Tuple, Any
import sys
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# (deprecated):
# from torch.cuda.amp import autocast, GradScaler
# self.scaler = GradScaler()
# with autocast():
# (recommended):
from torch.amp import autocast, GradScaler

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from scripts.models.scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
    from scripts.models.losses import create_scotus_loss_function
    from scripts.utils.logger import get_logger
    from scripts.utils.holdout_test_set import HoldoutTestSetManager
    from scripts.utils.progress import get_progress_bar
    from scripts.models.config import ModelConfig
    from scripts.utils.metrics import calculate_f1_macro
except ImportError:
    # Fallback for when running as module from root
    from .scotus_voting_model import SCOTUSVotingModel, SCOTUSDataset, collate_fn
    from .losses import create_scotus_loss_function
    from ..utils.logger import get_logger
    from ..utils.holdout_test_set import HoldoutTestSetManager
    from ..utils.progress import get_progress_bar
    from .config import ModelConfig
    from ..utils.metrics import calculate_f1_macro

# Log year ranges for train/val/test
def _range_str(case_ids, holdout_manager):
    if not case_ids:
        return "N/A"
    years = [holdout_manager.extract_year_from_case_id(cid) for cid in case_ids]
    years = [y for y in years if y > 0]
    if not years:
        return "N/A"
    return f"{min(years)}-{max(years)}"

class SCOTUSModelTrainer:
    """
    Simplified SCOTUS Model Trainer.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = ModelConfig()
        self.scaler = GradScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_string = "cuda" if torch.cuda.is_available() else "cpu"

    def load_case_dataset(self, dataset_file: str = None) -> Dict:
        """Load the case dataset."""
        if dataset_file is None:
            dataset_file = self.config.dataset_file
            
        self.logger.info(f"📂 Loading case dataset from: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        self.logger.info(f"✅ Loaded {len(dataset)} cases from dataset")
        return dataset

    def split_dataset(self, dataset: Dict, train_ratio: float = 0.85, 
                     val_ratio: float = 0.15, random_state: int = 42) -> Tuple[Dict, Dict]:
        """Split dataset into train and validation sets."""
        random.seed(random_state)
        
        case_ids = list(dataset.keys())
        random.shuffle(case_ids)
        
        n_total = len(case_ids)
        n_train = int(n_total * train_ratio)
        
        train_case_ids = case_ids[:n_train]
        val_case_ids = case_ids[n_train:]
        
        train_dataset = {case_id: dataset[case_id] for case_id in train_case_ids}
        val_dataset = {case_id: dataset[case_id] for case_id in val_case_ids}
        
        self.logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset

    def get_tokenized_file_paths(self) -> Tuple[str, str]:
        """Get paths to tokenized files."""
        # Use fixed filenames for the tokenized data
        bio_tokenized_file = os.path.join(
            self.config.bio_tokenized_file
        )
        description_tokenized_file = os.path.join(
            self.config.description_tokenized_file
        )
        
        self.logger.info(f"🔗 Bio tokenized file: {bio_tokenized_file}")
        self.logger.info(f"🔗 Description tokenized file: {description_tokenized_file}")
        
        return bio_tokenized_file, description_tokenized_file

    def load_tokenized_data(self, bio_tokenized_file: str, description_tokenized_file: str) -> Tuple[dict, dict]:
        """Load pre-tokenized data."""
        self.logger.info("📥 Loading pre-tokenized data...")
        
        with open(bio_tokenized_file, 'rb') as f:
            bio_data = pickle.load(f)
            
        with open(description_tokenized_file, 'rb') as f:
            description_data = pickle.load(f)
            
        self.logger.info(f"✅ Loaded {len(bio_data['tokenized_data'])} bio and {len(description_data['tokenized_data'])} description tokenized files")
        
        return bio_data, description_data

    def prepare_processed_data(self, dataset: Dict, bio_data: dict, description_data: dict, verbose: bool = True) -> List[Dict]:
        """Process dataset into tokenized format suitable for the model."""
        processed_data = []
        
        bio_tokenized_data = bio_data['tokenized_data']
        bio_tokenized_data = {k.split('\\')[-1]: v for k, v in bio_tokenized_data.items()}
        bio_tokenized_data = {k.split('/')[-1]: v for k, v in bio_tokenized_data.items()}
        description_tokenized_data = description_data['tokenized_data']
        description_tokenized_data = {k.split('\\')[-1]: v for k, v in description_tokenized_data.items()}
        description_tokenized_data = {k.split('/')[-1]: v for k, v in description_tokenized_data.items()}
        
        if verbose:
            self.logger.info("🔄 Processing dataset entries...")
        
        skipped_cases = []
        
        for case_id, case_data in dataset.items():
            try:
                # Extract data based on the format: [justice_bio_paths, case_description_path, voting_percentages, metadata]
                justice_bio_paths = case_data[0]
                case_description_path = case_data[1].split('\\')[-1]
                voting_percentages = case_data[2]
                case_tokenized = description_tokenized_data[case_description_path]
                
                # Process justice biographies
                justice_tokenized_list = []
                valid_justice_count = 0
                
                for bio_path in justice_bio_paths:
                    bio_path = bio_path.split('\\')[-1]
                    try:
                        bio_tokenized = bio_tokenized_data[bio_path]
                        justice_tokenized_list.append(bio_tokenized)
                        valid_justice_count += 1
                    except Exception as e:
                        self.logger.warning(f"🔗 Bio path not found for {case_id}: {bio_path}")
                        continue
                
                if valid_justice_count == 0:
                    skipped_cases.append(f"{case_id}: no valid justice biographies found")
                    continue
                
                # Create processed entry
                processed_entry = {
                    'case_id': case_id,
                    'case_input_ids': case_tokenized['input_ids'],
                    'case_attention_mask': case_tokenized['attention_mask'],
                    'justice_input_ids': torch.stack([j['input_ids'] for j in justice_tokenized_list]),
                    'justice_attention_mask': torch.stack([j['attention_mask'] for j in justice_tokenized_list]),
                    'justice_count': valid_justice_count,
                    'target': torch.tensor(voting_percentages, dtype=torch.float32)
                }
                
                processed_data.append(processed_entry)
                
            except Exception as e:
                skipped_cases.append(f"{case_id}: {str(e)}")
                continue
        
        if skipped_cases and verbose:
            self.logger.warning(f"⚠️ Skipped {len(skipped_cases)} cases due to missing data")
            if len(skipped_cases) <= 5:
                for skip_reason in skipped_cases:
                    self.logger.warning(f"   - {skip_reason}")
            else:
                for skip_reason in skipped_cases[:3]:
                    self.logger.warning(f"   - {skip_reason}")
                self.logger.warning(f"   ... and {len(skipped_cases) - 3} more")
        
        self.logger.info(f"✅ Successfully processed {len(processed_data)} cases")
        return processed_data

    def train_model(self, dataset_file: str = None):
        """Train the SCOTUS voting model."""
        
        # Load dataset
        dataset = self.load_case_dataset(dataset_file)

        # Determine holdout (test) set and earliest test year
        holdout_source_file = dataset_file if dataset_file is not None else self.config.dataset_file
        holdout_manager = HoldoutTestSetManager(dataset_file=holdout_source_file)
        test_holdout_dataset = holdout_manager.get_holdout_dataset(dataset)
        test_case_ids = list(test_holdout_dataset.keys()) if test_holdout_dataset else []
        test_years = [holdout_manager.extract_year_from_case_id(cid) for cid in test_case_ids]
        test_years = [y for y in test_years if y > 0]
        earliest_test_year = min(test_years) if test_years else float('inf')

        # Exclude holdout test set and restrict train/val to years strictly before earliest test year
        available_dataset = holdout_manager.filter_dataset_exclude_holdout(dataset)
        pre_test_dataset = {
            case_id: case_data
            for case_id, case_data in available_dataset.items()
            if holdout_manager.extract_year_from_case_id(case_id) < earliest_test_year
        } if earliest_test_year != float('inf') else available_dataset

        # Sort remaining cases by year (oldest first)
        sorted_case_ids = sorted(
            pre_test_dataset.keys(),
            key=holdout_manager.extract_year_from_case_id
        )

        # Allocate train/val within remaining data using TRAIN_RATIO and VAL_RATIO
        total_tv_ratio = self.config.train_ratio + self.config.val_ratio
        if total_tv_ratio <= 0:
            total_tv_ratio = 1.0
        train_share = self.config.train_ratio / total_tv_ratio
        num_available = len(sorted_case_ids)
        num_train = int(num_available * train_share)

        train_case_ids = sorted_case_ids[:num_train]
        val_case_ids = sorted_case_ids[num_train:]

        train_dataset = {case_id: pre_test_dataset[case_id] for case_id in train_case_ids}
        val_dataset = {case_id: pre_test_dataset[case_id] for case_id in val_case_ids}

        self.logger.info(f"📊 Time-based split (excluding holdout): {len(train_dataset)} train (oldest), {len(val_dataset)} val (newer)")

        train_range = _range_str(train_case_ids, holdout_manager)
        val_range = _range_str(val_case_ids, holdout_manager)
        test_range = _range_str(test_case_ids, holdout_manager)

        self.logger.info(f"🗓️ Train years: {train_range}")
        self.logger.info(f"🗓️ Val years:   {val_range}")
        if test_case_ids:
            self.logger.info(f"🔒 Test (holdout) years: {test_range} ({len(test_case_ids)} cases)")
        
        # Get tokenized file paths and load tokenized data
        bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths()
        bio_data, description_data = self.load_tokenized_data(bio_tokenized_file, description_tokenized_file)
        
        # Process datasets
        train_processed = self.prepare_processed_data(train_dataset, bio_data, description_data)
        val_processed = self.prepare_processed_data(val_dataset, bio_data, description_data)
        
        # Create datasets and data loaders
        train_dataset_obj = SCOTUSDataset(train_processed)
        val_dataset_obj = SCOTUSDataset(val_processed)
        
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
        
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
        
        # Initialize model
        model = SCOTUSVotingModel(
            bio_model_name=self.config.bio_model_name,
            description_model_name=self.config.description_model_name,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate,
            max_justices=self.config.max_justices,
            num_attention_heads=self.config.num_attention_heads,
            use_justice_attention=self.config.use_justice_attention,
            use_noise_reg=self.config.use_noise_reg,
            noise_reg_alpha=self.config.noise_reg_alpha,
            pretrained_bio_model=self.config.pretrained_bio_model,
            device=self.device
        )
        model.to(self.device)
        
        # Create loss function
        criterion = create_scotus_loss_function(self.config)
        
        # Create optimizer for non-transformer parameters
        main_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize sentence transformer optimizer (will be used if models are unfrozen)
        sentence_transformer_optimizer = None
        
        # Set up model output directory
        model_output_dir = Path(self.config.model_output_dir)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Starting training...")
        self.logger.info(f"   Model output directory: {model_output_dir}")
        self.logger.info(f"   Training samples: {len(train_processed)}")
        self.logger.info(f"   Validation samples: {len(val_processed)}")
        self.logger.info(f"   Batch size: {self.config.batch_size}")
        self.logger.info(f"   Learning rate: {self.config.learning_rate}")
        
        # Training loop
        num_epochs = self.config.num_epochs
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.patience
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0
            
            progress_bar = get_progress_bar(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
            )
            
            for batch in progress_bar:
                main_optimizer.zero_grad()
                if sentence_transformer_optimizer:
                    sentence_transformer_optimizer.zero_grad()
                
                # Move batch to device
                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                justice_input_ids = batch['justice_input_ids'].to(self.device)
                justice_attention_mask = batch['justice_attention_mask'].to(self.device)
                justice_counts = batch['justice_counts']
                targets = batch['targets'].to(self.device)
                
                with autocast(device_type=self.device_string):
                    # Forward pass                
                    predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                
                    # Compute loss
                    loss = criterion(predictions, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
                
                # Optimizer steps
                self.scaler.step(main_optimizer)
                if sentence_transformer_optimizer:
                    self.scaler.step(sentence_transformer_optimizer)
                self.scaler.update()
                
                total_train_loss += loss.item()
                num_train_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation phase
            val_loss = self.evaluate_model(model, val_loader, criterion)
            
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Handle unfreezing at specified epochs
            if epoch + 1 == self.config.unfreeze_at_epoch and self.config.unfreeze_transformers:
                self.logger.info("🔓 Unfreezing sentence transformers...")
                if self.config.unfreeze_bio_model:
                    model.unfreeze_bio_model()
                    self.logger.info("   - Bio model unfrozen")
                if self.config.unfreeze_description_model:
                    model.unfreeze_description_model()
                    self.logger.info("   - Description model unfrozen")
                
                # Create sentence transformer optimizer if models were unfrozen
                if model.get_sentence_transformer_status()['any_trainable']:
                    st_params = []
                    if self.config.unfreeze_bio_model:
                        st_params.extend(model.bio_model.parameters())
                    if self.config.unfreeze_description_model:
                        st_params.extend(model.description_model.parameters())
                    
                    sentence_transformer_optimizer = torch.optim.Adam(
                        st_params,
                        lr=self.config.sentence_transformer_learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                    self.logger.info(f"   - Sentence transformer optimizer created with LR: {self.config.sentence_transformer_learning_rate}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model.save_model(str(model_output_dir / 'best_model.pth'))
                self.logger.info(f"💾 New best model saved with val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info("✅ Training completed")
        self.logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")

    def evaluate_model(self, model: SCOTUSVotingModel, data_loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate model on given data loader."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Log start of validation
        self.logger.info("🔍 Starting validation...")
        
        with torch.no_grad():
            # Force tqdm to work in Docker with explicit file output
            progress_bar = get_progress_bar(
                data_loader, 
                desc="Validation",
                file=sys.stdout,
                ncols=80,
                miniters=1,
                mininterval=0.5
            )
            
            for batch in progress_bar:
                # Move batch to device
                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                justice_input_ids = batch['justice_input_ids'].to(self.device)
                justice_attention_mask = batch['justice_attention_mask'].to(self.device)
                justice_counts = batch['justice_counts']
                targets = batch['targets'].to(self.device)
                
                # Forward pass with autocast for speed
                with autocast(device_type=self.device_string):
                    predictions = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                    loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Force flush every 10 batches for Docker visibility
                if num_batches % 10 == 0:
                    sys.stdout.flush()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.logger.info(f"✅ Validation completed - Average loss: {avg_loss:.4f}")
        
        return avg_loss

    def evaluate_on_holdout_test_set(self, model_path: str = None) -> Dict:
        """Evaluate model on holdout test set."""
        # Load holdout test dataset (same set used by optimization)
        dataset_source_file = self.config.dataset_file
        holdout_manager = HoldoutTestSetManager(dataset_file=dataset_source_file)
        full_dataset = self.load_case_dataset(dataset_source_file)
        test_dataset = holdout_manager.get_holdout_dataset(full_dataset)

        if not test_dataset:
            self.logger.warning("No holdout test cases found")
            return {'error': 'No test cases found'}
        
        # Load model
        if model_path is None:
            model_path = os.path.join(self.config.model_output_dir, 'best_model.pth')
        
        bio_tokenized_file, description_tokenized_file = self.get_tokenized_file_paths()
        model = SCOTUSVotingModel.load_model(
            model_path,
            bio_model_name=self.config.bio_model_name,
            description_model_name=self.config.description_model_name,
            device=self.device
        )
        
        # Load tokenized data and process test cases
        bio_data, description_data = self.load_tokenized_data(bio_tokenized_file, description_tokenized_file)
        test_processed = self.prepare_processed_data(test_dataset, bio_data, description_data, verbose=False)
        
        # Create test dataset and loader
        test_dataset = SCOTUSDataset(test_processed)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
        
        # Evaluate: compute loss and F1 Macro
        criterion = create_scotus_loss_function(self.config)
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_trues = []
        
        model.eval()
        with torch.no_grad():
            progress_bar = get_progress_bar(test_loader, desc="Computing F1 Score")
            for batch in progress_bar:
                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                justice_input_ids = batch['justice_input_ids'].to(self.device)
                justice_attention_mask = batch['justice_attention_mask'].to(self.device)
                justice_counts = batch['justice_counts']
                targets = batch['targets'].to(self.device)
                
                with autocast(device_type=self.device_string):
                    logits = model(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
                
                preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                trues = torch.argmax(targets, dim=-1).detach().cpu().tolist()
                all_preds.extend(preds)
                all_trues.extend(trues)
        
        f1_macro = float(calculate_f1_macro(all_preds, all_trues, num_classes=3)) if all_preds else 0.0
        
        results = {
            'test_loss': test_loss,
            'f1_macro': f1_macro,
            'num_test_cases': len(test_processed),
            'model_path': model_path
        }
        
        self.logger.info(f"🧪 Holdout test evaluation results:")
        self.logger.info(f"   Test loss: {test_loss:.4f}")
        self.logger.info(f"   F1-Score Macro: {f1_macro:.4f}")
        self.logger.info(f"   Test cases: {len(test_processed)}")
        
        return results