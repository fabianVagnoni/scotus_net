"""
Baseline SCOTUS Model Trainer.

This trainer mirrors the simplified structure of scripts/models/model_trainer.py
but adapts it to the single-pipeline baseline that consumes only case descriptions.

Key differences vs the dual-pipeline trainer:
- Uses only the encoded case descriptions (no justice bios)
- Uses the BaselineSCOTUSModel from scripts/baseline/baseline_model.py
- Uses a simple KL-Div loss over 3-class distributions
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# AMP (recommended imports)
from torch.amp import autocast, GradScaler

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from scripts.baseline.baseline_model import BaselineSCOTUSModel
    from scripts.baseline.config import BaselineConfig
    from scripts.utils.logger import get_logger
    from scripts.utils.holdout_test_set import HoldoutTestSetManager
    from scripts.utils.progress import get_progress_bar
    from scripts.utils.metrics import calculate_f1_macro
    # Use the KL loss that expects (batch, 3)
    from scripts.models.losses import create_scotus_loss_function
except ImportError:
    from .baseline_model import BaselineSCOTUSModel
    from .config import BaselineConfig
    from ..utils.logger import get_logger
    from ..utils.holdout_test_set import HoldoutTestSetManager
    from ..utils.progress import get_progress_bar
    from ..utils.metrics import calculate_f1_macro
    from ..models.losses import create_scotus_loss_function


def _range_str(case_ids: List[str], holdout_manager: HoldoutTestSetManager) -> str:
    if not case_ids:
        return "N/A"
    years = [holdout_manager.extract_year_from_case_id(cid) for cid in case_ids]
    years = [y for y in years if y > 0]
    if not years:
        return "N/A"
    return f"{min(years)}-{max(years)}"


class BaselineDataset(torch.utils.data.Dataset):
    """
    Dataset for baseline trainer containing only case description inputs and 3-class targets.
    """

    def __init__(self, processed_data: List[Dict]):
        self.data = processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Case IDs (optional informational)
    case_ids = [item['case_id'] for item in batch]

    # Targets (batch, 3)
    targets = torch.stack([item['target'] for item in batch])

    # Pad case sequences
    case_input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['case_input_ids'] for item in batch], batch_first=True
    )
    case_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['case_attention_mask'] for item in batch], batch_first=True
    )

    return {
        'case_ids': case_ids,
        'case_input_ids': case_input_ids,
        'case_attention_mask': case_attention_mask,
        'targets': targets,
    }


class BaselineTrainer:
    """
    Baseline trainer that follows scripts/models/model_trainer.py structure
    while adapting to the baseline model's single description pipeline.
    """

    def __init__(
        self,
        dataset_file: Optional[str] = None,
        description_tokenized_file: Optional[str] = None,
        description_model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        device: Optional[str] = None,
        train_dataset_json: Optional[Dict[str, Any]] = None,
        val_dataset_json: Optional[Dict[str, Any]] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = BaselineConfig()

        # Allow overrides (used by HPO or programmatic calls)
        self.dataset_file = dataset_file or self.config.dataset_file
        self.description_tokenized_file = description_tokenized_file or self.config.description_tokenized_file
        self.description_model_name = description_model_name or self.config.description_model_name
        self.embedding_dim = embedding_dim or self.config.embedding_dim
        self.hidden_dim = hidden_dim or self.config.hidden_dim
        self.dropout_rate = dropout_rate if dropout_rate is not None else self.config.dropout_rate
        self.batch_size = batch_size or self.config.batch_size
        self.learning_rate = learning_rate or self.config.learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else self.config.weight_decay
        self.num_epochs = num_epochs or self.config.num_epochs
        self.patience = patience or self.config.patience
        self.year_start = year_start or self.config.year_start
        self.year_end = year_end or self.config.year_end

        # Train/val split ratios (kept internal for simplicity)
        self.train_ratio = 0.85
        self.val_ratio = 0.15

        # Device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(self.config.device)
        self.device_string = "cuda" if self.device.type == 'cuda' else "cpu"

        # Optional pre-specified splits (used by HPO)
        self._train_dataset_json = train_dataset_json
        self._val_dataset_json = val_dataset_json

        # AMP scaler
        self.scaler = GradScaler()

    # --------------------- Data Loading & Processing ---------------------
    def load_case_dataset(self, dataset_file: Optional[str] = None) -> Dict[str, Any]:
        dataset_file = dataset_file or self.dataset_file
        self.logger.info(f"📂 Loading case dataset from: {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        self.logger.info(f"✅ Loaded {len(dataset)} cases from dataset")
        return dataset

    def filter_by_year_range(self, dataset: Dict[str, Any], holdout_manager: HoldoutTestSetManager) -> Dict[str, Any]:
        if self.year_start is None and self.year_end is None:
            return dataset
        filtered = {}
        for cid, cdata in dataset.items():
            y = holdout_manager.extract_year_from_case_id(cid)
            if (self.year_start is None or y >= self.year_start) and (self.year_end is None or y <= self.year_end):
                filtered[cid] = cdata
        self.logger.info(f"📆 Year filter [{self.year_start}, {self.year_end}] -> {len(filtered)} cases")
        return filtered

    def get_tokenized_file_path(self) -> str:
        desc_tokenized = os.path.join(self.description_tokenized_file)
        self.logger.info(f"🔗 Description tokenized file: {desc_tokenized}")
        return desc_tokenized

    def load_tokenized_descriptions(self, description_tokenized_file: str) -> dict:
        self.logger.info("📥 Loading pre-tokenized description data...")
        with open(description_tokenized_file, 'rb') as f:
            description_data = pickle.load(f)
        self.logger.info(f"✅ Loaded {len(description_data['tokenized_data'])} description tokenized files")
        return description_data

    def prepare_processed_data(self, dataset: Dict, description_data: dict, verbose: bool = True) -> List[Dict]:
        """Process dataset into tokenized format suitable for the baseline model."""
        processed_data = []
        
        description_tokenized_data = description_data['tokenized_data']
        description_tokenized_data = {k.split('\\')[-1]: v for k, v in description_tokenized_data.items()}
        description_tokenized_data = {k.split('/')[-1]: v for k, v in description_tokenized_data.items()}
        
        if verbose:
            self.logger.info("🔄 Processing dataset entries...")
        
        skipped_cases = []
        
        for case_id, case_data in dataset.items():
            try:
                # Extract data based on the format: [justice_bio_paths, case_description_path, voting_percentages, metadata]
                justice_bio_paths = case_data[0]  # Not used in baseline but maintain structure
                case_description_path = case_data[1].split('\\')[-1]
                voting_percentages = case_data[2]
                case_tokenized = description_tokenized_data[case_description_path]
                
                # Create processed entry (baseline only needs case description)
                processed_entry = {
                    'case_id': case_id,
                    'case_input_ids': case_tokenized['input_ids'],
                    'case_attention_mask': case_tokenized['attention_mask'],
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

    # ---------------------------- Training ----------------------------
    def train(self) -> Dict[str, Any]:
        # Load dataset
        if self._train_dataset_json is not None and self._val_dataset_json is not None:
            # Pre-specified splits (used by HPO)
            holdout_manager = HoldoutTestSetManager(dataset_file=self.dataset_file)
            train_dataset = self._train_dataset_json
            val_dataset = self._val_dataset_json
            train_range = _range_str(list(train_dataset.keys()), holdout_manager)
            val_range = _range_str(list(val_dataset.keys()), holdout_manager)
            test_range = "N/A (HPO mode)"
        else:
            # Create independent baseline splits within year filter range
            dataset = self.load_case_dataset(self.dataset_file)
            holdout_manager = HoldoutTestSetManager(dataset_file=self.dataset_file)
            
            # Apply year filter to get baseline subset (independent of main model holdout)
            baseline_dataset = self.filter_by_year_range(dataset, holdout_manager)
            
            if len(baseline_dataset) == 0:
                raise ValueError(f"No cases found in baseline year range [{self.year_start}, {self.year_end}]")
            
            # Sort baseline cases by year (oldest first for temporal splits)
            sorted_case_ids = sorted(
                baseline_dataset.keys(), key=holdout_manager.extract_year_from_case_id
            )
            
            # Create temporal train/val/test splits within baseline data
            # Use 70% train, 15% val, 15% test (can be configured)
            baseline_train_ratio = 0.70
            baseline_val_ratio = 0.15
            baseline_test_ratio = 0.15
            
            num_total = len(sorted_case_ids)
            num_train = int(num_total * baseline_train_ratio)
            num_val = int(num_total * baseline_val_ratio)
            
            # Temporal splits: oldest for train, middle for val, newest for test
            train_case_ids = sorted_case_ids[:num_train]
            val_case_ids = sorted_case_ids[num_train:num_train + num_val]
            test_case_ids = sorted_case_ids[num_train + num_val:]  # Remaining for test
            
            train_dataset = {cid: baseline_dataset[cid] for cid in train_case_ids}
            val_dataset = {cid: baseline_dataset[cid] for cid in val_case_ids}
            test_dataset = {cid: baseline_dataset[cid] for cid in test_case_ids}
            
            # Store test set for later evaluation
            self._baseline_test_dataset = test_dataset
            
            train_range = _range_str(train_case_ids, holdout_manager)
            val_range = _range_str(val_case_ids, holdout_manager)
            test_range = _range_str(test_case_ids, holdout_manager)

        self.logger.info(f"📊 Baseline temporal split: {len(train_dataset)} train (oldest), {len(val_dataset)} val (newer)")
        self.logger.info(f"🗓️ Train years: {train_range}")
        self.logger.info(f"🗓️ Val years:   {val_range}")
        if self._train_dataset_json is None and 'test_case_ids' in locals():
            self.logger.info(f"🧪 Baseline test years: {test_range} ({len(test_case_ids)} cases)")

        # Tokenized descriptions
        description_tokenized_file = self.get_tokenized_file_path()
        description_data = self.load_tokenized_descriptions(description_tokenized_file)

        # Process datasets
        train_processed = self.prepare_processed_data(train_dataset, description_data)
        val_processed = self.prepare_processed_data(val_dataset, description_data)

        # Datasets / Loaders
        train_ds = BaselineDataset(train_processed)
        val_ds = BaselineDataset(val_processed)

        num_workers = self.config.num_workers
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )

        # Model
        model = BaselineSCOTUSModel(
            description_model_name=self.description_model_name,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            device=self.device_string,
        )
        model.to(self.device)

        # Loss and optimizer
        criterion = create_scotus_loss_function()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training loop
        self.logger.info("🚀 Starting baseline training...")
        self.logger.info(f"   Training samples: {len(train_processed)}")
        self.logger.info(f"   Validation samples: {len(val_processed)}")
        self.logger.info(f"   Batch size: {self.batch_size}")
        self.logger.info(f"   Learning rate: {self.learning_rate}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0

            progress_bar = get_progress_bar(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for batch in progress_bar:
                optimizer.zero_grad()

                case_input_ids = batch['case_input_ids'].to(self.device)
                case_attention_mask = batch['case_attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)

                with autocast(device_type=self.device_string):
                    logits = model(case_input_ids, case_attention_mask)
                    loss = criterion(logits, targets)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                total_train_loss += loss.item()
                num_train_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            val_loss = self.evaluate(model, val_loader, criterion)
            avg_train_loss = total_train_loss / max(1, num_train_batches)
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model in a consistent location under models_output/baseline
                out_dir = Path('models_output') / 'baseline'
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'config': {
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'dropout_rate': self.dropout_rate,
                }}, str(out_dir / 'best_model.pth'))
                self.logger.info(f"💾 New best baseline model saved with val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                    break

        self.logger.info("✅ Baseline training completed")
        self.logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")

        return {
            'best_val_loss': float(best_val_loss),
        }

    @torch.no_grad()
    def evaluate(self, model: BaselineSCOTUSModel, data_loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in data_loader:
            case_input_ids = batch['case_input_ids'].to(self.device)
            case_attention_mask = batch['case_attention_mask'].to(self.device)
            targets = batch['targets'].to(self.device)

            logits = model(case_input_ids, case_attention_mask)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate_on_baseline_test_set(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        # Use baseline's own test set (created during training)
        if not hasattr(self, '_baseline_test_dataset') or not self._baseline_test_dataset:
            self.logger.warning("No baseline test set found. Run training first to create test set.")
            return {'error': 'No baseline test cases found'}
        
        test_dataset = self._baseline_test_dataset

        # Load model
        model = BaselineSCOTUSModel(
            description_model_name=self.description_model_name,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            device=self.device_string,
        )
        if model_path is None:
            model_path = str(Path('models_output') / 'baseline' / 'best_model.pth')
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        model.to(self.device)
        model.eval()

        # Tokenized descriptions
        description_tokenized_file = self.get_tokenized_file_path()
        description_data = self.load_tokenized_descriptions(description_tokenized_file)
        test_processed = self.prepare_processed_data(test_dataset, description_data, verbose=False)

        test_ds = BaselineDataset(test_processed)
        num_workers = self.config.num_workers
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )

        criterion = create_scotus_loss_function()
        total_loss = 0.0
        num_batches = 0
        all_preds: List[int] = []
        all_trues: List[int] = []

        for batch in test_loader:
            case_input_ids = batch['case_input_ids'].to(self.device)
            case_attention_mask = batch['case_attention_mask'].to(self.device)
            targets = batch['targets'].to(self.device)

            logits = model(case_input_ids, case_attention_mask)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            trues = torch.argmax(targets, dim=-1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_trues.extend(trues)

        test_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        f1_macro = float(calculate_f1_macro(all_preds, all_trues, num_classes=3)) if all_preds else 0.0

        results = {
            'test_loss': test_loss,
            'f1_macro': f1_macro,
            'num_test_cases': len(test_processed),
            'model_path': model_path,
        }

        self.logger.info(f"🧪 Baseline test set evaluation:")
        self.logger.info(f"   Test loss: {test_loss:.4f}")
        self.logger.info(f"   F1-Score Macro: {f1_macro:.4f}")
        self.logger.info(f"   Test cases: {len(test_processed)}")

        return results

    # Alias for backward compatibility
    def evaluate_on_holdout_test_set(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Alias for evaluate_on_baseline_test_set for backward compatibility."""
        return self.evaluate_on_baseline_test_set(model_path)

