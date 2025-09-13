import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from typing import List, Optional, Dict, Any, Tuple
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
import sys
MAX_JUSTICES=11

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    from models.justice_cross_attention import JusticeCrossAttention
except ImportError:
    from .justice_cross_attention import JusticeCrossAttention


# Fallbacks for common Docker mounts
def first_existing_candidate(original: Path) -> Path:
    if original.exists():
        return original
    parts = list(original.parts)
    tail = None
    if 'models_output' in parts:
        idx = parts.index('models_output')
        tail = Path(*parts[idx+1:]) if idx + 1 < len(parts) else Path()
    elif 'models' in parts:
        idx = parts.index('models')
        tail = Path(*parts[idx+1:]) if idx + 1 < len(parts) else Path()
    else:
        # Default to last 3 components as a heuristic
        tail = Path(*parts[-3:]) if len(parts) >= 3 else Path(*parts)
    candidates = [
        Path('/app/models') / tail,
        Path('/app/models_output') / tail,
        Path('/app') / tail,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # Try glob search to handle accidental whitespace or minor name differences
    try:
        if len(tail.parts) >= 1:
            base_name = tail.parts[0].strip()
            for base_dir in [Path('/app/models'), Path('/app/models_output')]:
                for match in base_dir.glob(f"{base_name}*/best_model"):
                    if match.exists():
                        return match
    except Exception:
        pass
    return original


class SCOTUSVotingModel(nn.Module):
    """
    Simplified SCOTUS Voting Prediction Model using dual sentence transformers.
    
    Architecture:
    1. Legal sentence transformer for case descriptions
    2. General sentence transformer for justice biographies
    3. Concatenate embeddings and pass through FC layer
    4. Output 3 neurons with softmax for voting percentages
    
    Note: This simplified version only handles batch processing.
    Single samples should be passed as batches of size 1.
    """
    
    def __init__(
        self,
        bio_model_name: str,  # Required: Model name for biographies
        description_model_name: str,  # Required: Model name for case descriptions
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        max_justices: int = 15,  # Maximum number of justices that can be on the court
        num_attention_heads: int = 4,  # Number of attention heads for justice cross-attention
        use_justice_attention: bool = True,  # Whether to use attention or simple concatenation
        use_noise_reg: bool = True,
        noise_reg_alpha: float = 5.0,
        pretrained_bio_model: str = "",
        device: str = 'cuda'  # Device to place models and data on
    ):
        super(SCOTUSVotingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_justices = max_justices
        self.num_attention_heads = num_attention_heads
        self.use_justice_attention = use_justice_attention
        self.use_noise_reg = use_noise_reg
        self.noise_reg_alpha = noise_reg_alpha
        self.device = device
        self.frozen_bio_model = True
        self.frozen_description_model = True
        
        # Initialize sentence transformer models
        print(f"📥 Loading sentence transformer models...")
        device_arg = device if isinstance(device, str) else str(device)
        self.bio_model = SentenceTransformer(bio_model_name, device=device_arg)
        print(f"PRETRAINED BIO MODEL: {pretrained_bio_model}")
        if pretrained_bio_model:
            # Resolve path: expand ~ and make relative paths absolute to repo root
            raw_path = str(pretrained_bio_model)
            expanded = os.path.expandvars(os.path.expanduser(raw_path))
            path_obj = Path(expanded)
            if not path_obj.is_absolute():
                # Resolve relative to container workdir
                path_obj = (Path('/app') / path_obj).resolve()

            resolved_path = first_existing_candidate(path_obj)
            if not resolved_path.exists():
                # Simple remaps for common Docker host->container path diffs
                remapped = str(resolved_path)
                remapped = remapped.replace('/app/scotus_ai/', '/app/')
                remapped = remapped.replace('/models_output/', '/models/')
                alt_path = Path(remapped)
                if alt_path.exists():
                    resolved_path = alt_path
            print(f"🔗 Loading pretrained bio model from {resolved_path}")
            self.bio_model = SentenceTransformer(str(resolved_path), device=device_arg)
        self.description_model = SentenceTransformer(description_model_name, device=device_arg)
        
        # Register lightweight NEFTune-style noise at embedding layer (train-only)
        self._register_embedding_noise(self.bio_model)
        self._register_embedding_noise(self.description_model)

        # Initially freeze sentence transformers (will be unfrozen during training if configured)
        self.freeze_sentence_transformers()

        # Gradient checkpointing for reducing memory usage
        try:
            t = self.bio_model._first_module()
            if hasattr(t, "auto_model"):
                t.auto_model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for bio model")
                if hasattr(t.auto_model, "config"):
                    t.auto_model.config.use_cache = False
                    print("Disabled use cache for bio model")
            else:
                print("No auto model found for bio model")
            t = self.description_model._first_module()
            if hasattr(t, "auto_model"):
                t.auto_model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for description model")
                if hasattr(t.auto_model, "config"):
                    t.auto_model.config.use_cache = False
                    print("Disabled use cache for description model")
            else:
                print("No auto model found for description model")
        except Exception:
            print("Failed to enable gradient checkpointing for sentence transformers")
            pass

        # Verify embedding dimensions match and create projection layers if needed
        bio_actual_dim = self.bio_model.get_sentence_embedding_dimension()
        desc_actual_dim = self.description_model.get_sentence_embedding_dimension()
        
        # Initialize projection layers
        self.bio_projection_layer = None
        self.description_projection_layer = None
        
        if bio_actual_dim != embedding_dim:
            print(f"Biography model embedding dimension {bio_actual_dim} doesn't match expected {embedding_dim}")
            self.bio_projection_layer = nn.Linear(bio_actual_dim, embedding_dim)
        else:
            print(f"Biography model embedding dimension {bio_actual_dim} matches expected {embedding_dim}")
            
        if desc_actual_dim != embedding_dim:
            print(f"Description model embedding dimension {desc_actual_dim} doesn't match expected {embedding_dim}")
            self.description_projection_layer = nn.Linear(desc_actual_dim, embedding_dim)
        else:
            print(f"Description model embedding dimension {desc_actual_dim} matches expected {embedding_dim}")
        
        # Justice attention mechanism or concatenation
        self.justice_dropout = nn.Dropout(dropout_rate)
        if use_justice_attention:
            self.justice_attention = JusticeCrossAttention(
                embedding_dim=embedding_dim,
                num_heads=num_attention_heads,
                dropout_rate=dropout_rate
            )
            fc_input_dim = embedding_dim * 2  # case_emb + court_emb
        else:
            # Original concatenation approach
            self.justice_attention = None
            fc_input_dim = embedding_dim * (1 + max_justices)  # case + all justices
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)  # 3 output neurons for voting percentages
        )
    
    def _register_embedding_noise(self, sbert_model: SentenceTransformer) -> None:
        """Attach a forward hook on the underlying HF embedding layer to add noise during training.
        Mirrors the approach used in scripts/pretraining/constrastive_justice.py.
        """
        try:
            transformer_module = sbert_model._first_module()
            hf_model = getattr(transformer_module, 'auto_model', None)
            if hf_model is None:
                return
            embeddings_layer = getattr(hf_model, 'embeddings', None)
            if embeddings_layer is None:
                return

            def _noise_hook(module, inputs, output):
                if self.training and self.use_noise_reg:
                    d = output.size(-1)
                    if d == 0:
                        return output
                    scale = self.noise_reg_alpha / (d ** 0.5)
                    noise = torch.empty_like(output).uniform_(-1, 1) * scale
                    output = output + noise
                return output

            embeddings_layer.register_forward_hook(_noise_hook)
        except Exception:
            # Fail silently if internals differ
            pass
    
    def forward(self, 
                case_input_ids: torch.Tensor, 
                case_attention_mask: torch.Tensor,
                justice_input_ids: torch.Tensor, 
                justice_attention_mask: torch.Tensor,
                justice_counts: List[int]) -> torch.Tensor:
        """
        Forward pass of the model using tokenized inputs.
        
        Args:
            case_input_ids: Tensor of shape (batch_size, case_seq_len)
            case_attention_mask: Tensor of shape (batch_size, case_seq_len)
            justice_input_ids: Tensor of shape (batch_size, max_justices, justice_seq_len)
            justice_attention_mask: Tensor of shape (batch_size, max_justices, justice_seq_len)
            justice_counts: List of integers indicating number of justices per case
            
        Returns:
            Tensor of shape (batch_size, 3) with voting percentage predictions
        """
        batch_size = case_input_ids.size(0)
        
        # Case embeddings via direct SentenceTransformer call
        case_features = {
            'input_ids': case_input_ids,
            'attention_mask': case_attention_mask
        }
        with torch.no_grad() if not self.training or self.frozen_description_model else torch.enable_grad():
            case_embeddings = self.description_model(case_features)['sentence_embedding']
            if self.description_projection_layer:
                case_embeddings = self.description_projection_layer(case_embeddings)

        # Justice embeddings: select only valid justices, encode, then scatter back
        max_justices = justice_input_ids.size(1)
        valid_mask = torch.zeros(batch_size, max_justices, dtype=torch.bool, device=justice_input_ids.device)
        for i, count in enumerate(justice_counts):
            valid_mask[i, :count] = True
        valid_indices = valid_mask.nonzero(as_tuple=False)
        if len(valid_indices) == 0:
            justice_embeddings = torch.zeros(batch_size, max_justices, self.embedding_dim, device=justice_input_ids.device)
        else:
            valid_input_ids = justice_input_ids[valid_indices[:, 0], valid_indices[:, 1]]
            valid_attention_mask = justice_attention_mask[valid_indices[:, 0], valid_indices[:, 1]]
            bio_features = {
                'input_ids': valid_input_ids,
                'attention_mask': valid_attention_mask
            }
            with torch.no_grad() if not self.training or self.frozen_bio_model else torch.enable_grad():
                valid_embeddings = self.bio_model(bio_features)['sentence_embedding']
                if self.bio_projection_layer:
                    valid_embeddings = self.bio_projection_layer(valid_embeddings)
            justice_embeddings = torch.zeros(batch_size, max_justices, self.embedding_dim, 
                                             device=justice_input_ids.device, dtype=valid_embeddings.dtype)
            justice_embeddings[valid_indices[:, 0], valid_indices[:, 1]] = valid_embeddings

        if self.use_justice_attention:
            # Create batch masks for real vs padded justices
            justice_masks = torch.zeros(batch_size, MAX_JUSTICES, dtype=torch.bool, device=case_embeddings.device)
            for i, count in enumerate(justice_counts):
                justice_masks[i, :count] = True
            
            # Apply dropout to justice embeddings
            justice_embeddings = self.justice_dropout(justice_embeddings)
            
            # BATCHED PROCESSING - modify JusticeCrossAttention to accept batched inputs
            court_embeddings = self.justice_attention(
                case_emb=case_embeddings,  # (batch_size, embedding_dim)
                justice_embs=justice_embeddings,  # (batch_size, max_justices, embedding_dim)
                justice_mask=justice_masks  # (batch_size, max_justices)
            )
            
            # Combine case and court embeddings in batch
            combined_embeddings = torch.cat([case_embeddings, court_embeddings], dim=1)  # (batch_size, embedding_dim * 2)
            
            # Pass through FC layers in parallel
            return self.fc_layers(combined_embeddings)  # (batch_size, 3)
        else:
            # Original concatenation approach - process in parallel
            # Apply dropout
            justice_embeddings = self.justice_dropout(justice_embeddings)
            
            # Flatten justice embeddings: (batch_size, max_justices * embedding_dim)
            justice_embeddings_flat = justice_embeddings.view(batch_size, -1)
            
            # Concatenate case and justice embeddings: (batch_size, embedding_dim + max_justices * embedding_dim)
            combined_embeddings = torch.cat([case_embeddings, justice_embeddings_flat], dim=1)
            
            # Pass through fully connected layers in parallel
            return self.fc_layers(combined_embeddings)  # (batch_size, 3)
    
    def predict_probabilities(self, 
                            case_input_ids: torch.Tensor, 
                            case_attention_mask: torch.Tensor,
                            justice_input_ids: torch.Tensor, 
                            justice_attention_mask: torch.Tensor,
                            justice_counts: List[int]) -> torch.Tensor:
        """
        Make predictions returning probabilities.
        
        Args:
            case_input_ids: Tensor of shape (batch_size, case_seq_len)
            case_attention_mask: Tensor of shape (batch_size, case_seq_len)
            justice_input_ids: Tensor of shape (batch_size, max_justices, justice_seq_len)
            justice_attention_mask: Tensor of shape (batch_size, max_justices, justice_seq_len)
            justice_counts: List of integers indicating number of justices per case
            
        Returns:
            Tensor of shape (batch_size, 3) with voting percentage predictions (probabilities)
        """
        with torch.no_grad():
            logits = self.forward(case_input_ids, case_attention_mask, justice_input_ids, justice_attention_mask, justice_counts)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def save_model(self, filepath: str):
        """Save the model state dict."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_justices': self.max_justices,
            'num_attention_heads': self.num_attention_heads,
            'use_justice_attention': self.use_justice_attention,
            'device': self.device
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, bio_model_name: str, description_model_name: str, device: str = None):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
        
        # Use device from checkpoint if not specified
        if device is None:
            device = checkpoint.get('device', 'cpu')
        
        model = cls(
            bio_model_name=bio_model_name,
            description_model_name=description_model_name,
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            max_justices=checkpoint['max_justices'],
            num_attention_heads=checkpoint.get('num_attention_heads', 4),  # Default for backward compatibility
            use_justice_attention=checkpoint.get('use_justice_attention', True),
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    # The explicit noise_reg method is no longer used since noise is injected via embedding hooks

    def freeze_sentence_transformers(self):
        """Freeze all sentence transformers."""
        self.frozen_bio_model = True
        self.frozen_description_model = True
        for param in self.bio_model.parameters():
            param.requires_grad = False
        for param in self.description_model.parameters():
            param.requires_grad = False

    def unfreeze_sentence_transformers(self):
        """Unfreeze all sentence transformers."""
        self.frozen_bio_model = False
        self.frozen_description_model = False
        for param in self.bio_model.parameters():
            param.requires_grad = True
        for param in self.description_model.parameters():
            param.requires_grad = True

    def freeze_bio_model(self):
        """Freeze only the biography sentence transformer."""
        self.frozen_bio_model = True
        for param in self.bio_model.parameters():
            param.requires_grad = False

    def unfreeze_bio_model(self):
        """Unfreeze only the biography sentence transformer."""
        self.frozen_bio_model = False
        for param in self.bio_model.parameters():
            param.requires_grad = True

    def freeze_description_model(self):
        """Freeze only the description sentence transformer."""
        self.frozen_description_model = True
        for param in self.description_model.parameters():
            param.requires_grad = False

    def unfreeze_description_model(self):
        """Unfreeze only the description sentence transformer."""
        self.frozen_description_model = False
        for param in self.description_model.parameters():
            param.requires_grad = True

    def get_sentence_transformer_status(self) -> Dict[str, bool]:
        """
        Get the frozen/unfrozen status of sentence transformers.
        
        Returns:
            Dictionary with model names and their trainable status
        """
        bio_trainable = any(param.requires_grad for param in self.bio_model.parameters())
        desc_trainable = any(param.requires_grad for param in self.description_model.parameters())
        
        return {
            'bio_model_trainable': bio_trainable,
            'description_model_trainable': desc_trainable,
            'any_trainable': bio_trainable or desc_trainable
        }


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    """
    batch_size = len(batch)
    
    # Extract all data
    case_ids = [item['case_id'] for item in batch]
    justice_counts = [item['justice_count'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # Pad case inputs efficiently
    case_input_ids = pad_sequence([item['case_input_ids'] for item in batch], batch_first=True)
    case_attention_mask = pad_sequence([item['case_attention_mask'] for item in batch], batch_first=True)
    
    # Pre-allocate justice tensors for efficiency
    max_seq_len = max(item['justice_input_ids'].size(-1) for item in batch)
    justice_input_ids = torch.zeros(batch_size, MAX_JUSTICES, max_seq_len, dtype=torch.long)
    justice_attention_mask = torch.zeros(batch_size, MAX_JUSTICES, max_seq_len, dtype=torch.long)
    
    # Fill justice tensors efficiently
    for i, item in enumerate(batch):
        item_justices = item['justice_input_ids'].size(0)
        actual_seq_len = item['justice_input_ids'].size(1)
        
        # Copy actual justice data
        justice_input_ids[i, :item_justices, :actual_seq_len] = item['justice_input_ids']
        justice_attention_mask[i, :item_justices, :actual_seq_len] = item['justice_attention_mask']
    
    return {
        'case_ids': case_ids,
        'case_input_ids': case_input_ids,
        'case_attention_mask': case_attention_mask,
        'justice_input_ids': justice_input_ids,
        'justice_attention_mask': justice_attention_mask,
        'justice_counts': justice_counts,
        'targets': targets
    }


class SCOTUSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for SCOTUS voting data.
    """
    
    def __init__(self, processed_data: List[Dict], transform=None):
        """
        Initialize dataset.
        
        Args:
            processed_data: List of processed data dictionaries with tokenized inputs
        """
        self.data = processed_data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample