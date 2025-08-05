import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
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
        self.bio_model = SentenceTransformer(bio_model_name, device=device)
        self.description_model = SentenceTransformer(description_model_name, device=device)
        
        # Initially freeze sentence transformers (will be unfrozen during training if configured)
        self.freeze_sentence_transformers()
        
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
    
    def encode_case_descriptions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode case descriptions using input_ids and attention_mask.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) representing the cases
        """
        # Prepare features for sentence transformer
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Use the sentence transformer's encoding
        with torch.no_grad() if not self.training or self.frozen_description_model else torch.enable_grad():
            embeddings = self.description_model(features)['sentence_embedding']
            
            if self.use_noise_reg and self.training:
                embeddings = self.noise_reg(embeddings)
                
            if self.description_projection_layer:
                embeddings = self.description_projection_layer(embeddings)
        
        return embeddings
    
    def encode_justice_bios(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode justice biographies using input_ids and attention_mask.
        
        Args:
            input_ids: Tensor of shape (total_justices, seq_len)
            attention_mask: Tensor of shape (total_justices, seq_len)
            
        Returns:
            Tensor of shape (total_justices, embedding_dim) representing the justices
        """
        # Prepare features for sentence transformer
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Use the sentence transformer's encoding
        with torch.no_grad() if not self.training or self.frozen_bio_model else torch.enable_grad():
            embeddings = self.bio_model(features)['sentence_embedding']
            
            if self.use_noise_reg and self.training:
                embeddings = self.noise_reg(embeddings)
                
            if self.bio_projection_layer:
                embeddings = self.bio_projection_layer(embeddings)
        
        return embeddings
    
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
        
        # Encode case descriptions
        case_embeddings = self.encode_case_descriptions(case_input_ids, case_attention_mask)  # (batch_size, embedding_dim)
        
        # Encode all justice biographies in parallel
        # Reshape to process all justices at once: (batch_size * max_justices, justice_seq_len)
        justice_input_ids_flat = justice_input_ids.view(-1, justice_input_ids.size(-1))
        justice_attention_mask_flat = justice_attention_mask.view(-1, justice_attention_mask.size(-1))
        
        # Get embeddings for all justices: (batch_size * max_justices, embedding_dim)
        all_justice_embeddings = self.encode_justice_bios(justice_input_ids_flat, justice_attention_mask_flat)
        
        # Reshape back to batch format: (batch_size, max_justices, embedding_dim)
        justice_embeddings = all_justice_embeddings.view(batch_size, MAX_JUSTICES, self.embedding_dim)
        
        if self.use_justice_attention:
            # Create batch masks for real vs padded justices
            justice_masks = torch.zeros(batch_size, MAX_JUSTICES, dtype=torch.bool, device=case_embeddings.device)
            for i, count in enumerate(justice_counts):
                justice_masks[i, :count] = True  # True for real justices, False for padding
            
            # Apply dropout and noise regularization to justice embeddings
            justice_embeddings = self.justice_dropout(justice_embeddings)
            if self.use_noise_reg and self.training:
                justice_embeddings = self.noise_reg(justice_embeddings)
            
            # Process all cases through attention in parallel
            batch_outputs = []
            for i in range(batch_size):
                court_embedding = self.justice_attention(
                    case_emb=case_embeddings[i],
                    justice_embs=justice_embeddings[i],
                    justice_mask=justice_masks[i]
                )
                # Combine case and court embeddings
                combined_embedding = torch.cat([case_embeddings[i], court_embedding], dim=0)
                output = self.fc_layers(combined_embedding)
                batch_outputs.append(output)
            
            return torch.stack(batch_outputs)  # (batch_size, 3)
        else:
            # Original concatenation approach - process in parallel
            # Apply dropout and noise regularization
            justice_embeddings = self.justice_dropout(justice_embeddings)
            if self.use_noise_reg and self.training:
                justice_embeddings = self.noise_reg(justice_embeddings)
            
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
    
    def noise_reg(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply noise regularization to an embedding following NEFTune (arXiv:2310.05914)."""
        alpha = self.noise_reg_alpha
        d = embedding.size(-1)
        U = torch.empty_like(embedding).uniform_(-1, 1)
        scale = alpha / d**0.5
        noise = U * scale
        return embedding + noise

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
    case_ids = [item['case_id'] for item in batch]
    case_input_ids = pad_sequence([item['case_input_ids'] for item in batch], batch_first=True)
    case_attention_mask = pad_sequence([item['case_attention_mask'] for item in batch], batch_first=True)
    justice_input_ids = [item['justice_input_ids'] for item in batch]
    justice_attention_mask = [item['justice_attention_mask'] for item in batch]
    
    # Pad case inputs to max_justices
    #print(f"🔗 Justice input ids pre: {justice_input_ids[0].shape}")
    pad = justice_input_ids[0].new_zeros((MAX_JUSTICES - justice_input_ids[0].shape[0],justice_input_ids[0].shape[1]))
    #print(f"pad: {pad.shape}")
    justice_input_ids_padded = torch.cat([justice_input_ids[0], pad], dim=0)
    #print(f"justice_input_ids_padded: {justice_input_ids_padded.shape}")
    justice_attention_mask_padded = torch.cat([justice_attention_mask[0], pad], dim=0)
    #print(f"justice_attention_mask_padded: {justice_attention_mask_padded.shape}")
    justice_input_ids[0] = justice_input_ids_padded
    justice_attention_mask[0] = justice_attention_mask_padded
    justice_input_ids = pad_sequence(justice_input_ids, batch_first=True)
    #print(f"justice_input_ids: {justice_input_ids.shape}")
    justice_attention_mask = pad_sequence(justice_attention_mask, batch_first=True)
    #print(f"justice_attention_mask: {justice_attention_mask.shape}")
    #print("justice_input_ids", justice_input_ids.__class__.__name__)

    justice_counts = [item['justice_count'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
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