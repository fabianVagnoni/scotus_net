import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Dict, Any, Tuple
import torch.nn.functional as F
import os
import pickle
from src.models.justice_cross_attention import JusticeCrossAttention

class SCOTUSVotingModel(nn.Module):
    """
    SCOTUS Voting Prediction Model using dual sentence transformers.
    
    Architecture:
    1. Legal sentence transformer for case descriptions
    2. General sentence transformer for justice biographies
    3. Concatenate embeddings and pass through FC layer
    4. Output 4 neurons with softmax for voting percentages
    """
    
    def __init__(
        self,
        bio_embeddings_file: str,  # Required: Path to pre-encoded biography embeddings
        description_embeddings_file: str,  # Required: Path to pre-encoded case description embeddings
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        max_justices: int = 15,  # Maximum number of justices that can be on the court
        num_attention_heads: int = 4,  # Number of attention heads for justice cross-attention
        use_justice_attention: bool = True,  # Whether to use attention or simple concatenation
        device: str = 'cpu'  # Device to place embeddings on
    ):
        super(SCOTUSVotingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_justices = max_justices
        self.num_attention_heads = num_attention_heads
        self.use_justice_attention = use_justice_attention
        self.device = device
        
        # Pre-encoded embeddings storage
        self.bio_embeddings = {}
        self.description_embeddings = {}
        self.bio_metadata = {}
        self.description_metadata = {}
        
        # Load required pre-encoded embeddings
        if not os.path.exists(bio_embeddings_file):
            raise FileNotFoundError(f"Biography embeddings file not found: {bio_embeddings_file}")
        if not os.path.exists(description_embeddings_file):
            raise FileNotFoundError(f"Description embeddings file not found: {description_embeddings_file}")
            
        self._load_bio_embeddings(bio_embeddings_file)
        self._load_description_embeddings(description_embeddings_file)
        
        # Verify embedding dimensions match
        if self.bio_metadata['embedding_dim'] != embedding_dim:
            raise ValueError(f"Biography embedding dimension {self.bio_metadata['embedding_dim']} doesn't match expected {embedding_dim}")
        if self.description_metadata['embedding_dim'] != embedding_dim:
            raise ValueError(f"Description embedding dimension {self.description_metadata['embedding_dim']} doesn't match expected {embedding_dim}")
        
        print(f"✅ Model initialized with {len(self.bio_embeddings)} biography and {len(self.description_embeddings)} case description embeddings")
        
        # Justice attention mechanism or concatenation
        if use_justice_attention:
            self.justice_attention = JusticeCrossAttention(
                embedding_dim=embedding_dim,
                num_heads=num_attention_heads,
                dropout_rate=dropout_rate
            )
            # With attention, we use case + contextualized court embedding
            fc_input_dim = embedding_dim * 2  # case_emb + court_emb
        else:
            # Original concatenation approach
            fc_input_dim = embedding_dim * (1 + max_justices)  # case + all justices
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 4)  # 4 output neurons for voting percentages
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def _load_bio_embeddings(self, embeddings_file: str):
        """Load pre-encoded biography embeddings."""
        print(f"📥 Loading pre-encoded biographies from: {embeddings_file}")
        
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load embeddings and move to specified device
        self.bio_embeddings = {}
        for path, embedding in data['embeddings'].items():
            self.bio_embeddings[path] = embedding.to(self.device)
        
        self.bio_metadata = data['metadata']
        
        print(f"✅ Loaded {self.bio_metadata['num_embeddings']} pre-encoded biographies to {self.device}")
    
    def _load_description_embeddings(self, embeddings_file: str):
        """Load pre-encoded case description embeddings."""
        print(f"📥 Loading pre-encoded descriptions from: {embeddings_file}")
        
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load embeddings and move to specified device
        self.description_embeddings = {}
        for path, embedding in data['embeddings'].items():
            self.description_embeddings[path] = embedding.to(self.device)
        
        self.description_metadata = data['metadata']
        
        print(f"✅ Loaded {self.description_metadata['num_embeddings']} pre-encoded descriptions to {self.device}")
    
    def encode_case_description(self, case_description_path: str) -> torch.Tensor:
        """
        Encode case description using ONLY pre-encoded embeddings from .pkl files.
        
        Args:
            case_description_path: Path to case description file (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (embedding_dim,) representing the case
            
        Raises:
            KeyError: If the case description path is not found in pre-encoded embeddings
        """
        if not case_description_path:
            raise ValueError("case_description_path is required when using pre-encoded embeddings")
        
        # Normalize path for consistent lookup
        normalized_path = case_description_path.replace('\\', '/')
        
        # Try different path variations for robust lookup
        path_variations = [
            normalized_path,
            os.path.abspath(normalized_path).replace('\\', '/'),
            os.path.normpath(normalized_path).replace('\\', '/'),
        ]
        
        for path_variant in path_variations:
            if path_variant in self.description_embeddings:
                return self.description_embeddings[path_variant]
        
        # If not found, raise error with helpful message
        available_paths = list(self.description_embeddings.keys())[:5]  # Show first 5 for debugging
        raise KeyError(
            f"Case description path not found in pre-encoded embeddings: {case_description_path}\n"
            f"Tried variations: {path_variations}\n"
            f"Available paths (first 5): {available_paths}\n"
            f"Total available: {len(self.description_embeddings)}"
        )
    
    def encode_justice_bio(self, justice_bio_path: str) -> torch.Tensor:
        """
        Encode justice biography using ONLY pre-encoded embeddings from .pkl files.
        
        Args:
            justice_bio_path: Path to justice biography file (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (embedding_dim,) representing the justice
            
        Raises:
            KeyError: If the justice biography path is not found in pre-encoded embeddings
        """
        if not justice_bio_path:
            raise ValueError("justice_bio_path is required when using pre-encoded embeddings")
        
        # Normalize path for consistent lookup
        normalized_path = justice_bio_path.replace('\\', '/')
        
        # Try different path variations for robust lookup
        path_variations = [
            normalized_path,
            os.path.abspath(normalized_path).replace('\\', '/'),
            os.path.normpath(normalized_path).replace('\\', '/'),
        ]
        
        for path_variant in path_variations:
            if path_variant in self.bio_embeddings:
                return self.bio_embeddings[path_variant]
        
        # If not found, raise error with helpful message
        available_paths = list(self.bio_embeddings.keys())[:5]  # Show first 5 for debugging
        raise KeyError(
            f"Justice biography path not found in pre-encoded embeddings: {justice_bio_path}\n"
            f"Tried variations: {path_variations}\n"
            f"Available paths (first 5): {available_paths}\n"
            f"Total available: {len(self.bio_embeddings)}"
        )
    
    def forward(self, case_description_path: str, justice_bio_paths: List[str]) -> torch.Tensor:
        """
        Forward pass of the model using ONLY pre-encoded embeddings from .pkl files.
        
        Args:
            case_description_path: Path to case description file (for pre-encoded lookup)
            justice_bio_paths: List of paths to justice biography files (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (4,) with voting percentage predictions
        """
        # Encode case description
        case_embedding = self.encode_case_description(case_description_path)
        
        # Encode justice biographies
        justice_embeddings = []
        num_real_justices = len(justice_bio_paths)
        
        for bio_path in justice_bio_paths:
            if bio_path:  # Skip None/empty paths
                justice_embedding = self.encode_justice_bio(bio_path)
                justice_embeddings.append(justice_embedding)
        
        # Update num_real_justices to reflect actual loaded justices
        num_real_justices = len(justice_embeddings)
        
        # Pad with zeros if fewer justices than max_justices
        while len(justice_embeddings) < self.max_justices:
            zero_embedding = torch.zeros(self.embedding_dim, device=case_embedding.device)
            justice_embeddings.append(zero_embedding)
        
        # Truncate if more justices than max_justices
        justice_embeddings = justice_embeddings[:self.max_justices]
        
        if self.use_justice_attention:
            # Use attention mechanism
            # Stack justice embeddings: (max_justices, embedding_dim)
            justice_embs_tensor = torch.stack(justice_embeddings)
            
            # Create mask for real vs padded justices
            justice_mask = torch.zeros(self.max_justices, dtype=torch.bool, device=case_embedding.device)
            justice_mask[:num_real_justices] = True  # True for real justices, False for padding
            
            # Apply cross-attention
            court_embedding = self.justice_attention(
                case_emb=case_embedding,
                justice_embs=justice_embs_tensor,
                justice_mask=justice_mask
            )
            
            # Combine case and court embeddings
            combined_embedding = torch.cat([case_embedding, court_embedding], dim=0)
        else:
            # Original concatenation approach
            all_embeddings = [case_embedding] + justice_embeddings
            combined_embedding = torch.cat(all_embeddings, dim=0)
        
        # Pass through fully connected layers
        output = self.fc_layers(combined_embedding)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=0)
        
        return probabilities
    
    def predict_from_files(self, case_description_path: str, justice_bio_paths: List[str]) -> torch.Tensor:
        """
        Make prediction from file paths using ONLY pre-encoded embeddings.
        
        Args:
            case_description_path: Path to case description file
            justice_bio_paths: List of paths to justice biography files
            
        Returns:
            Tensor of shape (4,) with voting percentage predictions
            
        Note:
            This method now requires all files to be pre-encoded in the .pkl files.
            No fallback to reading raw text files.
        """
        return self.forward(case_description_path, justice_bio_paths)
    
    def predict_batch_from_dataset(self, dataset_entries: List[Tuple[List[str], str, List[float]]]) -> torch.Tensor:
        """
        Make batch predictions from dataset entries.
        
        Args:
            dataset_entries: List of (justice_bio_paths, case_description_path, voting_percentages)
            
        Returns:
            Tensor of shape (batch_size, 4) with voting percentage predictions
        """
        batch_predictions = []
        
        for justice_bio_paths, case_description_path, _ in dataset_entries:
            prediction = self.predict_from_files(case_description_path, justice_bio_paths)
            batch_predictions.append(prediction)
        
        return torch.stack(batch_predictions)
    
    def save_model(self, filepath: str):
        """Save the model state dict."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_justices': self.max_justices,
            'num_attention_heads': self.num_attention_heads,
            'use_justice_attention': self.use_justice_attention,
            'bio_metadata': self.bio_metadata,
            'description_metadata': self.description_metadata,
            'device': self.device
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, bio_embeddings_file: str, description_embeddings_file: str, 
                   device: str = None):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
        
        # Use device from checkpoint if not specified
        if device is None:
            device = checkpoint.get('device', 'cpu')
        
        model = cls(
            bio_embeddings_file=bio_embeddings_file,
            description_embeddings_file=description_embeddings_file,
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
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded embeddings."""
        stats = {
            'bio_embeddings_loaded': len(self.bio_embeddings),
            'description_embeddings_loaded': len(self.description_embeddings),
            'bio_metadata': self.bio_metadata,
            'description_metadata': self.description_metadata,
            'device': self.device
        }
            
        return stats
    
    def get_available_paths(self) -> Dict[str, List[str]]:
        """Get lists of available file paths in embeddings."""
        return {
            'bio_paths': list(self.bio_embeddings.keys()),
            'description_paths': list(self.description_embeddings.keys())
        }


class SCOTUSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for SCOTUS voting data.
    """
    
    def __init__(self, dataset_dict: Dict[str, List], transform=None):
        """
        Initialize dataset.
        
        Args:
            dataset_dict: Dictionary with case_id as keys and 
                         [justice_bio_paths, case_description_path, voting_percentages] as values
        """
        self.dataset_dict = dataset_dict
        self.case_ids = list(dataset_dict.keys())
        self.transform = transform
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        justice_bio_paths, case_description_path, voting_percentages = self.dataset_dict[case_id]
        
        # Convert voting percentages to tensor
        target = torch.tensor(voting_percentages, dtype=torch.float32)
        
        sample = {
            'case_id': case_id,
            'justice_bio_paths': justice_bio_paths,
            'case_description_path': case_description_path,
            'target': target
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    """
    case_ids = [item['case_id'] for item in batch]
    justice_bio_paths = [item['justice_bio_paths'] for item in batch]
    case_description_paths = [item['case_description_path'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'case_ids': case_ids,
        'justice_bio_paths': justice_bio_paths,
        'case_description_paths': case_description_paths,
        'targets': targets
    } 