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
        legal_model_name: str = "nlpaueb/legal-bert-base-uncased",
        bio_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        max_justices: int = 15,  # Maximum number of justices that can be on the court
        max_sequence_length: int = 512,
        num_attention_heads: int = 4,  # Number of attention heads for justice cross-attention
        use_justice_attention: bool = True,  # Whether to use attention or simple concatenation
        bio_embeddings_file: Optional[str] = None,  # Path to pre-encoded biography embeddings
        description_embeddings_file: Optional[str] = None,  # Path to pre-encoded case description embeddings
        use_preencoded_only: bool = False  # If True, don't load transformer models (faster initialization)
    ):
        super(SCOTUSVotingModel, self).__init__()
        
        self.legal_model_name = legal_model_name
        self.bio_model_name = bio_model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_justices = max_justices
        self.max_sequence_length = max_sequence_length
        self.num_attention_heads = num_attention_heads
        self.use_justice_attention = use_justice_attention
        self.use_preencoded_only = use_preencoded_only
        
        # Pre-encoded embeddings storage
        self.bio_embeddings = {}
        self.description_embeddings = {}
        self.bio_metadata = {}
        self.description_metadata = {}
        
        # Load pre-encoded embeddings if provided
        if bio_embeddings_file and os.path.exists(bio_embeddings_file):
            self._load_bio_embeddings(bio_embeddings_file)
            
        if description_embeddings_file and os.path.exists(description_embeddings_file):
            self._load_description_embeddings(description_embeddings_file)
        
        # Initialize tokenizers and models only if needed
        self.legal_tokenizer = None
        self.legal_model = None
        self.bio_tokenizer = None
        self.bio_model = None
        self.legal_projection = None
        self.bio_projection = None
        
        if not use_preencoded_only:
            self._initialize_transformer_models()
        elif not (self.bio_embeddings and self.description_embeddings):
            print("⚠️  Warning: use_preencoded_only=True but no pre-encoded embeddings loaded.")
            print("   Loading transformer models as fallback...")
            self._initialize_transformer_models()
        
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
        
        self.bio_embeddings = data['embeddings']
        self.bio_metadata = data['metadata']
        
        print(f"✅ Loaded {self.bio_metadata['num_embeddings']} pre-encoded biographies")
    
    def _load_description_embeddings(self, embeddings_file: str):
        """Load pre-encoded case description embeddings."""
        print(f"📥 Loading pre-encoded descriptions from: {embeddings_file}")
        
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.description_embeddings = data['embeddings']
        self.description_metadata = data['metadata']
        
        print(f"✅ Loaded {self.description_metadata['num_embeddings']} pre-encoded descriptions")
    
    def _initialize_transformer_models(self):
        """Initialize the transformer models and tokenizers."""
        print("📥 Loading transformer models...")
        
        # Initialize tokenizers and models
        self.legal_tokenizer = AutoTokenizer.from_pretrained(self.legal_model_name)
        self.legal_model = AutoModel.from_pretrained(self.legal_model_name)
        
        self.bio_tokenizer = AutoTokenizer.from_pretrained(self.bio_model_name)
        self.bio_model = AutoModel.from_pretrained(self.bio_model_name)
        
        # Freeze the base models initially (can be unfrozen for fine-tuning)
        self._freeze_base_models()
        
        # Get actual embedding dimensions from the models
        self.legal_embedding_dim = self.legal_model.config.hidden_size
        self.bio_embedding_dim = self.bio_model.config.hidden_size
        
        # Projection layers to ensure consistent dimensions
        self.legal_projection = nn.Linear(self.legal_embedding_dim, self.embedding_dim)
        self.bio_projection = nn.Linear(self.bio_embedding_dim, self.embedding_dim)
        
        print("✅ Transformer models loaded successfully")
        
    def _freeze_base_models(self):
        """Freeze the base transformer models."""
        for param in self.legal_model.parameters():
            param.requires_grad = False
        for param in self.bio_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze the base transformer models for fine-tuning."""
        for param in self.legal_model.parameters():
            param.requires_grad = True
        for param in self.bio_model.parameters():
            param.requires_grad = True
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings.
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_case_description(self, case_text: str, case_description_path: str = None) -> torch.Tensor:
        """
        Encode case description using pre-encoded embeddings or legal sentence transformer.
        
        Args:
            case_text: Text of the case description (used if no pre-encoded embedding found)
            case_description_path: Path to case description file (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (embedding_dim,) representing the case
        """
        # First, try to get pre-encoded embedding
        if case_description_path:
            normalized_path = os.path.abspath(case_description_path)
            if normalized_path in self.description_embeddings:
                return self.description_embeddings[normalized_path]
        
        # Fallback to on-the-fly encoding
        if self.legal_model is None:
            print("⚠️  No pre-encoded embedding found and no transformer model loaded!")
            print(f"   Path: {case_description_path}")
            return torch.zeros(self.embedding_dim)
        
        # Tokenize
        encoded = self.legal_tokenizer(
            case_text,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.legal_model.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.legal_model(**encoded)
        
        # Mean pooling
        sentence_embedding = self._mean_pooling(model_output, encoded['attention_mask'])
        
        # Project to consistent dimension
        case_embedding = self.legal_projection(sentence_embedding.squeeze(0))
        case_embedding = self.layer_norm(case_embedding)
        
        return case_embedding
    
    def encode_justice_bio(self, bio_text: str, justice_bio_path: str = None) -> torch.Tensor:
        """
        Encode justice biography using pre-encoded embeddings or general sentence transformer.
        
        Args:
            bio_text: Text of the justice biography (used if no pre-encoded embedding found)
            justice_bio_path: Path to justice biography file (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (embedding_dim,) representing the justice
        """
        # First, try to get pre-encoded embedding
        if justice_bio_path:
            normalized_path = os.path.abspath(justice_bio_path)
            if normalized_path in self.bio_embeddings:
                return self.bio_embeddings[normalized_path]
        
        # Fallback to on-the-fly encoding
        if self.bio_model is None:
            print("⚠️  No pre-encoded embedding found and no transformer model loaded!")
            print(f"   Path: {justice_bio_path}")
            return torch.zeros(self.embedding_dim)
        
        # Tokenize
        encoded = self.bio_tokenizer(
            bio_text,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.bio_model.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.bio_model(**encoded)
        
        # Mean pooling
        sentence_embedding = self._mean_pooling(model_output, encoded['attention_mask'])
        
        # Project to consistent dimension
        bio_embedding = self.bio_projection(sentence_embedding.squeeze(0))
        bio_embedding = self.layer_norm(bio_embedding)
        
        return bio_embedding
    
    def forward(self, case_text: str, justice_bios: List[str], 
                case_description_path: str = None, justice_bio_paths: List[str] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            case_text: Text description of the case
            justice_bios: List of justice biography texts
            case_description_path: Optional path to case description file (for pre-encoded lookup)
            justice_bio_paths: Optional list of paths to justice biography files (for pre-encoded lookup)
            
        Returns:
            Tensor of shape (4,) with voting percentage predictions
        """
        # Encode case description
        case_embedding = self.encode_case_description(case_text, case_description_path)
        
        # Encode justice biographies
        justice_embeddings = []
        num_real_justices = len(justice_bios)
        
        for i, bio_text in enumerate(justice_bios):
            bio_path = justice_bio_paths[i] if justice_bio_paths and i < len(justice_bio_paths) else None
            justice_embedding = self.encode_justice_bio(bio_text, bio_path)
            justice_embeddings.append(justice_embedding)
        
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
        Make prediction from file paths (as in the dataset format).
        
        Args:
            case_description_path: Path to case description file
            justice_bio_paths: List of paths to justice biography files
            
        Returns:
            Tensor of shape (4,) with voting percentage predictions
        """
        # Check if we have pre-encoded embeddings for all files
        case_path_normalized = os.path.abspath(case_description_path) if case_description_path else None
        bio_paths_normalized = [os.path.abspath(path) if path else None for path in justice_bio_paths]
        
        has_case_preencoded = case_path_normalized and case_path_normalized in self.description_embeddings
        has_all_bios_preencoded = all(path and path in self.bio_embeddings for path in bio_paths_normalized if path)
        
        # If we have all pre-encoded embeddings, we don't need to read files
        if has_case_preencoded and has_all_bios_preencoded:
            # Use dummy texts since we have pre-encoded embeddings
            case_text = ""
            justice_bios = [""] * len(justice_bio_paths)
        else:
            # Read case description
            if case_description_path and os.path.exists(case_description_path):
                with open(case_description_path, 'r', encoding='utf-8') as f:
                    case_text = f.read().strip()
            else:
                case_text = "No case description available."
            
            # Read justice biographies
            justice_bios = []
            for bio_path in justice_bio_paths:
                if bio_path and os.path.exists(bio_path):
                    # Check if we have pre-encoded for this bio
                    bio_path_normalized = os.path.abspath(bio_path)
                    if bio_path_normalized in self.bio_embeddings:
                        justice_bios.append("")  # Dummy text since we have pre-encoded
                    else:
                        with open(bio_path, 'r', encoding='utf-8') as f:
                            bio_text = f.read().strip()
                        justice_bios.append(bio_text)
                else:
                    justice_bios.append("No biography available.")
        
        return self.forward(case_text, justice_bios, case_description_path, justice_bio_paths)
    
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
            'legal_model_name': self.legal_model_name,
            'bio_model_name': self.bio_model_name,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_justices': self.max_justices,
            'max_sequence_length': self.max_sequence_length,
            'num_attention_heads': self.num_attention_heads,
            'use_justice_attention': self.use_justice_attention,
            'use_preencoded_only': self.use_preencoded_only,
            'bio_metadata': self.bio_metadata,
            'description_metadata': self.description_metadata
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu', 
                   bio_embeddings_file: str = None, description_embeddings_file: str = None):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            legal_model_name=checkpoint['legal_model_name'],
            bio_model_name=checkpoint['bio_model_name'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            max_justices=checkpoint['max_justices'],
            max_sequence_length=checkpoint['max_sequence_length'],
            num_attention_heads=checkpoint.get('num_attention_heads', 4),  # Default for backward compatibility
            use_justice_attention=checkpoint.get('use_justice_attention', True),
            bio_embeddings_file=bio_embeddings_file,
            description_embeddings_file=description_embeddings_file,
            use_preencoded_only=checkpoint.get('use_preencoded_only', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    def load_bio_embeddings(self, embeddings_file: str):
        """Load pre-encoded biography embeddings after initialization."""
        if os.path.exists(embeddings_file):
            self._load_bio_embeddings(embeddings_file)
        else:
            print(f"❌ Bio embeddings file not found: {embeddings_file}")
    
    def load_description_embeddings(self, embeddings_file: str):
        """Load pre-encoded case description embeddings after initialization."""
        if os.path.exists(embeddings_file):
            self._load_description_embeddings(embeddings_file)
        else:
            print(f"❌ Description embeddings file not found: {embeddings_file}")
    
    def has_preencoded_embeddings(self) -> Dict[str, bool]:
        """Check what pre-encoded embeddings are available."""
        return {
            'bios': len(self.bio_embeddings) > 0,
            'descriptions': len(self.description_embeddings) > 0,
            'bio_count': len(self.bio_embeddings),
            'description_count': len(self.description_embeddings)
        }
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded embeddings."""
        stats = {
            'bio_embeddings_loaded': len(self.bio_embeddings),
            'description_embeddings_loaded': len(self.description_embeddings),
            'transformer_models_loaded': self.legal_model is not None,
            'use_preencoded_only': self.use_preencoded_only
        }
        
        if self.bio_metadata:
            stats['bio_metadata'] = self.bio_metadata
        if self.description_metadata:
            stats['description_metadata'] = self.description_metadata
            
        return stats


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