import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import AutoModel
import copy
from typing import Dict, List, Any
import os

class ContrastiveJustice(nn.Module):
    def __init__(self, trunc_bio_tokenized_file: str, full_bio_tokenized_file: str, model_name: str, dropout_rate: float = 0.1):
        super(ContrastiveJustice, self).__init__()
        self.truncated_bio_model = AutoModel.from_pretrained(model_name)
        self.full_bio_model = copy.deepcopy(self.truncated_bio_model)
        for param in self.truncated_bio_model.parameters():
            param.requires_grad = True
        for param in self.full_bio_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, trunc_bio_data: List[str], full_bio_data: List[str]):
        """
        Forward pass for contrastive learning.
        
        Args:
            justice_trunc_bio_paths: List of paths to truncated biography files
            justice_full_bio_paths: List of paths to full biography files
            
        Returns:
            Tuple of (truncated_embeddings, full_embeddings)
        """
        batch_size = len(trunc_bio_data)
        
        # Process truncated biographies
        trunc_input_ids = []
        trunc_attention_masks = []
        
        for data in trunc_bio_data:
            trunc_input_ids.append(data['input_ids'])
            trunc_attention_masks.append(data['attention_mask'])
        
        # Pad sequences for batch processing
        trunc_input_ids = torch.stack(trunc_input_ids)
        trunc_attention_masks = torch.stack(trunc_attention_masks)
        
        # Process full biographies
        full_input_ids = []
        full_attention_masks = []
        
        for data in full_bio_data:
            full_input_ids.append(data['input_ids'])
            full_attention_masks.append(data['attention_mask'])
        
        # Pad sequences for batch processing
        full_input_ids = torch.stack(full_input_ids)
        full_attention_masks = torch.stack(full_attention_masks)
        
        # Get embeddings from models
        trunc_bio_outputs = self.truncated_bio_model(
            input_ids=trunc_input_ids, 
            attention_mask=trunc_attention_masks
        )
        full_bio_outputs = self.full_bio_model(
            input_ids=full_input_ids, 
            attention_mask=full_attention_masks
        )

        # Extract CLS token embeddings
        out_t = trunc_bio_outputs.last_hidden_state[:, 0]  # CLS token
        out_f = full_bio_outputs.last_hidden_state[:, 0]   # CLS token
        
        return out_t, out_f



def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    
    Args:
        batch: List of dictionaries with 'justice_trunc_bio_paths' and 'justice_full_bio_paths'
        
    Returns:
        Dictionary with batched data
    """
    trunc_bio_data = [item[0] for item in batch]
    full_bio_data = [item[1] for item in batch]
    
    return {
        'trunc_bio_data': trunc_bio_data,
        'full_bio_data': full_bio_data,
    } 


class ContrastiveJusticeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for contrastive justice learning.
    
    Maps justice IDs to their truncated and full biography paths.
    """
    
    def __init__(self, justices_ids: List[str], trunc_bio_tokenized_file: str, full_bio_tokenized_file: str):
        """
        Initialize dataset.
        
        Args:
            justices_ids: List of justice IDs
            trunc_bio_tokenized_file: Path to the truncated bio tokenized file
            full_bio_tokenized_file: Path to the full bio tokenized file
        """
        self.justice_ids = justices_ids
        self.trunc_bio_tokenized_file = trunc_bio_tokenized_file
        self.full_bio_tokenized_file = full_bio_tokenized_file
        self.trunc_tokenized_data = pickle.load(open(self.trunc_bio_tokenized_file, "rb"))
        self.full_tokenized_data = pickle.load(open(self.full_bio_tokenized_file, "rb"))

    def __len__(self):
        return len(self.justice_ids)
    
    def __getitem__(self, idx):
        justice_id = self.justice_ids[idx]
        trunc_bio_data = self.trunc_tokenized_data["tokenized_data"][justice_id]
        full_bio_data = self.full_tokenized_data["tokenized_data"][justice_id]

        return trunc_bio_data, full_bio_data


