import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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
        self.padding_value = self.truncated_bio_model.config.pad_token_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def to(self, device):
        """Override to method to ensure all submodules are moved to device."""
        super().to(device)
        self.device = device
        self.truncated_bio_model = self.truncated_bio_model.to(device)
        self.full_bio_model = self.full_bio_model.to(device)
        return self
        
    def forward(self, trunc_bio_data: List[str], full_bio_data: List[str]):
        """
        Forward pass for contrastive learning.
        
        Args:
            justice_trunc_bio_paths: List of tokens of truncated biography
            justice_full_bio_paths: List of tokens of full biography
            
        Returns:
            Tuple of (truncated_embeddings, full_embeddings)
        """
        batch_size = len(trunc_bio_data)
        
        # Process truncated biographies
        trunc_input_ids = []
        trunc_attention_masks = []
        
        for data in trunc_bio_data:
            # Ensure input tensors are on the correct device
            input_ids = data['input_ids'].to(self.device) if hasattr(data['input_ids'], 'to') else data['input_ids']
            attention_mask = data['attention_mask'].to(self.device) if hasattr(data['attention_mask'], 'to') else data['attention_mask']
            trunc_input_ids.append(input_ids)
            trunc_attention_masks.append(attention_mask)
        
        # Pad sequences for batch processing
        trunc_input_ids = pad_sequence(trunc_input_ids, batch_first=True, padding_value=self.padding_value)
        trunc_attention_masks = pad_sequence(trunc_attention_masks, batch_first=True, padding_value=0)

        # Process full biographies
        full_input_ids = []
        full_attention_masks = []
        
        for data in full_bio_data:
            # Ensure input tensors are on the correct device
            input_ids = data['input_ids'].to(self.device) if hasattr(data['input_ids'], 'to') else data['input_ids']
            attention_mask = data['attention_mask'].to(self.device) if hasattr(data['attention_mask'], 'to') else data['attention_mask']
            full_input_ids.append(input_ids)
            full_attention_masks.append(attention_mask)
        
        # Pad sequences for batch processing
        full_input_ids = pad_sequence(full_input_ids, batch_first=True, padding_value=self.padding_value)
        full_attention_masks = pad_sequence(full_attention_masks, batch_first=True, padding_value=0)
        
        # Move tensors to device (in case pad_sequence created new tensors)
        trunc_input_ids = trunc_input_ids.to(self.device)
        trunc_attention_masks = trunc_attention_masks.to(self.device)
        full_input_ids = full_input_ids.to(self.device)
        full_attention_masks = full_attention_masks.to(self.device)
        
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


