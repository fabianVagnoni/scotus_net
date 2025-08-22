import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pickle
from sentence_transformers import SentenceTransformer
import copy
from typing import Dict, List, Any
import os

class ContrastiveJustice(nn.Module):
    def __init__(
        self,
        trunc_bio_tokenized_file: str,
        full_bio_tokenized_file: str,
        model_name: str,
        dropout_rate: float = 0.1,
        use_noise_reg: bool = True,
        noise_reg_alpha: float = 5.0,
        embedding_dropout_rate: float = 0.0,
    ):
        super(ContrastiveJustice, self).__init__()
        self.truncated_bio_model = SentenceTransformer(model_name)
        self.full_bio_model = copy.deepcopy(self.truncated_bio_model)
        for param in self.truncated_bio_model.parameters():
            param.requires_grad = True
        for param in self.full_bio_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.use_noise_reg = use_noise_reg
        self.noise_reg_alpha = float(noise_reg_alpha)
        self.embedding_dropout_rate = float(embedding_dropout_rate)

        # Obtain padding id from tokenizer when available, fallback to 0
        self.padding_value = 0
        try:
            tokenizer = getattr(self.truncated_bio_model, 'tokenizer', None)
            if tokenizer is not None:
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                if pad_id is not None:
                    self.padding_value = int(pad_id)
        except Exception:
            # Keep default padding_value = 0 if anything goes wrong
            pass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Register lightweight embedding hooks to inject NEFTune-style noise at token embedding level (train-only)
        self._register_embedding_noise(self.truncated_bio_model)
        self._register_embedding_noise(self.full_bio_model)
        
    def to(self, device):
        """Override to method to ensure all submodules are moved to device."""
        super().to(device)
        self.device = device
        self.truncated_bio_model = self.truncated_bio_model.to(device)
        self.full_bio_model = self.full_bio_model.to(device)
        return self

    def _register_embedding_noise(self, sbert_model: SentenceTransformer) -> None:
        """Attach a forward hook on the underlying HF embedding layer to add noise during training.
        This keeps changes minimal and avoids touching internal forward graphs.
        """
        try:
            # First module is the Transformer wrapper with .auto_model
            transformer_module = sbert_model._first_module()
            hf_model = getattr(transformer_module, 'auto_model', None)
            if hf_model is None:
                return
            embeddings_layer = getattr(hf_model, 'embeddings', None)
            if embeddings_layer is None:
                return

            def _noise_hook(module, inputs, output):
                # Apply only when training and enabled
                if self.training:
                    d = output.size(-1)
                    if d == 0:
                        return output
                    # NEFTune-style noise (if enabled)
                    if self.use_noise_reg:
                        scale = self.noise_reg_alpha / (d ** 0.5)
                        noise = torch.empty_like(output).uniform_(-1, 1) * scale
                        output = output + noise
                    # Embedding-dimension dropout (drop entire embedding dims across batch and seq)
                    if self.embedding_dropout_rate > 0.0:
                        keep_prob = 1.0 - self.embedding_dropout_rate
                        mask = output.new_empty(1, 1, d).bernoulli_(keep_prob).div_(keep_prob)
                        output = output * mask
                return output

            embeddings_layer.register_forward_hook(_noise_hook)
        except Exception:
            # If anything changes in internals, fail silently without breaking training
            pass
        
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
        
        # Get sentence embeddings from models using SentenceTransformer API
        trunc_bio_outputs = self.truncated_bio_model({
            'input_ids': trunc_input_ids,
            'attention_mask': trunc_attention_masks
        })
        # Full (teacher) path does not require gradients – save memory
        with torch.no_grad():
            full_bio_outputs = self.full_bio_model({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_masks
            })

        # Extract pooled sentence embeddings
        out_t = trunc_bio_outputs['sentence_embedding']
        out_f = full_bio_outputs['sentence_embedding']

        # Apply NEFTune-style noise regularization only during training
        if self.use_noise_reg and self.training:
            out_t = self.noise_reg(out_t)
            out_f = self.noise_reg(out_f)

        # Apply dropout to embeddings (nn.Dropout is a no-op during eval)
        out_t = self.dropout(out_t)
        out_f = self.dropout(out_f)
        
        return out_t, out_f

    def noise_reg(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply simple NEFTune-like noise regularization (train-time only)."""
        alpha = self.noise_reg_alpha
        d = embedding.size(-1)
        U = torch.empty_like(embedding).uniform_(-1, 1)
        scale = alpha / (d ** 0.5)
        noise = U * scale
        return embedding + noise



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


