import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class BaselineSCOTUSModel(nn.Module):
    """
    Minimal baseline model for SCOTUS voting.

    - Encodes case descriptions with a SentenceTransformer
    - Feeds the embedding into a small feed-forward network
    - Outputs logits reshaped to (batch_size, 3)
      where the 3 channels are [in_favor, against, absent]
    """

    def __init__(
        self,
        description_model_name: str,
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.frozen_description_model = True

        device_arg = device if isinstance(device, str) else str(device)
        self.description_model = SentenceTransformer(description_model_name, device=device_arg)

        # Project to expected embedding_dim if needed
        desc_actual_dim = self.description_model.get_sentence_embedding_dimension()
        self.description_projection_layer: Optional[nn.Module] = None
        if desc_actual_dim != embedding_dim:
            self.description_projection_layer = nn.Linear(desc_actual_dim, embedding_dim)

        # Small feed-forward head to 3 * num_justices logits
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Freeze sentence transformer by default (can be unfrozen later)
        self.freeze_description_model()

    def forward(
        self,
        case_input_ids: torch.Tensor,
        case_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            case_input_ids: (batch_size, seq_len)
            case_attention_mask: (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, 3)
        """
        features = {"input_ids": case_input_ids, "attention_mask": case_attention_mask}
        with torch.no_grad() if not self.training or self.frozen_description_model else torch.enable_grad():
            emb = self.description_model(features)["sentence_embedding"]
            if self.description_projection_layer is not None:
                emb = self.description_projection_layer(emb)

        logits = self.fc(emb)  # (batch_size, 3)
        return logits

    @torch.no_grad()
    def predict_probabilities(
        self,
        case_input_ids: torch.Tensor,
        case_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns probabilities over 3 classes.

        Shape: (batch_size, 3)
        """
        logits = self.forward(case_input_ids, case_attention_mask)
        probs = F.softmax(logits, dim=1)  # softmax over the 3 classes
        return probs

    def freeze_description_model(self) -> None:
        self.frozen_description_model = True
        for p in self.description_model.parameters():
            p.requires_grad = False

    def unfreeze_description_model(self) -> None:
        self.frozen_description_model = False
        for p in self.description_model.parameters():
            p.requires_grad = True


