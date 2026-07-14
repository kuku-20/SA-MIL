"""
Explainability wrapper for StructureAwareMIL.
Provides ResNetMIL_Explainer that returns (logits, attn_weights) for visualization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F
from src.models.mil_v12 import ResNetMIL

# Re-export config constants
FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12


class ResNetMIL_Explainer(ResNetMIL):
    """Wrapper around ResNetMIL that exposes attention weights for explainability."""

    def forward(self, windows, masks):
        B, N, c, h, w = windows.shape
        x = self.features(windows.view(B * N, c, h, w))
        x = self.avgpool(x).flatten(1)
        x = self.proj(x).view(B, N, -1)

        x = x + self.pos_embed[:, :N, :]
        padding_mask = ~masks
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        attn_scores = self.attention(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~masks, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights
