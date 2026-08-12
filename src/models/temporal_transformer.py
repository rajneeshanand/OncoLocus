
# Level 2: Temporal Transformer  (our NOVEL contribution)

"""
Key innovations over prior work:
  1. Continuous-time positional encoding using days elapsed
     (not discrete visit indices)
  2. Causal masking - only attends to past visits, mimicking
     clinical decision-making at each time point
  3. Cancer-type conditioning via learnable type embeddings

Input:  sequence of (batch, max_seq_len, hidden_dim) report embeddings
        + (batch, max_seq_len) days_elapsed + (batch,) cancer_type_ids
Output: (batch, max_seq_len, hidden_dim) contextualised embeddings
        OR (batch, hidden_dim) patient-level summary
"""

import math
import torch
import torch.nn as nn


class ContinuousTimeEncoding(nn.Module):

    def __init__(self, hidden_dim, max_years=10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_years = max_years

        # Learnable frequency scaling — the model can adjust how
        # sensitive it is to different time scales
        self.freq_scale = nn.Parameter(torch.ones(hidden_dim // 2))

    def forward(self, days_elapsed):

        t = days_elapsed.unsqueeze(-1)  # (batch, seq_len, 1)
        d = self.hidden_dim // 2

        # Frequencies: scaled version of standard sinusoidal PE
        freqs = torch.exp(
            torch.arange(d, device=days_elapsed.device).float()
            * (-math.log(self.max_years * 2) / d)
        )
        freqs = freqs * self.freq_scale.abs()  # learnable scaling

        angles = t * freqs.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, d)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.hidden_dim % 2 == 1:
            enc = torch.cat([enc, torch.zeros_like(enc[..., :1])], dim=-1)

        return enc  # (batch, seq_len, hidden_dim)


class TemporalTransformer(nn.Module):
    """
     Architecture:
      1. Time encoding added to report embeddings
      2. Cancer type embedding added (for cross-cancer generalization)
      3. Multi-layer transformer encoder (with causal mask option)
      4. Final linear head for binary classification
    """

    def __init__(
        self,
        hidden_dim=512,
        n_heads=8,
        n_layers=4,
        ff_dim=2048,
        dropout=0.1,
        n_cancer_types=5,
        max_seq_len=10,
        causal=False,  # set True for visit-by-visit prediction
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.causal = causal

        # Time encoding
        self.time_encoder = ContinuousTimeEncoding(hidden_dim)

        # Cancer type embedding (key for Gap 2 cross-cancer transfer)
        self.cancer_type_emb = nn.Embedding(n_cancer_types + 1, hidden_dim)

        # Input projection + norm
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-LN for training stability
        )
        # norm_first=True disables nested tensor optimization — suppress misleading warning
        encoder_layer.self_attn.batch_first = True
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Patient-level aggregation: learnable [CLS]-like query
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def _make_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, report_embeddings, days_elapsed, cancer_type_ids, seq_mask):

        batch_size, seq_len, _ = report_embeddings.shape

        # 1. Add temporal encoding
        time_enc = self.time_encoder(days_elapsed)
        x = report_embeddings + time_enc

        # 2. Add cancer type conditioning (broadcast over seq_len)
        cancer_emb = self.cancer_type_emb(cancer_type_ids)  # (batch, hidden_dim)
        x = x + cancer_emb.unsqueeze(1)

        # 3. Prepend learnable CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat([cls, x], dim=1)  # (batch, 1+seq_len, hidden_dim)

        # Extend seq_mask to include CLS position
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, seq_mask], dim=1)  # (batch, 1+seq_len)

        # 4. Norm
        x = self.input_norm(x)

        # 5. Causal mask (optional)
        attn_mask = None
        if self.causal:
            attn_mask = self._make_causal_mask(1 + seq_len, x.device)

        # 6. Transformer — key_padding_mask: True = IGNORE that position
        padding_mask = ~full_mask  # (batch, 1+seq_len)
        out = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)

        # 7. Split CLS token from visit outputs
        patient_summary = out[:, 0, :]          # (batch, hidden_dim)
        visit_outputs = out[:, 1:, :]           # (batch, seq_len, hidden_dim)

        return visit_outputs, patient_summary
