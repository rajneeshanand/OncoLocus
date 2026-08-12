
# HTT: Hierarchical Temporal Transformer

# Full model combining Level 1 (ReportEncoder) + Level 2 (TemporalTransformer).

"""
Forward pass:
  1. Flatten all visits across the batch
  2. Encode each report with BioGPT-Large + LoRA  (Level 1)
  3. Reshape back to (batch, seq_len, hidden_dim)
  4. Process the sequence with TemporalTransformer  (Level 2)
  5. Classify: binary progression prediction
"""

import torch
import torch.nn as nn
from .report_encoder import ReportEncoder
from .temporal_transformer import TemporalTransformer


class HTTModel(nn.Module):

    def __init__(
        self,
        # Level 1
        encoder_model="microsoft/BioGPT-Large",
        hidden_dim=512,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_lora=True,
        # Level 2
        n_heads=8,
        n_layers=4,
        ff_dim=2048,
        temporal_dropout=0.1,
        n_cancer_types=5,
        max_seq_len=10,
        causal=False,
        # Head
        n_classes=2,
        classifier_dropout=0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Level 1: report encoder
        self.report_encoder = ReportEncoder(
            model_name=encoder_model,
            hidden_dim=hidden_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_lora,
        )

        # Level 2: temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=temporal_dropout,
            n_cancer_types=n_cancer_types,
            max_seq_len=max_seq_len,
            causal=causal,
        )

        # Classification head (patient-level)
        self.patient_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # Visit-level classification head (for auxiliary loss)
        self.visit_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def encode_reports(self, input_ids, attention_mask):
 
        batch, seq_len, report_len = input_ids.shape

        # Flatten: (batch * seq_len, report_len)
        flat_ids = input_ids.view(batch * seq_len, report_len)
        flat_mask = attention_mask.view(batch * seq_len, report_len)

        # Encode: (batch * seq_len, hidden_dim)
        flat_emb = self.report_encoder(flat_ids, flat_mask)

        # Reshape: (batch, seq_len, hidden_dim)
        return flat_emb.view(batch, seq_len, self.hidden_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
        days_elapsed,
        seq_mask,
        cancer_type_ids,
    ):

        # Level 1: encode each report
        report_embs = self.encode_reports(input_ids, attention_mask)

        # Level 2: temporal modeling
        visit_outputs, patient_summary = self.temporal_transformer(
            report_embs, days_elapsed, cancer_type_ids, seq_mask
        )

        # Classification
        patient_logits = self.patient_classifier(patient_summary)
        visit_logits = self.visit_classifier(visit_outputs)

        return patient_logits, visit_logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nHTT Model Parameters:")
        print(f"  Total:     {total:>12,}")
        print(f"  Trainable: {trainable:>12,}  ({100*trainable/total:.1f}%)")
        frozen = total - trainable
        print(f"  Frozen:    {frozen:>12,}  ({100*frozen/total:.1f}%)")
        return trainable


def build_model(config=None, device="cpu"):
    if config is None:
        config = {}

    model = HTTModel(
        encoder_model=config.get("encoder_model", "microsoft/BioGPT-Large"),
        hidden_dim=config.get("hidden_dim", 512),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        use_lora=config.get("use_lora", True),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 4),
        ff_dim=config.get("ff_dim", 2048),
        temporal_dropout=config.get("temporal_dropout", 0.1),
        n_cancer_types=config.get("n_cancer_types", 5),
        max_seq_len=config.get("max_seq_len", 10),
        causal=config.get("causal", False),
        n_classes=2,
        classifier_dropout=config.get("classifier_dropout", 0.2),
    )

    model = model.to(device)
    model.count_parameters()
    return model
