
#Baseline: Non-Temporal Model  

#We compare: Baseline (last report only)  vs  HTT (full temporal sequence)


import torch
import torch.nn as nn
from .report_encoder import ReportEncoder 


class BaselineModel(nn.Module):

    def __init__(
        self,
        encoder_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        hidden_dim=512,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_lora=True,
        n_classes=2,
        classifier_dropout=0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.report_encoder = ReportEncoder(
            model_name=encoder_model,
            hidden_dim=hidden_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_lora,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input_ids, attention_mask, days_elapsed, seq_mask, cancer_type_ids):
 
        batch, seq_len, report_len = input_ids.shape

        # Flatten and encode all reports
        flat_ids = input_ids.view(batch * seq_len, report_len)
        flat_mask = attention_mask.view(batch * seq_len, report_len)
        flat_emb = self.report_encoder(flat_ids, flat_mask)
        all_embs = flat_emb.view(batch, seq_len, self.hidden_dim)

        # Find last real visit index for each patient
        # seq_mask: (batch, seq_len) bool — True = real visit
        last_idx = seq_mask.long().sum(dim=1) - 1  # (batch,)
        last_idx = last_idx.clamp(min=0)

        # Gather last visit embedding
        last_emb = all_embs[torch.arange(batch), last_idx]  # (batch, hidden_dim)

        patient_logits = self.classifier(last_emb)

        # Return dummy visit_logits with same shape as HTT for trainer compatibility
        visit_logits = self.classifier(all_embs.view(batch * seq_len, self.hidden_dim))
        visit_logits = visit_logits.view(batch, seq_len, -1)

        return patient_logits, visit_logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nBaseline Model Parameters:")
        print(f"  Total:     {total:>12,}")
        print(f"  Trainable: {trainable:>12,}  ({100*trainable/total:.1f}%)")
        return trainable
