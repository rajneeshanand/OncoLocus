
# Level 1: Report Encoder; Encodes each individual radiology report into a fixed-size embedding
# using BioGPT-Large with LoRA fine-tuning.



import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType


def _get_lora_target_modules(model_name, model):
  
    name_lower = model_name.lower()

    # Known GPT-style models
    if any(x in name_lower for x in ["biogpt", "gpt", "llama", "mistral", "falcon"]):
        candidates = ["q_proj", "v_proj"]
    # Known BERT-style models
    elif any(x in name_lower for x in ["bert", "biomedbert", "biomedlm", "roberta"]):
        candidates = ["query", "value"]
    else:
        candidates = ["query", "value"]  # safe BERT default

    # Verify against actual parameter names, fall back if not found
    param_names = {n for n, _ in model.named_modules()}
    found = [m for m in candidates if any(m in n for n in param_names)]
    if found:
        return found

    # Last resort: find any attention projection layers
    for suffix in ["query", "value", "q_proj", "v_proj", "q", "v"]:
        if any(n.endswith(suffix) for n in param_names):
            return [suffix]

    raise ValueError(
        f"Could not find LoRA target modules for {model_name}. "
        f"Set target_modules manually."
    )


class ReportEncoder(nn.Module):


    def __init__(
        self,
        model_name="microsoft/BioGPT-Large",
        hidden_dim=512,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_lora=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )

        if use_lora:
            # Target module names differ by architecture
            target_modules = _get_lora_target_modules(model_name, base_model)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(base_model, lora_config)
        else:
            self.encoder = base_model

        # Project from model's hidden size to our hidden_dim
        model_hidden = config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(model_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: (batch, seq_len, model_hidden)
        hidden = outputs.last_hidden_state

        # Mean pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        pooled = sum_hidden / count  # (batch, model_hidden)

        return self.projection(pooled)  # (batch, hidden_dim)

    def print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
