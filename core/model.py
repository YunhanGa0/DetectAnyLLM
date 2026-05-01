import os

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


def from_pretrained(cls, model_name, kwargs, cache_dir):
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)


class MeanPooler(nn.Module):
    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / denom


class DiscrepancyEstimator(nn.Module):
    def __init__(
        self,
        scoring_model_name=None,
        scoring_model=None,
        scoring_tokenizer=None,
        cache_dir="./model/",
        pretrained_ckpt=None,
        num_labels=6,
        dropout=0.1,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.num_labels = num_labels

        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt)
            return

        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.backbone = from_pretrained(
                AutoModel,
                scoring_model_name,
                kwargs={"torch_dtype": torch.float16, "trust_remote_code": True},
                cache_dir=cache_dir,
            )
            self.scoring_tokenizer = from_pretrained(
                AutoTokenizer,
                scoring_model_name,
                kwargs={
                    "padding_side": "right",
                    "use_fast": True if "facebook/opt-" not in scoring_model_name else False,
                    "trust_remote_code": True,
                },
                cache_dir=cache_dir,
            )
        else:
            if scoring_model is None or scoring_tokenizer is None:
                raise ValueError(
                    "You should provide scoring_model_name or scoring_model and scoring_tokenizer."
                )
            self.backbone = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = self.backbone.config._name_or_path

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        hidden_size = self.backbone.config.hidden_size
        self.pooler = MeanPooler()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.regressor_lir = nn.Linear(hidden_size, 1)
        self.regressor_jaccard = nn.Linear(hidden_size, 1)
        self.regressor_sentence_jaccard = nn.Linear(hidden_size, 1)
        self.dropout_p = dropout
        self.lora_config = None

    def add_lora_config(self, lora_config: LoraConfig):
        self.lora_config = lora_config
        if getattr(lora_config, "task_type", None) is None:
            lora_config.task_type = TaskType.FEATURE_EXTRACTION
        self.backbone = get_peft_model(self.backbone, lora_config)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.scoring_tokenizer.save_pretrained(save_directory)
        torch.save(
            {
                "scoring_model_name": self.scoring_model_name,
                "num_labels": self.num_labels,
                "dropout_p": self.dropout_p,
                "lora_config": self.lora_config.to_dict() if self.lora_config is not None else None,
                "state_dict": self.state_dict(),
            },
            os.path.join(save_directory, "checkpoint.pt"),
        )

    def load_pretrained(self, load_directory):
        checkpoint_path = os.path.join(load_directory, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Directory {load_directory} does not contain checkpoint.pt")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.scoring_model_name = checkpoint["scoring_model_name"]
        self.num_labels = checkpoint.get("num_labels", 6)
        self.dropout_p = checkpoint.get("dropout_p", 0.1)

        self.backbone = from_pretrained(
            AutoModel,
            self.scoring_model_name,
            kwargs={"torch_dtype": torch.float16, "trust_remote_code": True},
            cache_dir=self.cache_dir,
        )
        lora_config_dict = checkpoint.get("lora_config")
        self.lora_config = None
        if lora_config_dict is not None:
            self.add_lora_config(LoraConfig(**lora_config_dict))

        self.scoring_tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        hidden_size = self.backbone.config.hidden_size
        self.pooler = MeanPooler()
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.regressor_lir = nn.Linear(hidden_size, 1)
        self.regressor_jaccard = nn.Linear(hidden_size, 1)
        self.regressor_sentence_jaccard = nn.Linear(hidden_size, 1)
        self.load_state_dict(checkpoint["state_dict"])

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        features = self.dropout(pooled)
        logits = self.classifier(features)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "pred_class": torch.argmax(logits, dim=-1),
            "pred_lir": self.regressor_lir(features).squeeze(-1),
            "pred_jaccard": self.regressor_jaccard(features).squeeze(-1),
            "pred_sentence_jaccard": self.regressor_sentence_jaccard(features).squeeze(-1),
        }
