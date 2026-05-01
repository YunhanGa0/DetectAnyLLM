import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


RATIO_TO_CLASS = {
    0.0: 0,
    0.2: 1,
    0.4: 2,
    0.6: 3,
    0.8: 4,
    1.0: 5,
}

CLASS_TO_RATIO = {value: key for key, value in RATIO_TO_CLASS.items()}

TEXT_KEYS = (
    "mixed_text",
    "text",
    "rewritten",
    "content",
)

ID_KEYS = ("id", "sample_id", "uid")
SENTENCE_JACCARD_KEYS = (
    "sentence_jaccard",
    "sentence_jaccard_distance",
    "Sentence Jaccard",
    "Sentence_Jaccard",
    "cosine_distance",
)


def _load_records(data_path):
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, dict) and "original" in data and "rewritten" in data:
        records = []
        for idx, (original, rewritten) in enumerate(zip(data["original"], data["rewritten"])):
            records.append(
                {
                    "id": idx,
                    "text": rewritten,
                    "target_ai_ratio": 1.0,
                    "lir": 1.0,
                    "jaccard_distance": 1.0,
                    "sentence_jaccard": 1.0,
                    "original_text": original,
                }
            )
        return records
    raise ValueError(f"Unsupported data format for {data_path}")


def _pick_first(record, keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _to_float(value, field_name):
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")
    return float(value)


def _ratio_to_class(target_ai_ratio):
    ratio = round(float(target_ai_ratio), 1)
    if ratio not in RATIO_TO_CLASS:
        raise ValueError(
            f"Unsupported target_ai_ratio={target_ai_ratio}. Expected one of {sorted(RATIO_TO_CLASS)}"
        )
    return RATIO_TO_CLASS[ratio]


class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.data_path = data_path
        self.data = _load_records(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        record = self.data[index]
        text = _pick_first(record, TEXT_KEYS)
        if text is None:
            raise ValueError(f"Sample {index} is missing a text field. Tried keys: {TEXT_KEYS}")

        target_ai_ratio = _to_float(record.get("target_ai_ratio"), "target_ai_ratio")
        item = {
            "id": _pick_first(record, ID_KEYS, default=index),
            "text": text,
            "label": _ratio_to_class(target_ai_ratio),
            "target_ai_ratio": target_ai_ratio,
            "lir": _to_float(record.get("lir"), "lir"),
            "jaccard": _to_float(
                _pick_first(record, ("jaccard_distance", "jaccard")), "jaccard_distance"
            ),
            "sentence_jaccard": _to_float(
                _pick_first(record, SENTENCE_JACCARD_KEYS), "sentence_jaccard"
            ),
        }
        if "original_text" in record:
            item["original_text"] = record["original_text"]
        return item

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        return {
            "ids": [item["id"] for item in batch],
            "texts": texts,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "target_ai_ratio": torch.tensor(
                [item["target_ai_ratio"] for item in batch], dtype=torch.float
            ),
            "lir": torch.tensor([item["lir"] for item in batch], dtype=torch.float),
            "jaccard": torch.tensor([item["jaccard"] for item in batch], dtype=torch.float),
            "sentence_jaccard": torch.tensor(
                [item["sentence_jaccard"] for item in batch], dtype=torch.float
            ),
        }

    def __len__(self):
        return len(self.data)
