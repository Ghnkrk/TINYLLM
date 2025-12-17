# data.py
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# -------------------------------------------------
# Load config
# -------------------------------------------------

def load_config(path="./config/config_20M.json"):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Dataset loader
# -------------------------------------------------

def load_raw_dataset(cfg):
    name = cfg["dataset"]["name"]

    if name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text_col = "text"

    elif name == "simplewiki":
        ds = load_dataset("wikimedia/wikipedia","20231101.simple",split="train")
        text_col = "text"

    else:
        raise ValueError(f"Unknown dataset: {name}")

    max_samples = cfg["dataset"].get("max_samples")
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    return ds, text_col


# -------------------------------------------------
# Text cleaning
# -------------------------------------------------

def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()


# -------------------------------------------------
# Tokenize + chunk (autoregressive)
# -------------------------------------------------

def tokenize_and_chunk(dataset, text_col, tokenizer, block_size):
    tokens = []

    for ex in dataset:
        text = clean_text(ex[text_col])
        if len(text) < 20:
            continue
        tokens.extend(tokenizer.encode(text, out_type=int))

    input_ids, labels = [], []

    for i in range(0, len(tokens) - block_size, block_size):
        chunk = tokens[i : i + block_size + 1]
        input_ids.append(chunk[:-1])
        labels.append(chunk[1:])

    return input_ids, labels


# -------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------

class LMDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# -------------------------------------------------
# Public API
# -------------------------------------------------

def build_dataset(tokenizer, config_path="./config/config.json"):
    cfg = load_config(config_path)

    raw_ds, text_col = load_raw_dataset(cfg)
    block_size = cfg["training"]["block_size"]

    input_ids, labels = tokenize_and_chunk(
        raw_ds,
        text_col,
        tokenizer,
        block_size
    )

    return LMDataset(input_ids, labels)





