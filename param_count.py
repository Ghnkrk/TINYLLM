import json
import torch
from Architecture import DecoderOnlyTransformer
import sentencepiece as spm


def load_config(path="./config.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    cfg = load_config()

    # Load tokenizer to get vocab size
    sp = spm.SentencePieceProcessor()
    sp.load(cfg["tokenizer"]["model_path"])
    vocab_size = sp.get_piece_size()

    mcfg = cfg["model"]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ffn=mcfg["d_ffn"],
        max_len=mcfg["max_len"]
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n📊 Model Parameter Summary")
    print("-------------------------")
    print(f"Experiment      : {cfg['experiment_name']}")
    print(f"Vocab size      : {vocab_size}")
    print(f"Total params    : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")


if __name__ == "__main__":
    main()

