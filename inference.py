import json
import argparse
import math
import time

import torch
import torch.nn.functional as F
import sentencepiece as spm

from Architecture import DecoderOnlyTransformer


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(path="./config/config.json"):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Sampling Helpers
# -------------------------------------------------

def top_k_filter(logits, k):
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    cutoff = values[:, -1].unsqueeze(-1)

    logits = torch.where(
        logits < cutoff,
        torch.full_like(logits, -1e10),
        logits
    )

    return logits


def top_p_filter(logits, p):

    if p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(
        probs,
        descending=True
    )

    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > p
    mask[:, 0] = False

    sorted_probs[mask] = 0

    new_logits = torch.full_like(logits, -1e10)

    new_logits.scatter_(
        1,
        sorted_idx,
        torch.log(sorted_probs + 1e-12)
    )

    return new_logits


# -------------------------------------------------
# Generation
# -------------------------------------------------

@torch.no_grad()
def generate_stream(
    model,
    sp,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    device
):

    model.eval()

    ids = sp.encode(prompt, out_type=int)

    prompt_tokens = len(ids)

    x = torch.tensor([ids], dtype=torch.long).to(device)

    generated_ids = []

    start = time.perf_counter()

    for _ in range(max_new_tokens):

        logits = model(x)[:, -1, :]

        if repetition_penalty != 1.0:
            for t in set(x[0].tolist()):
                logits[0, t] /= repetition_penalty

        logits = logits / max(temperature, 1e-6)

        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)

        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, 1)

        token = next_id.item()

        if token == sp.eos_id():
            break

        generated_ids.append(token)

        x = torch.cat([x, next_id], dim=1)

    elapsed = time.perf_counter() - start

    generated_text = sp.decode(generated_ids)

    print()
    print(prompt + generated_text)
    print()

    return (
        prompt_tokens,
        len(generated_ids),
        elapsed
    )

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    cfg = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", required=True)

    parser.add_argument(
        "--ckpt",
        default="best_model.pt"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=cfg["inference"]["max_new_tokens"]
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=cfg["inference"]["temperature"]
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=cfg["inference"]["top_k"]
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=cfg["inference"]["top_p"]
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=cfg["inference"]["repetition_penalty"]
    )

    args = parser.parse_args()


    # ---------------------------
    # Tokenizer
    # ---------------------------

    sp = spm.SentencePieceProcessor()

    sp.load(cfg["tokenizer"]["model_path"])

    vocab_size = sp.get_piece_size()


    # ---------------------------
    # Model
    # ---------------------------

    mcfg = cfg["model"]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ffn=mcfg["d_ffn"],
        max_len=mcfg["max_len"]
    )

    checkpoint = torch.load(
        args.ckpt,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    print(
        f"\n🚀 Inference : {checkpoint['experiment_name']}"
    )


    prompt_tokens, generated_tokens, elapsed = generate_stream(
        model=model,
        sp=sp,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )


    total_params = checkpoint["total_params"]

    val_loss = checkpoint["best_val_loss"]

    perplexity = checkpoint["val_perplexity"]

    print("============== Benchmark ==============")

    print(f"Parameters        : {total_params/1e6:.2f} M")

    print(f"Validation Loss   : {val_loss:.4f}")

    print(f"Perplexity        : {perplexity:.2f}")

    print(f"Prompt Tokens     : {prompt_tokens}")

    print(f"Generated Tokens  : {generated_tokens}")

    print(f"Generation Time   : {elapsed:.2f} sec")

    print(
        f"Tokens / Second   : {generated_tokens/elapsed:.2f}"
    )

    print(
        f"Latency / Token   : {(elapsed/generated_tokens)*1000:.2f} ms"
    )

    print(f"Device            : {device}")

    print("=======================================\n")


if __name__ == "__main__":
    main()