import json
import os
from datasets import load_dataset
import sentencepiece as spm



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------
# Load config
# -------------------------------------------------

def load_config():
    # config.json is one level above tokenizer/
    config_path = os.path.join(BASE_DIR, "..", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Load raw dataset
# -------------------------------------------------

def load_texts(cfg):
    name = cfg["dataset"]["name"]

    if name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text_col = "text"

    elif name == "simplewiki":
        ds = load_dataset("wikipedia", "20220301.simple", split="train")
        text_col = "text"

    else:
        raise ValueError(f"Unknown dataset: {name}")

    max_samples = cfg["dataset"].get("max_samples")
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    texts = []
    min_len = cfg["dataset"].get("min_text_length", 20)

    for ex in ds:
        t = ex[text_col].replace("\n", " ").strip()
        if len(t) >= min_len:
            texts.append(t)

    return texts


# -------------------------------------------------
# Train SentencePiece tokenizer
# -------------------------------------------------

def train_tokenizer():
    cfg = load_config()

    os.makedirs(BASE_DIR, exist_ok=True)

    vocab_size = cfg["tokenizer"]["vocab_size"]
    model_name = os.path.basename(cfg["tokenizer"]["model_path"]).replace(".model", "")
    model_prefix = os.path.join(BASE_DIR, model_name)

    print("\n🔤 Training SentencePiece tokenizer")
    print(f"Dataset      : {cfg['dataset']['name']}")
    print(f"Vocab size   : {vocab_size}")
    print(f"Save prefix  : {model_prefix}\n")

    texts = load_texts(cfg)

    # write temporary corpus inside tokenizer/
    corpus_path = os.path.join(BASE_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=cfg["tokenizer"].get("model_type", "bpe"),
        character_coverage=cfg["tokenizer"].get("character_coverage", 1.0),
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        bos_id=-1,
        eos_id=-1,
        pad_id=0,
        unk_id=1
    )

    os.remove(corpus_path)
    print("✅ Tokenizer training complete\n")


if __name__ == "__main__":
    train_tokenizer()
