import json
import math
import os
import time
import getpass

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import sentencepiece as spm

from Architecture import DecoderOnlyTransformer
from data import build_dataset


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(path="./config/config.json"):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Collate
# -------------------------------------------------

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return input_ids, labels


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    cfg = load_config()

    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 Using device : {device}\n")


    # -------------------------------------------------
    # W&B
    # -------------------------------------------------

    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = getpass.getpass(
            "Enter your Weights & Biases API key: "
        )

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment_name"],
        config=cfg
    )


    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    sp = spm.SentencePieceProcessor()

    sp.load(cfg["tokenizer"]["model_path"])

    vocab_size = sp.get_piece_size()

    print(f"Tokenizer Vocabulary : {vocab_size}")


    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = build_dataset(
        tokenizer=sp,
        config_path="./config/config.json"
    )

    val_size = int(0.02 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Train Samples : {len(train_set):,}")
    print(f"Val Samples   : {len(val_set):,}")


    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    mcfg = cfg["model"]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ffn=mcfg["d_ffn"],
        max_len=mcfg["max_len"]
    ).to(device)


    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nModel Parameters : {total_params/1e6:.2f} M\n")


    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"])
    )


    # -------------------------------------------------
    # LR Scheduler
    # -------------------------------------------------

    total_steps_est = (
        train_cfg["epochs"] *
        len(train_loader)
    )

    max_steps = (
        train_cfg["max_steps"]
        if train_cfg["max_steps"]
        else total_steps_est
    )

    warmup_steps = int(
        train_cfg["warmup_ratio"] *
        max_steps
    )


    def lr_schedule(step):

        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (
            (step - warmup_steps)
            /
            max(1, max_steps - warmup_steps)
        )

        return 0.5 * (
            1 +
            math.cos(math.pi * progress)
        )


    scaler = torch.cuda.amp.GradScaler(
        enabled=train_cfg["mixed_precision"]
    )


    # -------------------------------------------------
    # Early stopping
    # -------------------------------------------------

    early_cfg = train_cfg["early_stopping"]

    eval_every_steps = train_cfg["eval_every_steps"]

    best_val_loss = float("inf")

    steps_since_improve = 0

    global_step = 0

    training_start = time.perf_counter()

    print("\n🔥 Training Started\n")

        # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(train_cfg["epochs"]):

        model.train()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}"
        )

        for input_ids, labels in pbar:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # -------------------------
            # Forward
            # -------------------------

            with torch.cuda.amp.autocast(
                enabled=train_cfg["mixed_precision"]
            ):

                logits = model(input_ids)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            # -------------------------
            # Backward
            # -------------------------

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            # -------------------------
            # LR Scheduler
            # -------------------------

            lr_now = (
                train_cfg["learning_rate"] *
                lr_schedule(global_step)
            )

            for group in optimizer.param_groups:
                group["lr"] = lr_now

            train_ppl = math.exp(loss.item())

            wandb.log({
                "train/loss": loss.item(),
                "train/perplexity": train_ppl,
                "lr": lr_now,
                "step": global_step
            })

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ppl=f"{train_ppl:.2f}",
                lr=f"{lr_now:.2e}"
            )

            global_step += 1

            # =====================================================
            # Validation
            # =====================================================

            if global_step % eval_every_steps == 0:

                model.eval()

                val_loss = 0.0

                with torch.no_grad():

                    for vi, vl in val_loader:

                        vi = vi.to(device)
                        vl = vl.to(device)

                        logits = model(vi)

                        loss_val = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            vl.view(-1)
                        )

                        val_loss += loss_val.item()

                val_loss /= len(val_loader)

                val_ppl = math.exp(val_loss)

                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "step": global_step
                })

                print(
                    f"\nValidation | "
                    f"Step {global_step:,} | "
                    f"Loss {val_loss:.4f} | "
                    f"PPL {val_ppl:.2f}\n"
                )

                # -------------------------------------------------
                # Save Best Checkpoint
                # -------------------------------------------------

                if val_loss < best_val_loss - early_cfg["min_delta"]:

                    best_val_loss = val_loss

                    steps_since_improve = 0

                    torch.save({

                        "model_state_dict": model.state_dict(),

                        "experiment_name": cfg["experiment_name"],

                        "epoch": epoch + 1,

                        "step": global_step,

                        "best_val_loss": best_val_loss,

                        "val_perplexity": val_ppl,

                        "total_params": total_params,

                        "trainable_params": trainable_params,

                        "config": cfg

                    }, "best_model.pt")

                    print("💾 Saved Best Model")

                else:

                    steps_since_improve += eval_every_steps

                    if (
                        early_cfg["enabled"]
                        and
                        steps_since_improve >= early_cfg["patience_steps"]
                    ):

                        print("\n⏹ Early Stopping Triggered")

                        training_time = (
                            time.perf_counter()
                            - training_start
                        )

                        print("\n========== Training Summary ==========")
                        print(f"Experiment        : {cfg['experiment_name']}")
                        print(f"Best Val Loss     : {best_val_loss:.4f}")
                        print(f"Best Perplexity   : {math.exp(best_val_loss):.2f}")
                        print(f"Training Steps    : {global_step:,}")
                        print(f"Epochs Completed  : {epoch+1}")
                        print(f"Training Time     : {training_time/60:.2f} min")
                        print(f"Parameters        : {total_params/1e6:.2f} M")
                        print("======================================\n")

                        wandb.finish()

                        return

                model.train()

            # =====================================================
            # Max Steps
            # =====================================================

            if global_step >= max_steps:

                training_time = (
                    time.perf_counter()
                    - training_start
                )

                print("\nReached max_steps.")

                print("\n========== Training Summary ==========")
                print(f"Experiment        : {cfg['experiment_name']}")
                print(f"Best Val Loss     : {best_val_loss:.4f}")
                print(f"Best Perplexity   : {math.exp(best_val_loss):.2f}")
                print(f"Training Steps    : {global_step:,}")
                print(f"Epochs Completed  : {epoch+1}")
                print(f"Training Time     : {training_time/60:.2f} min")
                print(f"Parameters        : {total_params/1e6:.2f} M")
                print("======================================\n")

                wandb.finish()

                return

    # =====================================================
    # Finished
    # =====================================================

    training_time = (
        time.perf_counter()
        - training_start
    )

    print("\n========== Training Complete ==========")
    print(f"Experiment        : {cfg['experiment_name']}")
    print(f"Best Val Loss     : {best_val_loss:.4f}")
    print(f"Best Perplexity   : {math.exp(best_val_loss):.2f}")
    print(f"Training Steps    : {global_step:,}")
    print(f"Epochs Completed  : {train_cfg['epochs']}")
    print(f"Training Time     : {training_time/60:.2f} min")
    print(f"Parameters        : {total_params/1e6:.2f} M")
    print("=======================================\n")

    wandb.finish()


if __name__ == "__main__":
    main()