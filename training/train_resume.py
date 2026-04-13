"""
train_resume.py
---------------
Resumes Mamba Mod training from epoch 3 checkpoint.
Trains 2 more epochs (epochs 4 and 5) to give FiLM
layers time to fully converge.

Usage:
    CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    python training/train_resume.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict

sys.path.append(".")
from training.dataset  import ConversationTripletDataset, collate_triplets
from training.model    import PersonaTrajectoryMamba
from training.losses   import TrajectoryLosses
from personas.personas import HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Resume config ──────────────────────────────────────────────────────────────
RESUME_FROM   = "outputs/checkpoints/checkpoint-epoch3.pt"
OUTPUT_DIR    = "outputs/checkpoints"
EXTRA_EPOCHS  = 2      # train epochs 4 and 5
LEARNING_RATE = 5e-5   # lower LR for fine convergence


def load_config(path="configs/trajectory_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, val_loader, device):
    model.eval()
    persona_drifts = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (anchor, pos, neg) in enumerate(val_loader):
            out = model(
                anchor["persona_input_ids"].to(device),
                anchor["persona_attention_mask"].to(device),
                anchor["utt_input_ids"].to(device),
                anchor["utt_attention_mask"].to(device),
                anchor["turn_mask"].to(device),
            )
            traj = out["trajectory"]

            for i in range(traj.size(0)):
                pid     = anchor["persona_id"][i]
                pid_int = pid.item() if isinstance(pid, torch.Tensor) else int(pid)
                n_turns = anchor["num_turns"][i].item()
                t       = traj[i, :n_turns, :]

                if t.size(0) > 1:
                    drift = t[1:] - t[:-1]
                    mag   = torch.norm(drift, dim=-1).mean().item()
                    persona_drifts[pid_int].append(mag)

            if batch_idx >= 99:
                break

    volatile = [d for pid, drifts in persona_drifts.items()
                if pid in HIGH_VOLATILITY_IDS for d in drifts]
    stoic    = [d for pid, drifts in persona_drifts.items()
                if pid in LOW_VOLATILITY_IDS  for d in drifts]

    ratio = np.mean(volatile) / np.mean(stoic) if volatile and stoic else 1.0
    return {
        "drift_volatile"      : np.mean(volatile) if volatile else 0,
        "drift_stoic"         : np.mean(stoic)    if stoic    else 0,
        "drift_variance_ratio": ratio,
    }


def train():
    cfg    = load_config()
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Model       : Mamba Mod (h₀ + FiLM)")
    print(f"Resuming    : {RESUME_FROM}")
    print(f"Extra epochs: {EXTRA_EPOCHS} (epochs 4 and 5)")
    print(f"LR          : {LEARNING_RATE} (lower for convergence)")

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\n📦 Loading datasets...")
    train_dataset = ConversationTripletDataset(
        csv_path          = cfg["data"]["train_path"],
        max_utt_length    = cfg["encoder"]["max_utt_length"],
        max_persona_length= cfg["personas"]["max_persona_length"],
        max_turns         = cfg["data"]["max_turns"],
        seed              = 42,
    )
    val_dataset = ConversationTripletDataset(
        csv_path          = cfg["data"]["val_path"],
        max_utt_length    = cfg["encoder"]["max_utt_length"],
        max_persona_length= cfg["personas"]["max_persona_length"],
        max_turns         = cfg["data"]["max_turns"],
        seed              = 43,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["dataloader"]["num_workers"],
        collate_fn  = collate_triplets,
        pin_memory  = cfg["dataloader"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["dataloader"]["num_workers"],
        collate_fn  = collate_triplets,
        pin_memory  = cfg["dataloader"]["pin_memory"],
    )

    print(f"Steps per epoch: {len(train_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n🧠 Building Mamba Mod model...")
    model = PersonaTrajectoryMamba(
        encoder_name    = cfg["encoder"]["model_name"],
        encoder_hidden  = cfg["encoder"]["hidden_size"],
        d_model         = cfg["mamba"]["d_model"],
        d_state         = cfg["mamba"]["d_state"],
        d_conv          = cfg["mamba"]["d_conv"],
        expand          = cfg["mamba"]["expand"],
        num_layers      = cfg["mamba"]["num_layers"],
        persona_proj_dim= cfg["film"]["persona_proj_dim"],
        d_trajectory    = cfg["trajectory"]["d_trajectory"],
        dropout         = cfg["mamba"]["dropout"],
        freeze_encoder  = cfg["encoder"]["freeze"],
    ).to(device)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\n🔄 Loading checkpoint: {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch      = ckpt["epoch"] + 1
    best_drift_ratio = ckpt.get("drift_ratio", 0.0)
    print(f"   Resumed from epoch : {ckpt['epoch']}")
    print(f"   Starting epoch     : {start_epoch}")
    print(f"   Previous drift ratio: {best_drift_ratio:.4f}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_fn = TrajectoryLosses(
        lambda_contrastive = cfg["losses"]["lambda_contrastive"],
        lambda_smoothness  = cfg["losses"]["lambda_smoothness"],
        lambda_drift       = cfg["losses"]["lambda_drift"],
        lambda_separation  = cfg["losses"]["lambda_separation"],
        triplet_margin     = cfg["losses"]["triplet_margin"],
    )

    # ── Optimizer — lower LR for fine convergence ──────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = LEARNING_RATE,
        weight_decay = cfg["training"]["weight_decay"],
    )
    total_steps  = len(train_loader) * EXTRA_EPOCHS
    warmup_steps = int(total_steps * 0.05)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["fp16"])

    end_epoch = start_epoch + EXTRA_EPOCHS - 1

    print(f"\n🚀 Resuming training epochs {start_epoch} → {end_epoch}...")
    print("=" * 60)

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        total_loss     = 0.0
        total_contrast = 0.0
        total_smooth   = 0.0
        total_drift    = 0.0
        total_sep      = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}")

        for step, (anchor, positive, negative) in enumerate(pbar):

            def to_device(b):
                return {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in b.items()
                }

            anchor   = to_device(anchor)
            positive = to_device(positive)
            negative = to_device(negative)

            with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"]):
                out_a = model(
                    anchor["persona_input_ids"],
                    anchor["persona_attention_mask"],
                    anchor["utt_input_ids"],
                    anchor["utt_attention_mask"],
                    anchor["turn_mask"],
                )
                out_p = model(
                    positive["persona_input_ids"],
                    positive["persona_attention_mask"],
                    positive["utt_input_ids"],
                    positive["utt_attention_mask"],
                    positive["turn_mask"],
                )
                out_n = model(
                    negative["persona_input_ids"],
                    negative["persona_attention_mask"],
                    negative["utt_input_ids"],
                    negative["utt_attention_mask"],
                    negative["turn_mask"],
                )

                loss_dict = loss_fn(
                    out_a, out_p, out_n,
                    anchor["turn_mask"],
                    positive["turn_mask"],
                    negative["turn_mask"],
                    anchor["persona_id"],
                )

            loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_loss     += loss.item()
            total_contrast += loss_dict["L_contrastive"]
            total_smooth   += loss_dict["L_smoothness"]
            total_drift    += loss_dict["L_drift"]
            total_sep      += loss_dict["L_separation"]

            if step % cfg["logging"]["logging_steps"] == 0:
                pbar.set_postfix({
                    "loss"    : f"{total_loss/(step+1):.4f}",
                    "contrast": f"{total_contrast/(step+1):.4f}",
                    "drift"   : f"{total_drift/(step+1):.4f}",
                })

        avg_loss = total_loss / len(train_loader)

        print(f"\n📊 Epoch {epoch} evaluation...")
        metrics     = evaluate(model, val_loader, device)
        drift_ratio = metrics["drift_variance_ratio"]

        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Results — Mamba Mod (h₀ + FiLM):")
        print(f"  Train Loss          : {avg_loss:.4f}")
        print(f"  L_contrastive       : {total_contrast/len(train_loader):.4f}")
        print(f"  L_smoothness        : {total_smooth/len(train_loader):.4f}")
        print(f"  Volatile drift      : {metrics['drift_volatile']:.4f}")
        print(f"  Stoic drift         : {metrics['drift_stoic']:.4f}")
        print(f"  Drift variance ratio: {drift_ratio:.4f} "
              f"{'✅' if drift_ratio > 1.5 else '⚠️'}")
        print(f"\n  --- Comparison ---")
        print(f"  Mamba Init (h₀ only): 3.8450")
        print(f"  Mamba Mod  (current): {drift_ratio:.4f} "
              f"{'✅ FiLM winning!' if drift_ratio > 3.8450 else '⚠️ FiLM still behind'}")
        print(f"{'='*60}\n")

        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch}.pt")
        torch.save({
            "epoch"              : epoch,
            "model_state_dict"   : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict"  : scaler.state_dict(),
            "train_loss"         : avg_loss,
            "drift_ratio"        : drift_ratio,
            "metrics"            : metrics,
        }, ckpt_path)
        print(f"💾 Checkpoint saved → {ckpt_path}")

        if drift_ratio > best_drift_ratio:
            best_drift_ratio = drift_ratio
            best_path = os.path.join(OUTPUT_DIR, "checkpoint-best.pt")
            torch.save({
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "best_drift_ratio": best_drift_ratio,
                "metrics"         : metrics,
            }, best_path)
            print(f"✅ New best model saved (drift_ratio={best_drift_ratio:.4f})")

    print(f"\n🎉 Resume training complete!")
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON:")
    print(f"  Mamba Init (h₀ only, 3 epochs): 3.8450")
    print(f"  Mamba Mod  (h₀+FiLM, 5 epochs): {best_drift_ratio:.4f}")
    if best_drift_ratio > 3.8450:
        print(f"  → FiLM WINS with more training ✅")
        print(f"  → Paper claim: FiLM needs more epochs to converge")
    else:
        print(f"  → h₀ initialization is sufficient ✅")
        print(f"  → Paper claim: simpler model outperforms complex one")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
