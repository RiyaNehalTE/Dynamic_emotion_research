"""
train.py
--------
Training loop for Pure Trajectory Mamba Model.

No classification. No accuracy metric.
Trains using contrastive + smoothness + drift + separation losses.
Saves checkpoint after every epoch.
Evaluates trajectory separation on val set after every epoch.
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
from training.dataset import ConversationTripletDataset, collate_triplets
from training.model   import PersonaTrajectoryMamba
from training.losses  import TrajectoryLosses
from personas.personas import HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_config(path: str = "configs/trajectory_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_trajectory_separation(
    model     : PersonaTrajectoryMamba,
    val_loader: DataLoader,
    device    : torch.device,
    cfg       : dict,
) -> dict:
    """
    Evaluate how well trajectories separate by persona.

    Key metrics:
      1. Drift variance ratio  — Volatile drift / Stoic drift
         (should be > 1.5 to prove your hypothesis)
      2. Trajectory distance   — total path length per persona group
      3. Persona vec spread    — how spread are persona embeddings
    """
    model.eval()

    persona_drift_magnitudes = defaultdict(list)
    persona_distances        = defaultdict(list)
    all_persona_vecs         = []
    all_persona_ids          = []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
            # Only need anchor for evaluation
            out = model(
                persona_input_ids      = anchor["persona_input_ids"].to(device),
                persona_attention_mask = anchor["persona_attention_mask"].to(device),
                utt_input_ids          = anchor["utt_input_ids"].to(device),
                utt_attention_mask     = anchor["utt_attention_mask"].to(device),
                turn_mask              = anchor["turn_mask"].to(device),
            )

            traj        = out["trajectory"]   # [batch, turns, 64]
            persona_vec = out["persona_vec"]  # [batch, 256]
            turn_mask   = anchor["turn_mask"].to(device)

            for i in range(traj.size(0)):
                pid       = anchor["persona_id"][i]
                pid_int   = pid.item() if isinstance(pid, torch.Tensor) else int(pid)
                num_turns = anchor["num_turns"][i].item()

                # Get real turns only
                t = traj[i, :num_turns, :]  # [real_turns, 64]

                # Drift: driftₜ = hₜ - hₜ₋₁ (your formula)
                if t.size(0) > 1:
                    drift     = t[1:] - t[:-1]               # [turns-1, 64]
                    drift_mag = torch.norm(drift, dim=-1)     # [turns-1]
                    persona_drift_magnitudes[pid_int].append(
                        drift_mag.mean().item()
                    )

                    # Total distance traveled in emotion space
                    distance = drift_mag.sum().item()
                    persona_distances[pid_int].append(distance)

            # Collect persona vectors
            all_persona_vecs.append(persona_vec.cpu())
            all_persona_ids.extend(
                [anchor["persona_id"][i] for i in range(traj.size(0))]
            )

            # Only use first 100 batches for speed
            if batch_idx >= 99:
                break

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = {}

    # Drift magnitude per persona group
    volatile_drifts = []
    stoic_drifts    = []
    for pid, drifts in persona_drift_magnitudes.items():
        pid_int = pid.item() if isinstance(pid, torch.Tensor) else int(pid)
        if pid_int in HIGH_VOLATILITY_IDS:
            volatile_drifts.extend(drifts)
        elif pid_int in LOW_VOLATILITY_IDS:
            stoic_drifts.extend(drifts)

    if volatile_drifts and stoic_drifts:
        avg_volatile = np.mean(volatile_drifts)
        avg_stoic    = np.mean(stoic_drifts)
        ratio        = avg_volatile / (avg_stoic + 1e-8)
        metrics["drift_volatile"]      = avg_volatile
        metrics["drift_stoic"]         = avg_stoic
        metrics["drift_variance_ratio"] = ratio
    else:
        metrics["drift_volatile"]       = 0.0
        metrics["drift_stoic"]          = 0.0
        metrics["drift_variance_ratio"] = 1.0

    # Overall drift per persona (for display)
    metrics["per_persona_drift"] = {
        pid: np.mean(drifts)
        for pid, drifts in persona_drift_magnitudes.items()
    }

    # Persona vector spread (higher = more separated)
    if len(all_persona_vecs) > 0:
        vecs   = torch.cat(all_persona_vecs, dim=0)
        normed = torch.nn.functional.normalize(vecs, dim=-1)
        sim    = torch.mm(normed, normed.t())
        # Average off-diagonal similarity (lower = better separation)
        mask   = ~torch.eye(sim.size(0), dtype=torch.bool)
        metrics["avg_persona_similarity"] = sim[mask].mean().item()

    return metrics


def train():
    cfg    = load_config()
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Config loaded ✅")

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

    print(f"Train size : {len(train_dataset)} samples")
    print(f"Val size   : {len(val_dataset)} samples")
    print(f"Train steps: {len(train_loader)} per epoch")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n🧠 Building model...")
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

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_fn = TrajectoryLosses(
        lambda_contrastive = cfg["losses"]["lambda_contrastive"],
        lambda_smoothness  = cfg["losses"]["lambda_smoothness"],
        lambda_drift       = cfg["losses"]["lambda_drift"],
        lambda_separation  = cfg["losses"]["lambda_separation"],
        triplet_margin     = cfg["losses"]["triplet_margin"],
    )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    total_steps  = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["fp16"])

    os.makedirs(cfg["checkpointing"]["output_dir"], exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training ({cfg['training']['epochs']} epochs)...")
    print("=" * 60)

    best_drift_ratio = 0.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        total_loss      = 0.0
        total_contrast  = 0.0
        total_smooth    = 0.0
        total_drift     = 0.0
        total_sep       = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}")

        for step, (anchor, positive, negative) in enumerate(pbar):
            # Move to GPU
            def to_device(batch):
                return {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            anchor   = to_device(anchor)
            positive = to_device(positive)
            negative = to_device(negative)

            with torch.amp.autocast("cuda", enabled=cfg["training"]["fp16"]):
                # Forward pass for all three
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

                # Compute losses
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
                    "loss"     : f"{total_loss/(step+1):.4f}",
                    "contrast" : f"{total_contrast/(step+1):.4f}",
                    "smooth"   : f"{total_smooth/(step+1):.4f}",
                    "drift"    : f"{total_drift/(step+1):.4f}",
                })

        avg_loss = total_loss / len(train_loader)

        # ── Epoch evaluation ───────────────────────────────────────────────────
        print(f"\n📊 Epoch {epoch} evaluation...")
        metrics = evaluate_trajectory_separation(model, val_loader, device, cfg)

        drift_ratio = metrics["drift_variance_ratio"]

        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss          : {avg_loss:.4f}")
        print(f"  L_contrastive       : {total_contrast/len(train_loader):.4f}")
        print(f"  L_smoothness        : {total_smooth/len(train_loader):.4f}")
        print(f"  L_drift             : {total_drift/len(train_loader):.4f}")
        print(f"  L_separation        : {total_sep/len(train_loader):.4f}")
        print(f"  --- Trajectory Metrics ---")
        print(f"  Volatile drift      : {metrics['drift_volatile']:.4f}")
        print(f"  Stoic drift         : {metrics['drift_stoic']:.4f}")
        print(f"  Drift variance ratio: {drift_ratio:.4f} "
              f"{'✅' if drift_ratio > 1.5 else '⚠️ (target > 1.5)'}")
        print(f"  Persona similarity  : {metrics.get('avg_persona_similarity', 0):.4f}")
        print(f"{'='*60}\n")

        # ── Save checkpoint ────────────────────────────────────────────────────
        ckpt_path = os.path.join(
            cfg["checkpointing"]["output_dir"],
            f"checkpoint-epoch{epoch}.pt"
        )
        torch.save({
            "epoch"             : epoch,
            "model_state_dict"  : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict" : scaler.state_dict(),
            "train_loss"        : avg_loss,
            "drift_ratio"       : drift_ratio,
            "metrics"           : metrics,
        }, ckpt_path)
        print(f"💾 Checkpoint saved → {ckpt_path}")

        # Save best model (based on drift ratio)
        if drift_ratio > best_drift_ratio:
            best_drift_ratio = drift_ratio
            best_path = os.path.join(
                cfg["checkpointing"]["output_dir"], "checkpoint-best.pt"
            )
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "best_drift_ratio" : best_drift_ratio,
                "metrics"          : metrics,
            }, best_path)
            print(f"✅ Best model saved (drift_ratio={best_drift_ratio:.4f})")

    print(f"\n🎉 Training complete!")
    print(f"   Best drift variance ratio: {best_drift_ratio:.4f}")
    print(f"   (Target: > 1.5 — proves Volatile drifts more than Stoic)")


if __name__ == "__main__":
    train()
