"""
extract.py
----------
Extracts emotional trajectories from the test set
using the best trained model.

For each (conversation, persona) pair extracts:
  - trajectory : [turns, 64]  — the emotion video
  - hidden     : [turns, 256] — Mamba hidden states
  - persona_vec: [256]        — persona embedding

Saves everything to outputs/trajectories/
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.append(".")
from training.dataset  import ConversationTripletDataset, collate_triplets
from training.model    import PersonaTrajectoryMamba
from personas.personas import PERSONAS, PERSONA_BY_ID

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_best_model(checkpoint_path: str, device: torch.device):
    """Load best model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PersonaTrajectoryMamba(
        encoder_name    = "roberta-base",
        encoder_hidden  = 768,
        d_model         = 256,
        d_state         = 64,
        d_conv          = 4,
        expand          = 2,
        num_layers      = 2,
        persona_proj_dim= 256,
        d_trajectory    = 64,
        dropout         = 0.1,
        freeze_encoder  = True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from epoch {ckpt['epoch']}")
    print(f"   Best drift ratio: {ckpt['best_drift_ratio']:.4f}")
    return model


def extract_trajectories(
    model      : PersonaTrajectoryMamba,
    test_loader: DataLoader,
    device     : torch.device,
    max_batches: int = None,
) -> dict:
    """
    Extract trajectories for all test conversations.

    Returns dict:
    {
      (conv_id, persona_id): {
        "trajectory" : np.array [turns, 64],
        "hidden"     : np.array [turns, 256],
        "persona_vec": np.array [256],
        "num_turns"  : int,
        "persona_group"    : str,
        "persona_volatility": str,
      }
    }
    """
    all_trajectories = {}

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(
            tqdm(test_loader, desc="Extracting trajectories")
        ):
            if max_batches and batch_idx >= max_batches:
                break

            # Only process anchor — we need one per (conv, persona) pair
            out = model(
                persona_input_ids      = anchor["persona_input_ids"].to(device),
                persona_attention_mask = anchor["persona_attention_mask"].to(device),
                utt_input_ids          = anchor["utt_input_ids"].to(device),
                utt_attention_mask     = anchor["utt_attention_mask"].to(device),
                turn_mask              = anchor["turn_mask"].to(device),
            )

            traj        = out["trajectory"].cpu().numpy()  # [B, T, 64]
            hidden      = out["hidden"].cpu().numpy()       # [B, T, 256]
            persona_vec = out["persona_vec"].cpu().numpy()  # [B, 256]

            for i in range(len(anchor["conversation_id"])):
                conv_id   = anchor["conversation_id"][i]
                persona_id = anchor["persona_id"][i]
                pid_int    = persona_id.item() if isinstance(
                    persona_id, torch.Tensor) else int(persona_id)
                num_turns  = anchor["num_turns"][i].item()

                persona_info = PERSONA_BY_ID[pid_int]

                key = (conv_id, pid_int)
                all_trajectories[key] = {
                    "trajectory"         : traj[i, :num_turns, :],
                    "hidden"             : hidden[i, :num_turns, :],
                    "persona_vec"        : persona_vec[i],
                    "num_turns"          : num_turns,
                    "persona_name"       : persona_info["name"],
                    "persona_group"      : persona_info["group"],
                    "persona_volatility" : persona_info["volatility"],
                }

    print(f"\nExtracted {len(all_trajectories)} trajectories")
    return all_trajectories


def save_trajectories(
    trajectories: dict,
    output_dir  : str,
):
    """Save trajectories organized by persona group."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Save full trajectories as numpy ────────────────────────────────────────
    np.save(
        os.path.join(output_dir, "all_trajectories.npy"),
        trajectories,
        allow_pickle=True,
    )
    print(f"✅ Saved all_trajectories.npy")

    # ── Save summary CSV ───────────────────────────────────────────────────────
    rows = []
    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]  # [turns, 64]

        # Compute drift: driftₜ = hₜ - hₜ₋₁
        if traj.shape[0] > 1:
            drift     = np.diff(traj, axis=0)           # [turns-1, 64]
            drift_mag = np.linalg.norm(drift, axis=1)   # [turns-1]
            avg_drift = drift_mag.mean()
            total_dist = drift_mag.sum()
            drift_variance = drift_mag.var()
        else:
            avg_drift = 0.0
            total_dist = 0.0
            drift_variance = 0.0

        rows.append({
            "conversation_id"    : conv_id,
            "persona_id"         : pid,
            "persona_name"       : data["persona_name"],
            "persona_group"      : data["persona_group"],
            "persona_volatility" : data["persona_volatility"],
            "num_turns"          : data["num_turns"],
            "avg_drift"          : avg_drift,
            "total_distance"     : total_dist,
            "drift_variance"     : drift_variance,
        })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "trajectory_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved trajectory_summary.csv ({len(summary_df)} rows)")

    # ── Print quick stats ──────────────────────────────────────────────────────
    print("\n=== Per-Persona Drift Summary ===")
    persona_stats = summary_df.groupby(
        ["persona_name", "persona_group", "persona_volatility"]
    )["avg_drift"].agg(["mean", "std"]).reset_index()
    persona_stats = persona_stats.sort_values("mean", ascending=False)

    print(f"\n{'Persona':<25} {'Group':<12} {'Volatility':<10} "
          f"{'Avg Drift':<12} {'Std':<8}")
    print("-" * 70)
    for _, row in persona_stats.iterrows():
        print(f"{row['persona_name']:<25} {row['persona_group']:<12} "
              f"{row['persona_volatility']:<10} "
              f"{row['mean']:<12.4f} {row['std']:<8.4f}")

    # ── Key ratio ─────────────────────────────────────────────────────────────
    volatile_drift = summary_df[
        summary_df["persona_volatility"] == "high"
    ]["avg_drift"].mean()
    stoic_drift = summary_df[
        summary_df["persona_volatility"] == "low"
    ]["avg_drift"].mean()

    print(f"\n{'='*50}")
    print(f"HIGH volatility avg drift : {volatile_drift:.4f}")
    print(f"LOW  volatility avg drift : {stoic_drift:.4f}")
    print(f"Drift variance ratio      : {volatile_drift/stoic_drift:.4f}")
    print(f"{'='*50}")

    return summary_df


if __name__ == "__main__":
    device = torch.device("cuda")
    print(f"Device: {device}\n")

    # ── Load best model ────────────────────────────────────────────────────────
    model = load_best_model(
        "outputs/checkpoints/checkpoint-best.pt",
        device,
    )

    # ── Load test dataset ──────────────────────────────────────────────────────
    print("\n📦 Loading test dataset...")
    test_dataset = ConversationTripletDataset(
        csv_path          = "data/splits/test.csv",
        max_utt_length    = 64,
        max_persona_length= 64,
        max_turns         = 10,
        seed              = 44,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = 32,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_triplets,
        pin_memory  = True,
    )
    print(f"Test size: {len(test_dataset)} samples")
    print(f"Test conversations: {len(test_dataset.conv_ids)}")

    # ── Extract trajectories ───────────────────────────────────────────────────
    print("\n🔍 Extracting trajectories...")
    trajectories = extract_trajectories(model, test_loader, device)

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n💾 Saving trajectories...")
    output_dir = "outputs/trajectories"
    summary_df = save_trajectories(trajectories, output_dir)

    print(f"\n✅ Extraction complete!")
    print(f"   Files saved to: {output_dir}/")
    print(f"   Next: python evaluation/metrics.py")
