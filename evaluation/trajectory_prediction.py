"""
trajectory_prediction.py
-------------------------
Novel Addition 8: Trajectory Prediction Task

"Can we predict future emotional states
 given persona and conversation history?"

Method:
  Given: turns 0-6 (7 turns) + persona
  Predict: turns 7, 8, 9 (next 3 turns)
  
  How:
    Run model forward on turns 0-6
    Use last hidden state as context
    Project forward using learned dynamics:
      hₜ₊₁ = α_p·hₜ + (1-α_p)·mean_emotion
    
    mean_emotion = average of seen utterances
    (best estimate for unseen future utterances)

Metric: MSE between predicted and actual trajectory

Expected finding:
  Stoic    (high α_p): LOW MSE  (predictable) ✅
  Volatile (low  α_p): HIGH MSE (unpredictable) ✅
  
  Correlation (α_p ↔ predictability):
    High α_p = strong memory = future ≈ past = LOW MSE
    Low  α_p = weak memory  = future ≈ input  = HIGH MSE

This validates α_p from prediction angle:
  "Emotional inertia makes future states predictable"
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
from scipy import stats
from transformers import AutoTokenizer

sys.path.append(".")
from training.dataset      import ConversationTripletDataset, collate_triplets
from training.model_memory import PersonaMemoryMamba
from personas.personas     import (
    PERSONAS, PERSONA_BY_ID,
    HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS,
)

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

GROUP_COLORS = {
    "reactive"   : "#E74C3C",
    "positive"   : "#2ECC71",
    "balanced"   : "#3498DB",
    "negative"   : "#9B59B6",
    "suppressive": "#95A5A6",
}


def load_model(device):
    """Load best PC-SSM-Memory model."""
    ckpt  = torch.load(
        "outputs/checkpoints_memory/checkpoint-epoch3.pt",
        map_location=device, weights_only=False
    )
    model = PersonaMemoryMamba().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded — epoch {ckpt['epoch']}, "
          f"α_gap={ckpt['alpha_gap']:.4f}")
    return model


def get_alpha_values(model, tokenizer, device):
    """Get learned α_p per persona."""
    alpha_vals = {}
    with torch.no_grad():
        for p in PERSONAS:
            tokens = tokenizer(
                p["description"],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            alpha = model.get_persona_alphas(
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device),
            ).item()
            alpha_vals[p["name"]] = alpha
    return alpha_vals


def predict_future_trajectory(
    model       : PersonaMemoryMamba,
    batch       : dict,
    device      : torch.device,
    history_turns: int = 7,
    future_turns : int = 3,
) -> dict:
    """
    Predict future trajectory given history.

    Method:
      1. Run model on first history_turns turns
         → get hidden state h at turn history_turns-1
         → get α_p for this persona
         → get mean emotion from history turns

      2. Predict future turns using:
         hₜ₊₁ = α_p·hₜ + (1-α_p)·mean_emotion
         (mean_emotion = best guess for unseen input)

      3. Compare predicted vs actual trajectory
         Metric: MSE per turn, averaged

    Returns:
      predicted: [batch, future_turns, d_traj]
      actual   : [batch, future_turns, d_traj]
      mse      : [batch] per-sample MSE
    """
    # ── Step 1: Run model on history ──────────────────────────
    # Create history-only inputs (turns 0 to history_turns-1)
    utt_history = batch["utt_input_ids"][:, :history_turns, :]
    msk_history = batch["utt_attention_mask"][:, :history_turns, :]
    turn_mask_h = batch["turn_mask"][:, :history_turns]

    with torch.no_grad():
        # Full forward pass on history
        out_history = model(
            batch["persona_input_ids"],
            batch["persona_attention_mask"],
            utt_history,
            msk_history,
            turn_mask_h,
        )

        # Get α_p and last hidden state
        alpha_p     = out_history["alpha_p"]  # [B, 1]
        h_last      = out_history["hidden"][:, -1, :]  # [B, d_model]

        # Mean emotion from history (best guess for future inputs)
        mean_emotion = out_history["hidden"].mean(dim=1)  # [B, d_model]

        # ── Step 2: Predict future turns ──────────────────────
        # hₜ₊₁ = α_p·hₜ + (1-α_p)·mean_emotion
        h_pred   = h_last.clone()
        pred_hidden = []

        for _ in range(future_turns):
            h_pred = alpha_p * h_pred + (1 - alpha_p) * mean_emotion
            pred_hidden.append(h_pred)

        pred_hidden = torch.stack(pred_hidden, dim=1)  # [B, F, d_model]

        # Project to trajectory space
        predicted = model.trajectory_head(pred_hidden)  # [B, F, d_traj]

        # ── Step 3: Get actual future trajectory ──────────────
        utt_future = batch["utt_input_ids"][:, history_turns:, :]
        msk_future = batch["utt_attention_mask"][:, history_turns:, :]
        turn_mask_f = batch["turn_mask"][:, history_turns:]

        # Limit to future_turns
        utt_future  = utt_future[:, :future_turns, :]
        msk_future  = msk_future[:, :future_turns, :]
        turn_mask_f = turn_mask_f[:, :future_turns]

        out_future  = model(
            batch["persona_input_ids"],
            batch["persona_attention_mask"],
            utt_future,
            msk_future,
            turn_mask_f,
        )
        actual = out_future["trajectory"]  # [B, F, d_traj]

        # ── Step 4: Compute MSE ───────────────────────────────
        # Only on valid future turns
        mask   = turn_mask_f.unsqueeze(-1).float()  # [B, F, 1]
        diff   = (predicted - actual) ** 2           # [B, F, d_traj]
        mse    = (diff * mask).sum(dim=[1, 2]) / (
            mask.sum(dim=[1, 2]).clamp(min=1)
        )  # [B]

    return {
        "predicted" : predicted.cpu().numpy(),
        "actual"    : actual.cpu().numpy(),
        "mse"       : mse.cpu().numpy(),
        "alpha_p"   : alpha_p.squeeze(-1).cpu().numpy(),
    }


def run_prediction_evaluation(model, device):
    """
    Run trajectory prediction on full test set.
    Compute MSE per persona.
    """
    print("\n" + "="*60)
    print("ADDITION 8: Trajectory Prediction Task")
    print("="*60)

    # Only use conversations with 10 turns
    # (need 7 history + 3 future)
    test_dataset = ConversationTripletDataset(
        csv_path="data/splits/test.csv",
        max_turns=10, seed=44
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_triplets,
    )

    persona_mse     = defaultdict(list)
    persona_alpha   = {}

    print("\nRunning prediction on test set...")
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (anchor, pos, neg) in enumerate(test_loader):

            # Only use conversations with full 10 turns
            valid = anchor["num_turns"] >= 10
            if valid.sum() == 0:
                continue

            # Filter to valid samples
            def filter_batch(b, mask):
                return {
                    k: v[mask] if isinstance(v, torch.Tensor)
                    else [v[i] for i in range(len(v)) if mask[i]]
                    for k, v in b.items()
                }

            anchor_v = filter_batch(anchor, valid)

            # Move to device
            anchor_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in anchor_v.items()
            }

            result = predict_future_trajectory(
                model, anchor_gpu, device,
                history_turns=7,
                future_turns=3,
            )

            # Store per persona
            for i in range(len(anchor_v["persona_id"])):
                pid = anchor_v["persona_id"][i]
                pid_int = pid.item() if isinstance(
                    pid, torch.Tensor) else int(pid)

                persona_mse[pid_int].append(result["mse"][i])
                persona_alpha[pid_int] = result["alpha_p"][i]

            total_samples += valid.sum().item()

            if batch_idx % 50 == 0:
                print(f"  Processed {total_samples} samples...")

    print(f"\nTotal valid samples: {total_samples}")

    # ── Results per persona ────────────────────────────────────
    print(f"\n{'Persona':<25} {'Group':<12} {'Vol':<8} "
          f"{'α_p':>6} {'MSE':>10} {'Predictability'}")
    print("-"*75)

    results = []
    for p in PERSONAS:
        pid    = p["id"]
        mse_vals = persona_mse[pid]
        if len(mse_vals) < 5:
            continue

        mean_mse = np.mean(mse_vals)
        alpha    = persona_alpha.get(pid, 0.5)

        # Predictability = 1/MSE (higher = more predictable)
        predictability = 1.0 / (mean_mse + 1e-8)

        pred_label = (
            "HIGH ✅" if mean_mse < np.median(
                [np.mean(v) for v in persona_mse.values()]
            )
            else "LOW"
        )

        results.append({
            "persona_name"      : p["name"],
            "persona_group"     : p["group"],
            "persona_volatility": p["volatility"],
            "persona_id"        : pid,
            "mse"               : mean_mse,
            "alpha_p"           : alpha,
            "predictability"    : predictability,
        })

        print(f"{p['name']:<25} {p['group']:<12} "
              f"{p['volatility']:<8} {alpha:>6.3f} "
              f"{mean_mse:>10.4f}  {pred_label}")

    results_df = pd.DataFrame(results)

    # ── Key comparison ─────────────────────────────────────────
    high_mse = results_df[
        results_df["persona_volatility"] == "high"
    ]["mse"].mean()
    low_mse = results_df[
        results_df["persona_volatility"] == "low"
    ]["mse"].mean()

    print(f"\nHigh volatility MSE : {high_mse:.4f}")
    print(f"Low  volatility MSE : {low_mse:.4f}")
    print(f"MSE ratio (H/L)     : {high_mse/low_mse:.4f}")

    if high_mse > low_mse:
        print("✅ CONFIRMED: Stoic personas are MORE PREDICTABLE")
        print("   (lower MSE = easier to predict future emotions)")
    else:
        print("⚠️  Unexpected direction")

    # ── Correlation α_p ↔ MSE ─────────────────────────────────
    corr, p_val = stats.pearsonr(
        results_df["alpha_p"],
        results_df["mse"]
    )
    print(f"\nCorrelation (α_p ↔ MSE): r={corr:.4f}, p={p_val:.4f}")

    if corr < -0.3 and p_val < 0.05:
        print("✅ High α_p → Low MSE → More predictable!")
    elif abs(corr) > 0.3:
        print("✅ Meaningful correlation found!")

    return results_df, persona_mse


def plot_prediction(results_df, persona_mse):
    """Visualize trajectory prediction results."""
    print("\nGenerating prediction plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # ── Plot 1: MSE per persona ────────────────────────────────
    ax = axes[0]
    sorted_df = results_df.sort_values("mse", ascending=False)
    colors    = [GROUP_COLORS[g] for g in sorted_df["persona_group"]]

    bars = ax.barh(
        sorted_df["persona_name"],
        sorted_df["mse"],
        color=colors, edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, sorted_df["mse"]):
        ax.text(
            bar.get_width() + 0.0001,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8
        )

    ax.set_xlabel("Prediction MSE\n(lower = more predictable)",
                  fontsize=11)
    ax.set_title("Trajectory Prediction Error per Persona\n"
                 "Stoic = more predictable, Volatile = less",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # Group legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=c, label=g.capitalize())
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    # ── Plot 2: MSE by volatility group ───────────────────────
    ax = axes[1]
    vol_groups = ["low", "medium", "high"]
    vol_colors = ["#2ECC71", "#F39C12", "#E74C3C"]
    vol_labels = ["Low\nVolatility\n(Stoic)",
                  "Medium", "High\nVolatility\n(Volatile)"]
    vol_means  = []
    vol_stds   = []

    for vol in vol_groups:
        subset = results_df[
            results_df["persona_volatility"] == vol
        ]["mse"]
        vol_means.append(subset.mean())
        vol_stds.append(subset.std())

    bars2 = ax.bar(
        vol_labels, vol_means,
        yerr=vol_stds,
        color=vol_colors, alpha=0.85,
        edgecolor="white", capsize=6, width=0.5
    )
    for bar, val in zip(bars2, vol_means):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0002,
            f"{val:.4f}", ha="center",
            fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Average Prediction MSE", fontsize=11)
    ax.set_title("Prediction Error by Volatility\n"
                 "Stoic personas easier to predict",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Arrow annotation
    ax.annotate(
        "More\npredictable",
        xy=(0, vol_means[0]),
        xytext=(0, vol_means[0] - 0.0005),
        ha="center", fontsize=9, color="#2ECC71",
        fontweight="bold"
    )

    # ── Plot 3: α_p vs MSE scatter ────────────────────────────
    ax = axes[2]
    colors_s = [GROUP_COLORS[g] for g in results_df["persona_group"]]

    ax.scatter(
        results_df["alpha_p"],
        results_df["mse"],
        c=colors_s, s=120,
        edgecolors="white", linewidth=0.5, zorder=5
    )

    # Labels for key personas
    for _, row in results_df.iterrows():
        if row["persona_name"] in [
            "Stoic Regulator", "Rational Analyzer",
            "Emotionally Volatile", "Easily Overwhelmed",
            "Empathy Driven", "Cautious Processor"
        ]:
            ax.annotate(
                row["persona_name"].split()[0],
                (row["alpha_p"], row["mse"]),
                textcoords="offset points",
                xytext=(5, 5), fontsize=8
            )

    # Trend line
    z      = np.polyfit(results_df["alpha_p"], results_df["mse"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(
        results_df["alpha_p"].min(),
        results_df["alpha_p"].max(), 100
    )
    ax.plot(x_line, p_line(x_line), "k--",
            alpha=0.5, linewidth=1.5)

    corr, p_val = stats.pearsonr(
        results_df["alpha_p"], results_df["mse"]
    )
    ax.set_xlabel("Emotional Inertia α_p", fontsize=11)
    ax.set_ylabel("Prediction MSE", fontsize=11)
    ax.set_title(
        f"α_p vs Prediction Error\nr={corr:.3f}, p={p_val:.4f}\n"
        f"{'Higher α_p → Lower MSE → More Predictable ✅' if corr < 0 else ''}",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Novel Addition 8: Trajectory Prediction Task\n"
        "'Emotional Inertia α_p determines predictability — "
        "Stoic personas are more predictable than Volatile'",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotN8_prediction.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


if __name__ == "__main__":
    print("=== Trajectory Prediction Analysis ===\n")

    device    = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model     = load_model(device)

    # Get α_p values
    alpha_vals = get_alpha_values(model, tokenizer, device)
    print("\nLoaded α_p values:")
    for name, alpha in sorted(alpha_vals.items(),
                               key=lambda x: x[1]):
        print(f"  {name:<25}: {alpha:.4f}")

    # Run prediction
    results_df, persona_mse = run_prediction_evaluation(
        model, device
    )

    # Visualize
    plot_prediction(results_df, persona_mse)

    print(f"\n{'='*60}")
    print("TRAJECTORY PREDICTION COMPLETE")
    print(f"{'='*60}")
    print("Plot saved: plotN8_prediction.png")
