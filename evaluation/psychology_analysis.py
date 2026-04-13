"""
psychology_analysis.py
----------------------
Angle 2 — Psychology Additions:

Addition 3: Emotional Contagion
  "Do volatile personas absorb others' emotions more?"
  Measures: how much speaker B's trajectory shifts
  toward speaker A's emotional state

Addition 4: Emotional Recovery Rate
  "Do stoic personas recover faster from disruptions?"
  Measures: turns needed to return to baseline
  after an emotional spike
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

sys.path.append(".")
from personas.personas import (
    PERSONAS, PERSONA_BY_ID,
    HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS,
)

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_data():
    """Load trajectories and speaker info."""
    print("Loading trajectories...")
    trajectories = np.load(
        "outputs/trajectories_init/all_trajectories.npy",
        allow_pickle=True
    ).item()

    print("Loading speaker data...")
    test_df = pd.read_csv("data/splits/test.csv")
    test_df = test_df.sort_values(["conversation_id", "turn_index"])

    # Build speaker map: {conv_id: [(turn_idx, speaker), ...]}
    speaker_map = {}
    for conv_id, group in test_df.groupby("conversation_id"):
        group = group.sort_values("turn_index")
        speaker_map[conv_id] = list(
            zip(group["turn_index"].tolist(),
                group["speaker"].tolist())
        )

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Speaker data for {len(speaker_map)} conversations")
    return trajectories, speaker_map, test_df


# ══════════════════════════════════════════════════════════════
# ADDITION 3 — EMOTIONAL CONTAGION
# ══════════════════════════════════════════════════════════════

def compute_contagion(trajectories, speaker_map):
    """
    Emotional Contagion Analysis.

    For each conversation with persona P:
      At each turn transition (speaker A → speaker B):
        contagion = how much B's new state
                    moved toward A's previous state

    Formula:
      shift = traj_B[t+1] - traj_B[t-1]  (B's change)
      direction = traj_A[t] - traj_B[t-1] (toward A)

      contagion = cos_similarity(shift, direction)
      High → B moved toward A's emotion ✅
      Low  → B maintained own trajectory ✅

    Expected:
      Volatile (low α_p) → HIGH contagion
      Stoic   (high α_p) → LOW  contagion
    """
    print("\n" + "="*60)
    print("ADDITION 3: Emotional Contagion Analysis")
    print("="*60)

    persona_contagion = defaultdict(list)

    for (conv_id, pid), data in trajectories.items():
        traj  = data["trajectory"]  # [turns, 64]
        pinfo = PERSONA_BY_ID[pid]

        if conv_id not in speaker_map:
            continue
        if traj.shape[0] < 3:
            continue

        speakers = speaker_map[conv_id]  # [(turn_idx, speaker)]

        # Find speaker transitions
        # A speaks at turn t, B responds at turn t+1
        for i in range(1, len(speakers)-1):
            t_prev, spk_prev = speakers[i-1]
            t_curr, spk_curr = speakers[i]
            t_next, spk_next = speakers[i+1]

            # Need: A speaks, B responds, B had prior state
            if spk_curr == spk_prev:
                continue  # same speaker, skip
            if spk_next == spk_curr:
                continue  # B speaks again, not A→B transition
            if t_prev >= traj.shape[0]:
                continue
            if t_curr >= traj.shape[0]:
                continue
            if t_next >= traj.shape[0]:
                continue

            # A's state at t_curr (the input to B)
            state_A = traj[t_curr]

            # B's state before (t_prev) and after (t_next)
            state_B_before = traj[t_prev]
            state_B_after  = traj[t_next]

            # How much did B shift?
            shift = state_B_after - state_B_before

            # Direction toward A
            toward_A = state_A - state_B_before

            # Cosine similarity of shift with toward_A
            norm_shift    = np.linalg.norm(shift)
            norm_toward_A = np.linalg.norm(toward_A)

            if norm_shift < 1e-8 or norm_toward_A < 1e-8:
                continue

            contagion = np.dot(shift, toward_A) / (
                norm_shift * norm_toward_A
            )
            persona_contagion[pid].append(contagion)

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'Persona':<25} {'Group':<12} {'Vol':<8} "
          f"{'Contagion':>10} {'Std':>8} {'n':>6}")
    print("-"*70)

    results = []
    for p in PERSONAS:
        pid    = p["id"]
        values = persona_contagion[pid]
        if len(values) < 10:
            continue
        mean_c = np.mean(values)
        std_c  = np.std(values)
        results.append({
            "persona_name"      : p["name"],
            "persona_group"     : p["group"],
            "persona_volatility": p["volatility"],
            "persona_id"        : pid,
            "contagion"         : mean_c,
            "std"               : std_c,
            "n"                 : len(values),
        })
        print(f"{p['name']:<25} {p['group']:<12} "
              f"{p['volatility']:<8} {mean_c:>10.4f} "
              f"{std_c:>8.4f} {len(values):>6}")

    results_df = pd.DataFrame(results)

    # Key comparison
    high_vol = results_df[
        results_df["persona_volatility"] == "high"
    ]["contagion"].mean()
    low_vol  = results_df[
        results_df["persona_volatility"] == "low"
    ]["contagion"].mean()

    print(f"\nHigh volatility contagion: {high_vol:.4f}")
    print(f"Low  volatility contagion: {low_vol:.4f}")
    print(f"Ratio (high/low)          : {high_vol/low_vol:.4f}")

    if high_vol > low_vol:
        print("✅ CONFIRMED: Volatile personas show higher "
              "emotional contagion")
    else:
        print("⚠️  Unexpected: Stoic shows higher contagion")

    # Statistical test
    high_vals = []
    low_vals  = []
    for p in PERSONAS:
        pid = p["id"]
        if pid in HIGH_VOLATILITY_IDS:
            high_vals.extend(persona_contagion[pid])
        elif pid in LOW_VOLATILITY_IDS:
            low_vals.extend(persona_contagion[pid])

    t_stat, p_val = stats.ttest_ind(high_vals, low_vals)
    print(f"\nStatistical test:")
    print(f"  t={t_stat:.4f}, p={p_val:.6f} "
          f"{'✅ significant' if p_val < 0.05 else '⚠️ not significant'}")

    return results_df, persona_contagion


# ══════════════════════════════════════════════════════════════
# ADDITION 4 — EMOTIONAL RECOVERY RATE
# ══════════════════════════════════════════════════════════════

def compute_recovery_rate(trajectories):
    """
    Emotional Recovery Rate Analysis.

    For each trajectory:
      1. Find disruption turns (high drift spikes)
      2. Measure turns until trajectory returns
         close to pre-disruption baseline
      3. Recovery rate = 1 / turns_to_recover

    Expected:
      Stoic    (high α_p) → fast recovery (few turns)
      Volatile (low  α_p) → slow recovery (many turns)

    Psychological basis: Davidson (2000) resilience theory
    """
    print("\n" + "="*60)
    print("ADDITION 4: Emotional Recovery Rate")
    print("="*60)

    persona_recovery = defaultdict(list)

    for (conv_id, pid), data in trajectories.items():
        traj  = data["trajectory"]  # [turns, 64]

        if traj.shape[0] < 5:
            continue

        # Compute step drift per turn
        drifts = []
        for t in range(1, traj.shape[0]):
            d = np.linalg.norm(traj[t] - traj[t-1])
            drifts.append(d)
        drifts = np.array(drifts)

        if len(drifts) < 3:
            continue

        mean_drift = drifts.mean()
        std_drift  = drifts.std()

        # Disruption threshold: 1.5 std above mean
        threshold = mean_drift + 1.5 * std_drift

        # Find disruption turns
        for t_idx in range(len(drifts)):
            t = t_idx + 1  # actual turn index

            if drifts[t_idx] < threshold:
                continue

            # Found disruption at turn t
            # Measure recovery: turns until drift < mean
            pre_disruption = traj[t-1]  # state before spike

            recovery_turns = None
            for t_after in range(t+1, traj.shape[0]):
                dist_to_baseline = np.linalg.norm(
                    traj[t_after] - pre_disruption
                )
                dist_at_disruption = np.linalg.norm(
                    traj[t] - pre_disruption
                )

                # Recovered if back to 50% of disruption dist
                if dist_at_disruption > 1e-8:
                    recovery_ratio = (
                        dist_to_baseline / dist_at_disruption
                    )
                    if recovery_ratio < 0.5:
                        recovery_turns = t_after - t
                        break

            if recovery_turns is None:
                # Never fully recovered in conversation
                recovery_turns = traj.shape[0] - t

            recovery_rate = 1.0 / (recovery_turns + 1e-8)
            persona_recovery[pid].append(recovery_rate)

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'Persona':<25} {'Group':<12} {'Vol':<8} "
          f"{'Recovery Rate':>14} {'Std':>8}")
    print("-"*72)

    results = []
    for p in PERSONAS:
        pid    = p["id"]
        values = persona_recovery[pid]
        if len(values) < 5:
            continue

        mean_r = np.mean(values)
        std_r  = np.std(values)
        results.append({
            "persona_name"      : p["name"],
            "persona_group"     : p["group"],
            "persona_volatility": p["volatility"],
            "persona_id"        : pid,
            "recovery_rate"     : mean_r,
            "std"               : std_r,
            "n"                 : len(values),
        })
        print(f"{p['name']:<25} {p['group']:<12} "
              f"{p['volatility']:<8} {mean_r:>14.4f} "
              f"{std_r:>8.4f}")

    results_df = pd.DataFrame(results)

    # Key comparison
    high_vol = results_df[
        results_df["persona_volatility"] == "high"
    ]["recovery_rate"].mean()
    low_vol  = results_df[
        results_df["persona_volatility"] == "low"
    ]["recovery_rate"].mean()

    print(f"\nHigh volatility recovery rate: {high_vol:.4f}")
    print(f"Low  volatility recovery rate : {low_vol:.4f}")

    if low_vol > high_vol:
        print("✅ CONFIRMED: Stoic personas recover FASTER "
              "(higher rate = fewer turns needed)")
    else:
        print("⚠️  Unexpected direction")

    # Statistical test
    high_vals = []
    low_vals  = []
    for p in PERSONAS:
        pid = p["id"]
        if pid in HIGH_VOLATILITY_IDS:
            high_vals.extend(persona_recovery[pid])
        elif pid in LOW_VOLATILITY_IDS:
            low_vals.extend(persona_recovery[pid])

    t_stat, p_val = stats.ttest_ind(low_vals, high_vals)
    print(f"\nStatistical test:")
    print(f"  t={t_stat:.4f}, p={p_val:.6f} "
          f"{'✅ significant' if p_val < 0.05 else '⚠️ not significant'}")

    return results_df, persona_recovery


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_contagion(contagion_df, persona_contagion):
    """Visualize emotional contagion results."""
    print("\nGenerating contagion plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    GROUP_COLORS = {
        "reactive"   : "#E74C3C",
        "positive"   : "#2ECC71",
        "balanced"   : "#3498DB",
        "negative"   : "#9B59B6",
        "suppressive": "#95A5A6",
    }

    # ── Plot 1: Per-persona contagion bar ──────────────────────
    ax = axes[0]
    sorted_df = contagion_df.sort_values("contagion", ascending=True)
    colors    = [GROUP_COLORS[g] for g in sorted_df["persona_group"]]

    bars = ax.barh(
        sorted_df["persona_name"],
        sorted_df["contagion"],
        color=colors, edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, sorted_df["contagion"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8
        )

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Emotional Contagion Score", fontsize=11)
    ax.set_title("Emotional Contagion per Persona\n"
                 "(positive = absorbs others' emotions)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # ── Plot 2: By volatility group ────────────────────────────
    ax = axes[1]
    vol_groups   = ["low", "medium", "high"]
    vol_means    = []
    vol_stds     = []
    vol_colors   = ["#2ECC71", "#F39C12", "#E74C3C"]

    for vol in vol_groups:
        subset = contagion_df[
            contagion_df["persona_volatility"] == vol
        ]["contagion"]
        vol_means.append(subset.mean())
        vol_stds.append(subset.std())

    bars = ax.bar(
        ["Low\nVolatility", "Medium\nVolatility", "High\nVolatility"],
        vol_means,
        yerr=vol_stds,
        color=vol_colors,
        alpha=0.8,
        edgecolor="white",
        capsize=5
    )
    for bar, val in zip(bars, vol_means):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center",
            fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Avg Emotional Contagion Score", fontsize=11)
    ax.set_title("Contagion by Volatility Group\n"
                 "Volatile personas absorb more emotion",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

    # ── Plot 3: Distribution comparison ───────────────────────
    ax = axes[2]
    high_vals = []
    low_vals  = []
    for p in PERSONAS:
        pid = p["id"]
        if pid in HIGH_VOLATILITY_IDS:
            high_vals.extend(persona_contagion[pid])
        elif pid in LOW_VOLATILITY_IDS:
            low_vals.extend(persona_contagion[pid])

    ax.hist(low_vals,  bins=40, alpha=0.6, color="#2ECC71",
            density=True,
            label=f"Low Volatility (μ={np.mean(low_vals):.3f})")
    ax.hist(high_vals, bins=40, alpha=0.6, color="#E74C3C",
            density=True,
            label=f"High Volatility (μ={np.mean(high_vals):.3f})")

    ax.axvline(np.mean(low_vals),  color="#2ECC71",
               linestyle="--", linewidth=2)
    ax.axvline(np.mean(high_vals), color="#E74C3C",
               linestyle="--", linewidth=2)
    ax.axvline(0, color="black", linewidth=1, linestyle=":")

    ax.set_xlabel("Contagion Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Contagion Distribution\n"
                 "High vs Low Volatility Personas",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Novel Addition 3: Emotional Contagion Analysis\n"
        "'Volatile personas absorb conversation partners' "
        "emotions more readily than Stoic personas'",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotN3_contagion.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_recovery(recovery_df, persona_recovery):
    """Visualize emotional recovery rate results."""
    print("Generating recovery rate plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    GROUP_COLORS = {
        "reactive"   : "#E74C3C",
        "positive"   : "#2ECC71",
        "balanced"   : "#3498DB",
        "negative"   : "#9B59B6",
        "suppressive": "#95A5A6",
    }

    # ── Plot 1: Per-persona recovery rate ─────────────────────
    ax = axes[0]
    sorted_df = recovery_df.sort_values(
        "recovery_rate", ascending=True
    )
    colors = [GROUP_COLORS[g] for g in sorted_df["persona_group"]]

    bars = ax.barh(
        sorted_df["persona_name"],
        sorted_df["recovery_rate"],
        color=colors, edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, sorted_df["recovery_rate"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8
        )

    ax.set_xlabel("Recovery Rate (higher = faster recovery)",
                  fontsize=11)
    ax.set_title("Emotional Recovery Rate per Persona\n"
                 "(higher = more resilient)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # ── Plot 2: By volatility group ────────────────────────────
    ax = axes[1]
    vol_groups = ["low", "medium", "high"]
    vol_means  = []
    vol_stds   = []
    vol_colors = ["#2ECC71", "#F39C12", "#E74C3C"]

    for vol in vol_groups:
        subset = recovery_df[
            recovery_df["persona_volatility"] == vol
        ]["recovery_rate"]
        vol_means.append(subset.mean())
        vol_stds.append(subset.std())

    bars = ax.bar(
        ["Low\nVolatility\n(Stoic)", "Medium", "High\nVolatility\n(Volatile)"],
        vol_means,
        yerr=vol_stds,
        color=vol_colors,
        alpha=0.8,
        edgecolor="white",
        capsize=5
    )
    for bar, val in zip(bars, vol_means):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{val:.3f}", ha="center",
            fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Recovery Rate", fontsize=11)
    ax.set_title("Recovery Rate by Volatility\n"
                 "Stoic personas recover faster",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 3: Recovery vs α_p scatter ───────────────────────
    ax = axes[2]

    # Load α_p values from checkpoint
    import torch
    from training.model_memory import PersonaMemoryMamba
    from transformers import AutoTokenizer

    device    = torch.device("cuda")
    ckpt      = torch.load(
        "outputs/checkpoints_memory/checkpoint-epoch3.pt",
        map_location=device, weights_only=False
    )
    model     = PersonaMemoryMamba().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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

    recovery_df["alpha_p"] = recovery_df["persona_name"].map(
        alpha_vals
    )

    GROUP_C = {
        "reactive"   : "#E74C3C",
        "positive"   : "#2ECC71",
        "balanced"   : "#3498DB",
        "negative"   : "#9B59B6",
        "suppressive": "#95A5A6",
    }
    colors_scatter = [
        GROUP_C[g] for g in recovery_df["persona_group"]
    ]

    ax.scatter(
        recovery_df["alpha_p"],
        recovery_df["recovery_rate"],
        c=colors_scatter, s=100,
        edgecolors="white", linewidth=0.5, zorder=5
    )

    # Labels for key personas
    for _, row in recovery_df.iterrows():
        if row["persona_name"] in [
            "Stoic Regulator", "Rational Analyzer",
            "Emotionally Volatile", "Easily Overwhelmed",
            "Empathy Driven"
        ]:
            ax.annotate(
                row["persona_name"].split()[0],
                (row["alpha_p"], row["recovery_rate"]),
                textcoords="offset points",
                xytext=(5, 5), fontsize=8
            )

    # Trend line
    z = np.polyfit(
        recovery_df["alpha_p"],
        recovery_df["recovery_rate"], 1
    )
    p_line = np.poly1d(z)
    x_line = np.linspace(
        recovery_df["alpha_p"].min(),
        recovery_df["alpha_p"].max(), 100
    )
    ax.plot(x_line, p_line(x_line), "k--",
            alpha=0.5, linewidth=1.5,
            label="Trend")

    # Correlation
    corr, p_val = stats.pearsonr(
        recovery_df["alpha_p"],
        recovery_df["recovery_rate"]
    )

    ax.set_xlabel("Emotional Inertia α_p", fontsize=11)
    ax.set_ylabel("Recovery Rate", fontsize=11)
    ax.set_title(
        f"Recovery Rate vs α_p\nr={corr:.3f}, p={p_val:.4f}",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    print(f"\nCorrelation (recovery ↔ α_p): r={corr:.4f}, "
          f"p={p_val:.6f}")
    if corr > 0.3 and p_val < 0.05:
        print("✅ High α_p → faster recovery confirmed!")

    plt.suptitle(
        "Novel Addition 4: Emotional Recovery Rate\n"
        "'Stoic personas (high α_p) recover faster from "
        "emotional disruptions — validating resilience theory'",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotN4_recovery.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Psychology Additions Analysis ===\n")

    trajectories, speaker_map, test_df = load_data()

    # Addition 3: Emotional Contagion
    contagion_df, persona_contagion = compute_contagion(
        trajectories, speaker_map
    )
    plot_contagion(contagion_df, persona_contagion)

    # Addition 4: Emotional Recovery Rate
    recovery_df, persona_recovery = compute_recovery_rate(
        trajectories
    )
    plot_recovery(recovery_df, persona_recovery)

    print(f"\n{'='*60}")
    print("PSYCHOLOGY ADDITIONS COMPLETE")
    print(f"{'='*60}")
    print("Plots saved:")
    print("  plotN3_contagion.png")
    print("  plotN4_recovery.png")
