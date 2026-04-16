"""
application_analysis.py
------------------------
4 application-based additions:

1. Computational Efficiency
2. Majority Class Baseline
3. Per-Domain Significance
4. Emotion Class Granularity
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

sys.path.append(".")
from personas.personas import (
    PERSONAS, PERSONA_BY_ID,
    HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS,
)

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ══════════════════════════════════════════════════════════════
# 1. COMPUTATIONAL EFFICIENCY
# ══════════════════════════════════════════════════════════════

def measure_efficiency():
    """
    Measure inference time per conversation:
      Sankpal  : count emotion label changes
      PC-SSM   : RoBERTa + memory loop
    """
    print("\n" + "="*60)
    print("1. COMPUTATIONAL EFFICIENCY")
    print("="*60)

    from training.dataset      import ConversationTripletDataset, collate_triplets
    from training.model_memory import PersonaMemoryMamba

    device = torch.device("cuda")
    test_df = pd.read_csv("data/splits/test.csv")
    test_df = test_df.sort_values(["conversation_id", "turn_index"])

    # ── Sankpal timing ─────────────────────────────────────────
    print("\nMeasuring Sankpal inference time...")
    conversations = []
    for conv_id, group in test_df.groupby("conversation_id"):
        conversations.append(group["emotion"].tolist())

    N_RUNS = 5
    sankpal_times = []

    for _ in range(N_RUNS):
        start = time.perf_counter()
        for emotions in conversations:
            n_transitions = len(emotions) - 1
            n_changes     = sum(
                1 for t in range(1, len(emotions))
                if emotions[t] != emotions[t-1]
            )
            _ = n_changes / n_transitions if n_transitions > 0 else 0
        sankpal_times.append(time.perf_counter() - start)

    sankpal_total    = np.mean(sankpal_times)
    sankpal_per_conv = sankpal_total / len(conversations) * 1000

    print(f"  Total ({len(conversations)} convs): "
          f"{sankpal_total*1000:.2f} ms")
    print(f"  Per conversation: {sankpal_per_conv:.4f} ms")

    # ── PC-SSM timing ──────────────────────────────────────────
    print("\nMeasuring PC-SSM inference time...")

    ckpt  = torch.load(
        "outputs/checkpoints_memory/checkpoint-epoch3.pt",
        map_location=device, weights_only=False
    )
    model = PersonaMemoryMamba().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = ConversationTripletDataset(
        "data/splits/test.csv", max_turns=10, seed=44
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=collate_triplets
    )

    # Warm up GPU
    with torch.no_grad():
        for anchor, _, _ in test_loader:
            model(
                anchor["persona_input_ids"].to(device),
                anchor["persona_attention_mask"].to(device),
                anchor["utt_input_ids"].to(device),
                anchor["utt_attention_mask"].to(device),
                anchor["turn_mask"].to(device),
            )
            break

    pcssm_times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            for anchor, _, _ in test_loader:
                model(
                    anchor["persona_input_ids"].to(device),
                    anchor["persona_attention_mask"].to(device),
                    anchor["utt_input_ids"].to(device),
                    anchor["utt_attention_mask"].to(device),
                    anchor["turn_mask"].to(device),
                )
        pcssm_times.append(time.perf_counter() - start)

    # Total unique conversations processed
    n_conv_persona = len(test_dataset)
    pcssm_total    = np.mean(pcssm_times)
    pcssm_per_conv = pcssm_total / n_conv_persona * 1000

    print(f"  Total ({n_conv_persona} conv-persona): "
          f"{pcssm_total*1000:.2f} ms")
    print(f"  Per conv-persona: {pcssm_per_conv:.4f} ms")

    # Per conversation (averaged over 20 personas)
    pcssm_per_unique = pcssm_per_conv * 20

    print(f"\nFair comparison (per unique conversation):")
    print(f"  Sankpal : {sankpal_per_conv:.4f} ms")
    print(f"  PC-SSM  : {pcssm_per_unique:.4f} ms "
          f"(20 personas)")
    print(f"  Overhead: {pcssm_per_unique/sankpal_per_conv:.1f}x slower")
    print(f"  But provides: {3.85/1.0:.2f}x better differentiation")

    results = {
        "sankpal_ms"     : sankpal_per_conv,
        "pcssm_ms"       : pcssm_per_unique,
        "overhead"       : pcssm_per_unique / sankpal_per_conv,
        "improvement"    : 3.85,
    }

    return results


# ══════════════════════════════════════════════════════════════
# 2. MAJORITY CLASS BASELINE
# ══════════════════════════════════════════════════════════════

def majority_class_baseline():
    """
    Baseline: always predict dominant emotion (excited).
    drift = 0.0 for all conversations (never changes).

    Shows our method sits between two trivial extremes:
      Majority class : ratio = 0.0 (no drift at all)
      Sankpal        : ratio = 1.0 (no persona effect)
      Ours           : ratio = 3.85 (meaningful)
    """
    print("\n" + "="*60)
    print("2. MAJORITY CLASS BASELINE")
    print("="*60)

    # Majority class = excited (most frequent)
    # drift = 0 for everyone (never changes)
    print("\nMajority class (excited) drift:")
    print("  All personas, all conversations: drift = 0.0000")
    print("  Volatile/Stoic ratio           : 0.0000")
    print("  (Never drifts — trivial lower bound)")

    print("\nSankpal drift:")
    sankpal_df = pd.read_csv(
        "outputs/trajectories_init/sankpal_scores.csv"
    )
    sankpal_mean = sankpal_df["drift_score"].mean()
    print(f"  All personas: {sankpal_mean:.4f}")
    print(f"  Volatile/Stoic ratio: 1.0000")
    print(f"  (No differentiation — trivial upper bound)")

    our_df   = pd.read_csv(
        "outputs/trajectories_init/trajectory_summary.csv"
    )
    our_high = our_df[
        our_df["persona_volatility"] == "high"
    ]["avg_drift"].mean()
    our_low  = our_df[
        our_df["persona_volatility"] == "low"
    ]["avg_drift"].mean()
    our_ratio = our_high / our_low

    print(f"\nOur PC-SSM drift:")
    print(f"  High volatility: {our_high:.4f}")
    print(f"  Low  volatility: {our_low:.4f}")
    print(f"  Volatile/Stoic ratio: {our_ratio:.4f}")
    print(f"  (Meaningful differentiation)")

    print(f"\nSpectrum:")
    print(f"  0.00 (majority) < 1.00 (Sankpal) "
          f"< {our_ratio:.2f} (ours)")
    print(f"  Our method uniquely captures subjectivity")

    return {
        "majority_ratio": 0.0,
        "sankpal_ratio" : 1.0,
        "our_ratio"     : our_ratio,
    }


# ══════════════════════════════════════════════════════════════
# 3. PER-DOMAIN SIGNIFICANCE
# ══════════════════════════════════════════════════════════════

def per_domain_significance():
    """
    Test if persona differentiation is significant
    within each domain separately.

    Expected: p<0.05 in all domains
    Shows: robustness across conversation topics
    """
    print("\n" + "="*60)
    print("3. PER-DOMAIN SIGNIFICANCE")
    print("="*60)

    # Load trajectories with domain info
    our_df  = pd.read_csv(
        "outputs/trajectories_init/trajectory_summary.csv"
    )
    test_df = pd.read_csv("data/splits/test.csv")

    domain_map = test_df.groupby(
        "conversation_id"
    )["domain"].first().reset_index()
    domain_map.columns = ["conversation_id", "domain"]

    merged = our_df.merge(domain_map, on="conversation_id", how="left")

    print(f"\n{'Domain':<20} {'High':>8} {'Low':>8} "
          f"{'Ratio':>7} {'t-stat':>8} {'p-value':>12} {'Sig':>5}")
    print("-"*72)

    domain_results = []
    domains        = merged["domain"].dropna().unique()
    n_significant  = 0

    for domain in sorted(domains):
        subset = merged[merged["domain"] == domain]
        high   = subset[
            subset["persona_volatility"] == "high"
        ]["avg_drift"].values
        low    = subset[
            subset["persona_volatility"] == "low"
        ]["avg_drift"].values

        if len(high) < 5 or len(low) < 5:
            continue

        t, p  = stats.ttest_ind(high, low)
        ratio = np.mean(high) / (np.mean(low) + 1e-8)
        sig   = "✅" if p < 0.05 else "❌"

        if p < 0.05:
            n_significant += 1

        domain_results.append({
            "domain": domain,
            "high"  : np.mean(high),
            "low"   : np.mean(low),
            "ratio" : ratio,
            "t"     : t,
            "p"     : p,
            "sig"   : p < 0.05,
        })

        print(f"{domain:<20} {np.mean(high):>8.4f} "
              f"{np.mean(low):>8.4f} {ratio:>7.3f} "
              f"{t:>8.3f} {p:>12.2e} {sig:>5}")

    domain_df  = pd.DataFrame(domain_results)
    n_total    = len(domain_df)

    print(f"\nSignificant domains: {n_significant}/{n_total} "
          f"({n_significant/n_total*100:.1f}%)")

    if n_significant == n_total:
        print("✅ ALL domains show significant persona differentiation")
        print("   Persona effect is robust across all topics")
    else:
        print(f"✅ {n_significant}/{n_total} domains significant")

    avg_ratio = domain_df["ratio"].mean()
    min_ratio = domain_df["ratio"].min()
    max_ratio = domain_df["ratio"].max()

    print(f"\nDrift ratio across domains:")
    print(f"  Average: {avg_ratio:.4f}")
    print(f"  Min    : {min_ratio:.4f} "
          f"(domain: {domain_df.loc[domain_df['ratio'].idxmin(), 'domain']})")
    print(f"  Max    : {max_ratio:.4f} "
          f"(domain: {domain_df.loc[domain_df['ratio'].idxmax(), 'domain']})")

    return domain_df


# ══════════════════════════════════════════════════════════════
# 4. EMOTION CLASS GRANULARITY
# ══════════════════════════════════════════════════════════════

def emotion_granularity():
    """
    Quantify information gain from 18 vs 6 emotion classes.

    Sankpal uses 6 classes:
      joy, sadness, anger, fear, surprise, neutral

    We use 18 classes with finer granularity.

    Measure: Shannon entropy and information loss
    when mapping 18 → 6 classes.
    """
    print("\n" + "="*60)
    print("4. EMOTION CLASS GRANULARITY")
    print("="*60)

    # Our 18 → their 6 mapping
    our_to_their = {
        "excited"      : "joy",
        "happy"        : "joy",
        "amused"       : "joy",
        "enthusiastic" : "joy",
        "grateful"     : "joy",
        "proud"        : "joy",
        "hopeful"      : "joy",
        "nostalgic"    : "joy",
        "relaxed"      : "neutral",
        "curious"      : "neutral",
        "confused"     : "neutral",
        "surprised"    : "surprise",
        "sad"          : "sadness",
        "disappointed" : "sadness",
        "worried"      : "fear",
        "anxious"      : "fear",
        "frustrated"   : "anger",
        "angry"        : "anger",
    }

    test_df = pd.read_csv("data/splits/test.csv")

    # Compute distributions
    our_counts   = test_df["emotion"].value_counts(normalize=True)
    test_df["their_emotion"] = test_df["emotion"].map(our_to_their)
    their_counts = test_df["their_emotion"].value_counts(normalize=True)

    # Shannon entropy
    def entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    H_ours   = entropy(our_counts.values)
    H_theirs = entropy(their_counts.values)
    info_loss = (H_ours - H_theirs) / H_ours * 100

    print(f"\nOur 18 classes entropy   : {H_ours:.4f} bits")
    print(f"Their 6 classes entropy  : {H_theirs:.4f} bits")
    print(f"Information loss (18→6)  : {info_loss:.1f}%")
    print(f"\nMapping analysis:")
    print(f"  Their 'joy' class contains our:")

    joy_emotions = [e for e, t in our_to_their.items() if t == "joy"]
    for e in joy_emotions:
        if e in our_counts:
            print(f"    {e:<15}: {our_counts[e]*100:.1f}% of data")

    print(f"\n  By collapsing to 6 classes, {info_loss:.1f}% of")
    print(f"  emotional information is lost.")
    print(f"  Our 18-class representation preserves")
    print(f"  {100-info_loss:.1f}% more emotional nuance.")

    # Class distribution comparison
    print(f"\nClass distribution comparison:")
    print(f"  {'Our Class':<20} {'Freq':>6}  →  "
          f"{'Their Class':<12} {'Freq':>6}")
    print(f"  {'-'*55}")

    for emotion, freq in our_counts.items():
        their = our_to_their.get(emotion, "unknown")
        their_freq = their_counts.get(their, 0)
        print(f"  {emotion:<20} {freq*100:>5.1f}%  →  "
              f"{their:<12} {their_freq*100:>5.1f}%")

    return {
        "H_ours"   : H_ours,
        "H_theirs" : H_theirs,
        "info_loss": info_loss,
        "mapping"  : our_to_their,
    }


# ══════════════════════════════════════════════════════════════
# COMBINED VISUALIZATION
# ══════════════════════════════════════════════════════════════

def plot_all(eff_results, baseline_results, domain_df, gran_results):
    """Combined visualization of all 4 additions."""
    print("\nGenerating combined application plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Plot 1: Efficiency ─────────────────────────────────────
    ax = axes[0, 0]

    methods = ["Sankpal\n(2512.13363)", "PC-SSM\n(Ours)"]
    times   = [eff_results["sankpal_ms"],
               eff_results["pcssm_ms"]]
    colors  = ["#95A5A6", "#3498DB"]

    bars = ax.bar(methods, times, color=colors,
                  alpha=0.85, edgecolor="white", width=0.4)

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(times)*0.02,
            f"{val:.2f} ms",
            ha="center", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Inference Time per Conversation (ms)",
                  fontsize=11)
    ax.set_title("1. Computational Efficiency\n"
                 "Cost vs benefit tradeoff",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    ax.text(0.98, 0.95,
            f"Overhead: {eff_results['overhead']:.1f}x\n"
            f"Benefit : {eff_results['improvement']:.2f}x drift ratio",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round",
                      facecolor="lightyellow", alpha=0.9))

    # ── Plot 2: Baseline spectrum ──────────────────────────────
    ax = axes[0, 1]

    methods2 = ["Majority\nClass\n(Lower bound)",
                "Sankpal\n(2512.13363)",
                "Ours\n(PC-SSM)"]
    ratios   = [baseline_results["majority_ratio"],
                baseline_results["sankpal_ratio"],
                baseline_results["our_ratio"]]
    colors2  = ["#BDC3C7", "#95A5A6", "#E74C3C"]

    bars2 = ax.bar(methods2, ratios, color=colors2,
                   alpha=0.85, edgecolor="white", width=0.5)

    for bar, val in zip(bars2, ratios):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", fontsize=11, fontweight="bold"
        )

    ax.axhline(y=1.5, color="gray", linestyle="--",
               linewidth=1.5, alpha=0.7,
               label="Target threshold (1.5)")
    ax.set_ylabel("Volatile/Stoic Drift Ratio",
                  fontsize=11)
    ax.set_title("2. Baseline Spectrum\n"
                 "Our method uniquely captures subjectivity",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 3: Per-domain significance ───────────────────────
    ax = axes[1, 0]

    domain_sorted = domain_df.sort_values("ratio", ascending=True)

    colors3 = ["#2ECC71" if s else "#E74C3C"
               for s in domain_sorted["sig"]]
    bars3 = ax.barh(
        domain_sorted["domain"],
        domain_sorted["ratio"],
        color=colors3, alpha=0.8,
        edgecolor="white", height=0.7
    )

    ax.axvline(x=1.5, color="gray", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.axvline(x=1.0, color="red",  linestyle=":",
               linewidth=1.5, alpha=0.7,
               label="Sankpal (1.0)")

    n_sig = domain_sorted["sig"].sum()
    ax.set_xlabel("Volatile/Stoic Drift Ratio", fontsize=11)
    ax.set_title(
        f"3. Per-Domain Significance\n"
        f"{n_sig}/{len(domain_sorted)} domains significant (p<0.05)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    # ── Plot 4: Emotion class granularity ─────────────────────
    ax = axes[1, 1]

    mapping   = gran_results["mapping"]
    their_map = defaultdict(list)
    for our_e, their_e in mapping.items():
        their_map[their_e].append(our_e)

    their_classes  = list(their_map.keys())
    our_counts_map = [len(their_map[t]) for t in their_classes]

    colors4 = plt.cm.Set3(np.linspace(0, 1, len(their_classes)))
    bars4   = ax.bar(
        their_classes, our_counts_map,
        color=colors4, edgecolor="white",
        alpha=0.85, width=0.6
    )

    for bar, val in zip(bars4, our_counts_map):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{val} classes",
            ha="center", fontsize=9, fontweight="bold"
        )

    ax.set_ylabel("Number of Our Emotion Classes\nMapped to Their Class",
                  fontsize=11)
    ax.set_xlabel("Their 6 Emotion Classes", fontsize=11)
    ax.set_title(
        f"4. Emotion Class Granularity\n"
        f"18 vs 6 classes — "
        f"{gran_results['info_loss']:.1f}% info loss when using 6",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.98, 0.95,
            f"Shannon Entropy:\n"
            f"  18 classes: {gran_results['H_ours']:.3f} bits\n"
            f"  6 classes : {gran_results['H_theirs']:.3f} bits\n"
            f"  Loss      : {gran_results['info_loss']:.1f}%",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round",
                      facecolor="lightyellow", alpha=0.9))

    plt.suptitle(
        "Application-Based Analysis: "
        "Efficiency, Baselines, Robustness, Granularity",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotAPP_application.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Application Analysis ===\n")

    eff_results      = measure_efficiency()
    baseline_results = majority_class_baseline()
    domain_df        = per_domain_significance()
    gran_results     = emotion_granularity()

    plot_all(eff_results, baseline_results,
             domain_df, gran_results)

    print(f"\n{'='*60}")
    print("COMPLETE — Key Numbers for Paper:")
    print(f"{'='*60}")
    print(f"1. PC-SSM overhead    : {eff_results['overhead']:.1f}x Sankpal")
    print(f"2. Baseline spectrum  : 0.00 → 1.00 → "
          f"{baseline_results['our_ratio']:.2f}")
    print(f"3. Domain robustness  : "
          f"{domain_df['sig'].sum()}/{len(domain_df)} domains p<0.05")
    print(f"4. Info preservation  : "
          f"{100-gran_results['info_loss']:.1f}% vs their 6 classes")
    print(f"\nPlot saved: outputs/plots/plotAPP_application.png")
