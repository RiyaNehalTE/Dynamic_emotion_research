"""
visualize_comparison.py
-----------------------
Generates comparison visualizations between:
  Mamba Init (h₀ only)   — outputs/trajectories_init/
  Mamba Mod  (h₀ + FiLM) — outputs/trajectories/

Plots:
  Plot A: Side-by-side drift bar chart
  Plot B: Drift ratio comparison
  Plot C: Same conversation — both models overlaid
  Plot D: Per-group drift comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(".")
from personas.personas import PERSONAS

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

GROUP_COLORS = {
    "reactive"   : "#E74C3C",
    "positive"   : "#2ECC71",
    "balanced"   : "#3498DB",
    "negative"   : "#9B59B6",
    "suppressive": "#95A5A6",
}


def load_summaries():
    mod  = pd.read_csv("outputs/trajectories/trajectory_summary.csv")
    init = pd.read_csv("outputs/trajectories_init/trajectory_summary.csv")
    return mod, init


def plotA_side_by_side(mod_df, init_df):
    """
    Plot A: Side-by-side drift bar chart for both models.
    """
    print("Generating Plot A: Side-by-side drift comparison...")

    mod_by_p  = mod_df.groupby(
        ["persona_name","persona_group"]
    )["avg_drift"].mean().reset_index()
    init_by_p = init_df.groupby(
        ["persona_name","persona_group"]
    )["avg_drift"].mean().reset_index()

    # Merge
    merged = mod_by_p.merge(
        init_by_p, on=["persona_name","persona_group"],
        suffixes=("_mod","_init")
    )
    merged = merged.sort_values("avg_drift_init", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    y     = np.arange(len(merged))
    width = 0.35

    colors = [GROUP_COLORS[g] for g in merged["persona_group"]]

    bars1 = ax.barh(y - width/2, merged["avg_drift_mod"],
                    width, label="Mamba Mod (h₀+FiLM)",
                    color=colors, alpha=0.6, edgecolor="white")
    bars2 = ax.barh(y + width/2, merged["avg_drift_init"],
                    width, label="Mamba Init (h₀ only)",
                    color=colors, alpha=1.0, edgecolor="white",
                    hatch="//")

    ax.set_yticks(y)
    ax.set_yticklabels(merged["persona_name"], fontsize=9)
    ax.set_xlabel("Average Emotional Drift Magnitude", fontsize=12)
    ax.set_title(
        "Emotional Drift Comparison: Mamba Init vs Mamba Mod\n"
        "per Persona (test set)",
        fontsize=13, fontweight="bold"
    )

    # Legend for models
    legend1 = ax.legend(loc="lower right", fontsize=10)
    ax.add_artist(legend1)

    # Legend for groups
    group_patches = [
        mpatches.Patch(color=c, label=g.capitalize())
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=group_patches, loc="upper left",
              fontsize=9, title="Persona Group")

    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotA_side_by_side.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plotB_drift_ratio(mod_df, init_df):
    """
    Plot B: Drift ratio comparison across volatility groups.
    """
    print("Generating Plot B: Drift ratio comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart of drift ratios
    ax = axes[0]
    models  = ["Mamba Mod\n(h₀+FiLM, 5 epochs)", "Mamba Init\n(h₀ only, 3 epochs)"]
    ratios  = [
        mod_df[mod_df["persona_volatility"]=="high"]["avg_drift"].mean() /
        mod_df[mod_df["persona_volatility"]=="low"]["avg_drift"].mean(),
        init_df[init_df["persona_volatility"]=="high"]["avg_drift"].mean() /
        init_df[init_df["persona_volatility"]=="low"]["avg_drift"].mean(),
    ]
    colors  = ["#3498DB", "#E74C3C"]
    bars    = ax.bar(models, ratios, color=colors,
                     edgecolor="white", width=0.5)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"{val:.4f}", ha="center", fontsize=12,
                fontweight="bold")

    ax.axhline(y=1.5, color="gray", linestyle="--",
               linewidth=1.5, label="Target (1.5)")
    ax.set_ylabel("Drift Variance Ratio (High/Low Volatility)", fontsize=11)
    ax.set_title("Drift Variance Ratio\nby Model", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(ratios) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Group-level drift comparison
    ax = axes[1]
    groups = ["reactive", "positive", "balanced", "negative", "suppressive"]
    x      = np.arange(len(groups))
    width  = 0.35

    mod_vals  = [mod_df[mod_df["persona_group"]==g]["avg_drift"].mean()
                 for g in groups]
    init_vals = [init_df[init_df["persona_group"]==g]["avg_drift"].mean()
                 for g in groups]

    ax.bar(x - width/2, mod_vals,  width, label="Mamba Mod",
           color="#3498DB", alpha=0.8)
    ax.bar(x + width/2, init_vals, width, label="Mamba Init",
           color="#E74C3C", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
    ax.set_ylabel("Average Drift Magnitude", fontsize=11)
    ax.set_title("Per-Group Drift Comparison\nMamba Mod vs Mamba Init",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Model Comparison: Mamba Init (h₀ only) vs Mamba Mod (h₀+FiLM)\n"
        "Mamba Init achieves higher drift ratio with fewer parameters",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotB_drift_ratio.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plotC_training_curves():
    """
    Plot C: Training curves — drift ratio over epochs for both models.
    """
    print("Generating Plot C: Training curves...")

    # Known results from training
    mod_epochs  = [1, 2, 3, 4, 5]
    mod_ratios  = [3.1634, 2.4026, 2.6765, 2.2056, 2.0651]
    mod_losses  = [0.4032, 0.1473, 0.0904, 0.0792, 0.0678]

    init_epochs = [1, 2, 3]
    init_ratios = [1.8074, 3.4885, 3.8450]
    init_losses = [0.4216, 0.1581, 0.1206]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Drift ratio over epochs
    ax = axes[0]
    ax.plot(mod_epochs,  mod_ratios,  "b-o", linewidth=2.5,
            markersize=8, label="Mamba Mod (h₀+FiLM)")
    ax.plot(init_epochs, init_ratios, "r-s", linewidth=2.5,
            markersize=8, label="Mamba Init (h₀ only)")
    ax.axhline(y=1.5, color="gray", linestyle="--",
               linewidth=1.5, label="Target (1.5)", alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Drift Variance Ratio (High/Low)", fontsize=12)
    ax.set_title("Drift Ratio Over Training\n"
                 "Higher = better persona differentiation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))

    # Right: Training loss over epochs
    ax = axes[1]
    ax.plot(mod_epochs,  mod_losses,  "b-o", linewidth=2.5,
            markersize=8, label="Mamba Mod (h₀+FiLM)")
    ax.plot(init_epochs, init_losses, "r-s", linewidth=2.5,
            markersize=8, label="Mamba Init (h₀ only)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Over Epochs\n"
                 "Both models converge cleanly",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))

    plt.suptitle(
        "Training Dynamics: Mamba Init vs Mamba Mod",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotC_training_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plotD_summary_table(mod_df, init_df):
    """
    Plot D: Visual summary table for paper.
    """
    print("Generating Plot D: Summary table...")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    data = [
        ["Metric", "Mamba Mod\n(h₀+FiLM)", "Mamba Init\n(h₀ only)", "Winner"],
        ["Drift Ratio (H/L)",
         "3.1634", "3.8514", "Mamba Init ✅"],
        ["High Volatility Drift",
         "0.4256", "0.4820", "Mamba Init ✅"],
        ["Low Volatility Drift",
         "0.2061", "0.1252", "Mamba Init ✅"],
        ["Trainable Parameters",
         "1,787,840", "1,524,672", "Mamba Init ✅"],
        ["Training Epochs",
         "5", "3", "Mamba Init ✅"],
        ["Persona Classification",
         "67.2%", "—", "—"],
        ["Memory Coefficient α",
         "0.9988", "—", "—"],
    ]

    table = ax.table(
        cellText  = data[1:],
        colLabels = data[0],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style winner column
    for i in range(1, len(data)):
        cell = table[i, 3]
        if "Mamba Init" in data[i][3]:
            cell.set_facecolor("#D5F5E3")
        elif "—" in data[i][3]:
            cell.set_facecolor("#F8F9FA")

    # Alternate row colors
    for i in range(1, len(data)):
        for j in range(3):
            if i % 2 == 0:
                table[i, j].set_facecolor("#EBF5FB")

    ax.set_title(
        "Subjective Emotional Drift — Model Comparison Results",
        fontsize=14, fontweight="bold", pad=20
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plotD_summary_table.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


if __name__ == "__main__":
    print("=== Generating Comparison Visualizations ===\n")

    mod_df, init_df = load_summaries()

    plotA_side_by_side(mod_df, init_df)
    plotB_drift_ratio(mod_df, init_df)
    plotC_training_curves()
    plotD_summary_table(mod_df, init_df)

    print(f"\n✅ All comparison plots saved to: {PLOTS_DIR}/")
    print("\nNew files:")
    print("  plotA_side_by_side.png")
    print("  plotB_drift_ratio.png")
    print("  plotC_training_curves.png")
    print("  plotD_summary_table.png")

def plotD_final_complete():
    """Updated summary table with all metrics for both models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    data = [
        ["Metric",
         "Mamba Mod (h₀+FiLM)\n5 epochs, 1.79M params",
         "Mamba Init (h₀ only)\n3 epochs, 1.52M params",
         "Winner"],
        ["Drift Ratio (H/L)",        "3.1634",  "3.8514",  "Mamba Init ✅"],
        ["High Volatility Drift",    "0.4256",  "0.4820",  "Mamba Init ✅"],
        ["Low Volatility Drift",     "0.2061",  "0.1252",  "Mamba Init ✅"],
        ["Silhouette (5 groups)",    "0.0752",  "0.1851",  "Mamba Init ✅✅"],
        ["Silhouette (volatility)",  "0.0507",  "0.1481",  "Mamba Init ✅✅"],
        ["Volatility Clf (3 class)", "75.2%",   "72.6%",   "Mamba Mod ✅"],
        ["Group Clf (5 class)",      "67.2%",   "68.1%",   "Mamba Init ✅"],
        ["Memory Coefficient α",     "0.9988",  "0.9481",  "Mamba Mod ✅"],
        ["Trainable Parameters",     "1,787,840","1,524,672","Mamba Init ✅"],
        ["Training Epochs",          "5",       "3",       "Mamba Init ✅"],
    ]

    table = ax.table(
        cellText  = data[1:],
        colLabels = data[0],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Header style
    for j in range(4):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colors + winner highlight
    for i in range(1, len(data)):
        for j in range(4):
            if i % 2 == 0:
                table[i, j].set_facecolor("#EBF5FB")
        cell = table[i, 3]
        if "Mamba Init" in data[i][3]:
            cell.set_facecolor("#D5F5E3")
            cell.set_text_props(fontweight="bold")
        elif "Mamba Mod" in data[i][3]:
            cell.set_facecolor("#D6EAF8")
            cell.set_text_props(fontweight="bold")

    ax.set_title(
        "Subjective Emotional Drift — Complete Model Comparison",
        fontsize=14, fontweight="bold", pad=20
    )

    import os
    path = os.path.join("outputs/plots", "plotD_final_complete.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")

plotD_final_complete()
