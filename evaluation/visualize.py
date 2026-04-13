"""
visualize.py
------------
Generates all visualizations for the paper.

Plot 1: Per-persona drift bar chart
Plot 2: Drift curve over turns (your driftₜ = eₜ - eₜ₋₁)
Plot 3: 2D trajectory space (PCA)
Plot 4: Same conversation different personas
Plot 5: Emotional memory heatmap
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from collections import defaultdict

sys.path.append(".")
from personas.personas import PERSONAS, PERSONA_BY_ID

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color scheme per group
GROUP_COLORS = {
    "reactive"   : "#E74C3C",   # red
    "positive"   : "#2ECC71",   # green
    "balanced"   : "#3498DB",   # blue
    "negative"   : "#9B59B6",   # purple
    "suppressive": "#95A5A6",   # gray
}

VOLATILITY_COLORS = {
    "high"  : "#E74C3C",
    "medium": "#F39C12",
    "low"   : "#2ECC71",
}


def load_data():
    trajectories = np.load(
        "outputs/trajectories/all_trajectories.npy",
        allow_pickle=True
    ).item()
    summary_df = pd.read_csv("outputs/trajectories/trajectory_summary.csv")
    return trajectories, summary_df


def plot1_drift_bar(summary_df: pd.DataFrame):
    """
    Plot 1: Per-persona average drift — horizontal bar chart.
    Shows ranking of all 20 personas by drift magnitude.
    """
    print("Generating Plot 1: Per-persona drift bar chart...")

    by_persona = summary_df.groupby(
        ["persona_name", "persona_group", "persona_volatility"]
    )["avg_drift"].mean().reset_index()
    by_persona = by_persona.sort_values("avg_drift", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = [GROUP_COLORS[g] for g in by_persona["persona_group"]]

    bars = ax.barh(
        by_persona["persona_name"],
        by_persona["avg_drift"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # Add value labels
    for bar, val in zip(bars, by_persona["avg_drift"]):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=9
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=g.capitalize())
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)

    ax.set_xlabel("Average Emotional Drift Magnitude", fontsize=12)
    ax.set_title(
        "Subjective Emotional Drift per Persona\n"
        "(same conversation → different trajectories)",
        fontsize=13, fontweight="bold"
    )
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlim(0, by_persona["avg_drift"].max() * 1.15)

    # Add drift ratio annotation
    high = summary_df[summary_df["persona_volatility"]=="high"]["avg_drift"].mean()
    low  = summary_df[summary_df["persona_volatility"]=="low"]["avg_drift"].mean()
    ax.text(
        0.98, 0.02,
        f"Drift Ratio (High/Low): {high/low:.2f}x",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plot1_drift_bar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot2_drift_curve(trajectories: dict):
    """
    Plot 2: Drift curve over turns.
    Shows driftₜ = ||hₜ - hₜ₋₁|| across conversation turns.
    Compares High vs Low volatility personas.
    """
    print("Generating Plot 2: Drift curve over turns...")

    # Collect drift per turn per volatility group
    high_drifts_per_turn = defaultdict(list)
    low_drifts_per_turn  = defaultdict(list)

    for (conv_id, pid), data in trajectories.items():
        traj      = data["trajectory"]
        volatility= data["persona_volatility"]

        if traj.shape[0] < 2:
            continue

        for t in range(1, traj.shape[0]):
            drift_mag = np.linalg.norm(traj[t] - traj[t-1])
            if volatility == "high":
                high_drifts_per_turn[t].append(drift_mag)
            elif volatility == "low":
                low_drifts_per_turn[t].append(drift_mag)

    turns = sorted(set(list(high_drifts_per_turn.keys()) +
                       list(low_drifts_per_turn.keys())))

    high_means = [np.mean(high_drifts_per_turn[t]) for t in turns]
    high_stds  = [np.std(high_drifts_per_turn[t])  for t in turns]
    low_means  = [np.mean(low_drifts_per_turn[t])  for t in turns]
    low_stds   = [np.std(low_drifts_per_turn[t])   for t in turns]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(turns, high_means, color="#E74C3C", linewidth=2.5,
            marker="o", markersize=6, label="High Volatility Personas")
    ax.fill_between(turns,
                    [m-s for m,s in zip(high_means, high_stds)],
                    [m+s for m,s in zip(high_means, high_stds)],
                    color="#E74C3C", alpha=0.2)

    ax.plot(turns, low_means, color="#2ECC71", linewidth=2.5,
            marker="s", markersize=6, label="Low Volatility Personas")
    ax.fill_between(turns,
                    [m-s for m,s in zip(low_means, low_stds)],
                    [m+s for m,s in zip(low_means, low_stds)],
                    color="#2ECC71", alpha=0.2)

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Drift Magnitude (||hₜ - hₜ₋₁||)", fontsize=12)
    ax.set_title(
        "Emotional Drift Curve Over Conversation Turns\n"
        "driftₜ = eₜ − eₜ₋₁",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plot2_drift_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot3_trajectory_space(trajectories: dict):
    """
    Plot 3: 2D PCA projection of trajectory embeddings.
    Shows persona clusters in emotion space.
    """
    print("Generating Plot 3: 2D trajectory space...")

    embeddings = []
    groups     = []
    volatility = []

    for (conv_id, pid), data in trajectories.items():
        emb = data["trajectory"].mean(axis=0)
        embeddings.append(emb)
        groups.append(data["persona_group"])
        volatility.append(data["persona_volatility"])

    embeddings = np.array(embeddings)

    # Sample for speed
    idx = np.random.choice(len(embeddings), min(3000, len(embeddings)),
                           replace=False)
    embeddings = embeddings[idx]
    groups     = [groups[i] for i in idx]
    volatility = [volatility[i] for i in idx]

    # PCA
    pca    = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: by group
    ax = axes[0]
    for group, color in GROUP_COLORS.items():
        mask = [g == group for g in groups]
        pts  = emb_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, label=group.capitalize(),
                   alpha=0.4, s=8)

    ax.set_title("Trajectory Space by Persona Group", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.2)

    # Right: by volatility
    ax = axes[1]
    for vol, color in VOLATILITY_COLORS.items():
        mask = [v == vol for v in volatility]
        pts  = emb_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, label=f"{vol.capitalize()} Volatility",
                   alpha=0.4, s=8)

    ax.set_title("Trajectory Space by Volatility Level", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Emotional Trajectory Space — PCA Projection",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plot3_trajectory_space.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot4_same_conversation(trajectories: dict):
    """
    Plot 4: Same conversation, 4 different personas.
    THE key visualization — shows subjectivity directly.
    """
    print("Generating Plot 4: Same conversation, different personas...")

    # Find a conversation present in all 4 key personas
    target_personas = {
        0: ("Stoic Regulator",      "#95A5A6", "-"),
        4: ("Emotionally Volatile", "#E74C3C", "-"),
        8: ("Optimistic Interpreter","#2ECC71", "--"),
        12:("Negative Anticipator", "#9B59B6", "--"),
    }

    # Find conversations present for all 4
    conv_counts = defaultdict(set)
    for (conv_id, pid) in trajectories.keys():
        if pid in target_personas:
            conv_counts[conv_id].add(pid)

    valid_convs = [c for c, pids in conv_counts.items()
                   if len(pids) >= 4]

    if not valid_convs:
        print("  ⚠️  No conversation found with all 4 personas. Skipping.")
        return

    conv_id = valid_convs[0]
    print(f"  Using conversation: {conv_id}")

    # PCA fitted on all trajectories
    all_embs = np.array([
        data["trajectory"].mean(axis=0)
        for data in trajectories.values()
    ])
    pca = PCA(n_components=1, random_state=42)
    pca.fit(all_embs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trajectory over turns
    ax = axes[0]
    for pid, (name, color, ls) in target_personas.items():
        key = (conv_id, pid)
        if key not in trajectories:
            continue
        traj = trajectories[key]["trajectory"]  # [turns, 64]

        # Project to 1D
        traj_1d = pca.transform(traj).flatten()
        turns   = list(range(len(traj_1d)))

        ax.plot(turns, traj_1d, color=color, linestyle=ls,
                linewidth=2.5, marker="o", markersize=7, label=name)

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Emotional State (PC1)", fontsize=12)
    ax.set_title(
        "Emotional Trajectory — Same Conversation\nDifferent Personas",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: drift magnitude per turn
    ax = axes[1]
    for pid, (name, color, ls) in target_personas.items():
        key = (conv_id, pid)
        if key not in trajectories:
            continue
        traj = trajectories[key]["trajectory"]

        if traj.shape[0] > 1:
            drift = np.linalg.norm(
                traj[1:] - traj[:-1], axis=1
            )
            turns = list(range(1, len(drift)+1))
            ax.plot(turns, drift, color=color, linestyle=ls,
                    linewidth=2.5, marker="o", markersize=7, label=name)

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Drift Magnitude (driftₜ = eₜ − eₜ₋₁)", fontsize=12)
    ax.set_title(
        "Drift Curve — Same Conversation\nDifferent Personas",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Conversation: {conv_id}\n"
        "Same text → Different emotional trajectories per persona",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plot4_same_conversation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot5_memory_heatmap(trajectories: dict):
    """
    Plot 5: Emotional memory heatmap.
    Shows how each persona's trajectory evolves across turns.
    X: turns, Y: personas, Color: drift magnitude
    """
    print("Generating Plot 5: Memory heatmap...")

    # Average drift per turn per persona
    persona_turn_drift = defaultdict(lambda: defaultdict(list))

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]
        name = data["persona_name"]

        if traj.shape[0] > 1:
            for t in range(1, traj.shape[0]):
                drift = np.linalg.norm(traj[t] - traj[t-1])
                persona_turn_drift[name][t].append(drift)

    # Build matrix: personas × turns
    persona_names = [p["name"] for p in sorted(PERSONAS, key=lambda x: x["id"])]
    max_turn = 9
    turns    = list(range(1, max_turn+1))

    matrix = np.zeros((len(persona_names), len(turns)))
    for i, name in enumerate(persona_names):
        for j, t in enumerate(turns):
            if persona_turn_drift[name][t]:
                matrix[i, j] = np.mean(persona_turn_drift[name][t])

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")

    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels([f"Turn {t}" for t in turns], fontsize=10)
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=10)

    plt.colorbar(im, ax=ax, label="Drift Magnitude", shrink=0.8)

    ax.set_title(
        "Emotional Memory Heatmap\n"
        "Drift magnitude per persona per conversation turn",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)

    # Add value annotations
    for i in range(len(persona_names)):
        for j in range(len(turns)):
            ax.text(j, i, f"{matrix[i,j]:.2f}",
                   ha="center", va="center",
                   fontsize=7, color="black")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "plot5_memory_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


if __name__ == "__main__":
    print("=== Generating Visualizations ===\n")

    trajectories, summary_df = load_data()

    plot1_drift_bar(summary_df)
    plot2_drift_curve(trajectories)
    plot3_trajectory_space(trajectories)
    plot4_same_conversation(trajectories)
    plot5_memory_heatmap(trajectories)

    print(f"\n✅ All 5 plots saved to: {PLOTS_DIR}/")
    print("\nFiles:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")
