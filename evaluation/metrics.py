"""
metrics.py
----------
Computes comprehensive trajectory metrics for the paper.

Metrics:
  1. Drift variance ratio     — Volatile/Stoic drift (key result)
  2. Silhouette score         — persona cluster separation
  3. Persona classification   — can we predict persona from trajectory?
  4. Memory coefficient       — how much does past influence present?
  5. Per-group analysis       — suppressive/reactive/positive/negative
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from collections import defaultdict

sys.path.append(".")
from personas.personas import (
    PERSONAS, HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS,
    SUPPRESSIVE_IDS, REACTIVE_IDS, POSITIVE_IDS,
    NEGATIVE_IDS, BALANCED_IDS
)


def load_data(trajectory_dir: str = "outputs/trajectories"):
    """Load extracted trajectories and summary."""
    print("Loading trajectories...")
    trajectories = np.load(
        os.path.join(trajectory_dir, "all_trajectories.npy"),
        allow_pickle=True
    ).item()
    summary_df = pd.read_csv(
        os.path.join(trajectory_dir, "trajectory_summary.csv")
    )
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories, summary_df


def compute_drift_metrics(summary_df: pd.DataFrame) -> dict:
    """
    Metric 1: Drift variance ratio and per-group analysis.
    Your core research metric.
    """
    print("\n" + "="*60)
    print("METRIC 1: Drift Analysis")
    print("="*60)

    # Per volatility group
    high_drift = summary_df[
        summary_df["persona_volatility"] == "high"
    ]["avg_drift"]
    low_drift = summary_df[
        summary_df["persona_volatility"] == "low"
    ]["avg_drift"]
    med_drift = summary_df[
        summary_df["persona_volatility"] == "medium"
    ]["avg_drift"]

    ratio = high_drift.mean() / low_drift.mean()

    print(f"\nBy volatility group:")
    print(f"  High volatility : {high_drift.mean():.4f} ± {high_drift.std():.4f}")
    print(f"  Medium volatility: {med_drift.mean():.4f} ± {med_drift.std():.4f}")
    print(f"  Low volatility  : {low_drift.mean():.4f} ± {low_drift.std():.4f}")
    print(f"  Ratio (H/L)     : {ratio:.4f} ✅")

    # Per persona group
    print(f"\nBy persona group:")
    for group in ["reactive", "positive", "balanced", "negative", "suppressive"]:
        grp = summary_df[summary_df["persona_group"] == group]["avg_drift"]
        print(f"  {group:<12}: {grp.mean():.4f} ± {grp.std():.4f}")

    # Top 5 and bottom 5
    by_persona = summary_df.groupby("persona_name")["avg_drift"].mean()
    print(f"\nTop 5 highest drift:")
    for name, val in by_persona.nlargest(5).items():
        print(f"  {name:<25}: {val:.4f}")
    print(f"\nTop 5 lowest drift:")
    for name, val in by_persona.nsmallest(5).items():
        print(f"  {name:<25}: {val:.4f}")

    return {
        "drift_ratio"       : ratio,
        "high_volatility"   : high_drift.mean(),
        "low_volatility"    : low_drift.mean(),
        "medium_volatility" : med_drift.mean(),
    }


def compute_silhouette(
    trajectories: dict,
    summary_df  : pd.DataFrame,
) -> dict:
    """
    Metric 2: Silhouette score — how well separated are persona clusters?
    Uses mean trajectory embedding per conversation.
    """
    print("\n" + "="*60)
    print("METRIC 2: Silhouette Score (Persona Cluster Separation)")
    print("="*60)

    embeddings = []
    group_labels = []
    volatility_labels = []

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]  # [turns, 64]
        emb  = traj.mean(axis=0)   # [64] — mean trajectory embedding

        from personas.personas import PERSONA_BY_ID
        persona_info = PERSONA_BY_ID[pid]

        embeddings.append(emb)
        group_labels.append(persona_info["group"])
        volatility_labels.append(persona_info["volatility"])

    embeddings = np.array(embeddings)

    # Silhouette by group (5 groups)
    le_group = LabelEncoder()
    group_encoded = le_group.fit_transform(group_labels)
    sil_group = silhouette_score(embeddings, group_encoded, sample_size=5000)

    # Silhouette by volatility (3 levels)
    le_vol = LabelEncoder()
    vol_encoded = le_vol.fit_transform(volatility_labels)
    sil_vol = silhouette_score(embeddings, vol_encoded, sample_size=5000)

    print(f"\nSilhouette score by persona group    : {sil_group:.4f}")
    print(f"Silhouette score by volatility level : {sil_vol:.4f}")
    print(f"\nInterpretation:")
    print(f"  > 0.5 = strong separation")
    print(f"  > 0.3 = moderate separation")
    print(f"  > 0.1 = weak but present separation")

    return {
        "silhouette_group"      : sil_group,
        "silhouette_volatility" : sil_vol,
    }


def compute_persona_classification(
    trajectories: dict,
) -> dict:
    """
    Metric 3: Can we predict persona from trajectory embedding?
    If yes → trajectories encode persona information.
    """
    print("\n" + "="*60)
    print("METRIC 3: Persona Classification from Trajectory")
    print("="*60)

    from personas.personas import PERSONA_BY_ID

    embeddings  = []
    group_labels= []
    vol_labels  = []

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]
        emb  = traj.mean(axis=0)
        embeddings.append(emb)
        group_labels.append(PERSONA_BY_ID[pid]["group"])
        vol_labels.append(PERSONA_BY_ID[pid]["volatility"])

    embeddings = np.array(embeddings)

    # Classify by volatility (3 classes: high/medium/low)
    le = LabelEncoder()
    vol_encoded = le.fit_transform(vol_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, embeddings, vol_encoded,
                             cv=5, scoring="accuracy")

    print(f"\nVolatility classification (3 classes):")
    print(f"  Cross-val accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Random baseline   : {1/3:.4f} (33.3%)")
    print(f"  Improvement       : {scores.mean()/(1/3):.2f}x over random")

    # Classify by group (5 classes)
    le2 = LabelEncoder()
    grp_encoded = le2.fit_transform(group_labels)
    scores2 = cross_val_score(clf, embeddings, grp_encoded,
                              cv=5, scoring="accuracy")

    print(f"\nGroup classification (5 classes):")
    print(f"  Cross-val accuracy: {scores2.mean():.4f} ± {scores2.std():.4f}")
    print(f"  Random baseline   : {1/5:.4f} (20.0%)")
    print(f"  Improvement       : {scores2.mean()/(1/5):.2f}x over random")

    return {
        "volatility_clf_acc" : scores.mean(),
        "group_clf_acc"      : scores2.mean(),
    }


def compute_memory_coefficient(trajectories: dict) -> dict:
    """
    Metric 4: Emotional memory coefficient.
    Validates your formula: mₜ = α·mₜ₋₁ + (1-α)·eₜ
    α = cosine similarity between consecutive hidden states
    Higher α = stronger emotional memory
    """
    print("\n" + "="*60)
    print("METRIC 4: Emotional Memory Coefficient (α)")
    print("="*60)

    from personas.personas import PERSONA_BY_ID

    persona_alphas = defaultdict(list)

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]  # [turns, 64]

        if traj.shape[0] < 2:
            continue

        # Compute cosine similarity between consecutive states
        for t in range(1, traj.shape[0]):
            h_curr = traj[t]
            h_prev = traj[t-1]

            norm_curr = np.linalg.norm(h_curr)
            norm_prev = np.linalg.norm(h_prev)

            if norm_curr > 0 and norm_prev > 0:
                alpha = np.dot(h_curr, h_prev) / (norm_curr * norm_prev)
                persona_alphas[pid].append(alpha)

    print(f"\nMemory coefficient (α) per persona group:")
    group_alphas = defaultdict(list)

    for pid, alphas in persona_alphas.items():
        group = PERSONA_BY_ID[pid]["group"]
        group_alphas[group].extend(alphas)

    overall_alphas = []
    for group in ["suppressive", "balanced", "positive", "negative", "reactive"]:
        alphas = group_alphas[group]
        overall_alphas.extend(alphas)
        print(f"  {group:<12}: α = {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")

    overall_alpha = np.mean(overall_alphas)
    print(f"\n  Overall α        : {overall_alpha:.4f}")
    print(f"  (α > 0.7 = strong emotional memory)")
    print(f"  (α < 0.3 = weak memory, like static model)")

    return {
        "overall_alpha"   : overall_alpha,
        "group_alphas"    : {k: np.mean(v) for k, v in group_alphas.items()},
    }


def print_paper_table(
    drift_metrics : dict,
    sil_metrics   : dict,
    clf_metrics   : dict,
    mem_metrics   : dict,
):
    """Print formatted results table for paper."""
    print("\n" + "="*60)
    print("RESULTS TABLE FOR PAPER")
    print("="*60)

    print("""
┌─────────────────────────────────────────────────────┐
│         Subjective Emotional Drift Results           │
├──────────────────────────────┬──────────────────────┤
│ Metric                       │ Value                │
├──────────────────────────────┼──────────────────────┤
│ Drift Variance Ratio (H/L)   │ {:.4f}              │
│ High Volatility Drift        │ {:.4f}              │
│ Low Volatility Drift         │ {:.4f}              │
├──────────────────────────────┼──────────────────────┤
│ Silhouette (5 groups)        │ {:.4f}              │
│ Silhouette (3 volatility)    │ {:.4f}              │
├──────────────────────────────┼──────────────────────┤
│ Volatility Classification    │ {:.1f}%             │
│ Group Classification         │ {:.1f}%             │
│ (vs random baselines         │ 33.3% / 20.0%)      │
├──────────────────────────────┼──────────────────────┤
│ Memory Coefficient (α)       │ {:.4f}              │
└──────────────────────────────┴──────────────────────┘
""".format(
        drift_metrics["drift_ratio"],
        drift_metrics["high_volatility"],
        drift_metrics["low_volatility"],
        sil_metrics["silhouette_group"],
        sil_metrics["silhouette_volatility"],
        clf_metrics["volatility_clf_acc"] * 100,
        clf_metrics["group_clf_acc"] * 100,
        mem_metrics["overall_alpha"],
    ))


if __name__ == "__main__":
    # Load data
    trajectories, summary_df = load_data()

    # Compute all metrics
    drift_metrics = compute_drift_metrics(summary_df)
    sil_metrics   = compute_silhouette(trajectories, summary_df)
    clf_metrics   = compute_persona_classification(trajectories)
    mem_metrics   = compute_memory_coefficient(trajectories)

    # Print paper table
    print_paper_table(drift_metrics, sil_metrics, clf_metrics, mem_metrics)

    # Save metrics
    all_metrics = {**drift_metrics, **sil_metrics, **clf_metrics, **mem_metrics}
    metrics_df  = pd.DataFrame([all_metrics])
    metrics_df.to_csv("outputs/trajectories/metrics.csv", index=False)
    print("✅ Metrics saved → outputs/trajectories/metrics.csv")
    print("\nNext: python evaluation/visualize.py")
