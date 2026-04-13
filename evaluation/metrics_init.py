"""
metrics_init.py
---------------
Computes trajectory metrics for Mamba Init model.
Same metrics as metrics.py but uses trajectories_init/
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
    PERSONAS, PERSONA_BY_ID,
    HIGH_VOLATILITY_IDS, LOW_VOLATILITY_IDS,
)

TRAJ_DIR = "outputs/trajectories_init"


def load_data():
    print("Loading Mamba Init trajectories...")
    trajectories = np.load(
        os.path.join(TRAJ_DIR, "all_trajectories.npy"),
        allow_pickle=True
    ).item()
    summary_df = pd.read_csv(
        os.path.join(TRAJ_DIR, "trajectory_summary.csv")
    )
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories, summary_df


def compute_silhouette(trajectories):
    print("\n" + "="*60)
    print("METRIC: Silhouette Score")
    print("="*60)

    embeddings     = []
    group_labels   = []
    vol_labels     = []

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]
        emb  = traj.mean(axis=0)
        embeddings.append(emb)
        group_labels.append(PERSONA_BY_ID[pid]["group"])
        vol_labels.append(PERSONA_BY_ID[pid]["volatility"])

    embeddings = np.array(embeddings)

    le_group = LabelEncoder()
    le_vol   = LabelEncoder()

    sil_group = silhouette_score(
        embeddings, le_group.fit_transform(group_labels),
        sample_size=5000
    )
    sil_vol = silhouette_score(
        embeddings, le_vol.fit_transform(vol_labels),
        sample_size=5000
    )

    print(f"Silhouette by persona group    : {sil_group:.4f}")
    print(f"Silhouette by volatility level : {sil_vol:.4f}")

    return {"silhouette_group": sil_group, "silhouette_vol": sil_vol}


def compute_classification(trajectories):
    print("\n" + "="*60)
    print("METRIC: Persona Classification from Trajectory")
    print("="*60)

    embeddings  = []
    vol_labels  = []
    group_labels= []

    for (conv_id, pid), data in trajectories.items():
        traj = data["trajectory"]
        emb  = traj.mean(axis=0)
        embeddings.append(emb)
        vol_labels.append(PERSONA_BY_ID[pid]["volatility"])
        group_labels.append(PERSONA_BY_ID[pid]["group"])

    embeddings = np.array(embeddings)
    le         = LabelEncoder()

    # Volatility classification (3 classes)
    vol_enc  = le.fit_transform(vol_labels)
    clf      = LogisticRegression(max_iter=1000, random_state=42)
    scores_v = cross_val_score(clf, embeddings, vol_enc,
                               cv=5, scoring="accuracy")

    # Group classification (5 classes)
    grp_enc  = LabelEncoder().fit_transform(group_labels)
    scores_g = cross_val_score(clf, embeddings, grp_enc,
                               cv=5, scoring="accuracy")

    print(f"\nVolatility (3 classes):")
    print(f"  Accuracy : {scores_v.mean():.4f} ± {scores_v.std():.4f}")
    print(f"  Baseline : 0.3333 (33.3%)")
    print(f"  Improvement: {scores_v.mean()/(1/3):.2f}x over random")

    print(f"\nGroup (5 classes):")
    print(f"  Accuracy : {scores_g.mean():.4f} ± {scores_g.std():.4f}")
    print(f"  Baseline : 0.2000 (20.0%)")
    print(f"  Improvement: {scores_g.mean()/(1/5):.2f}x over random")

    return {
        "volatility_clf_acc": scores_v.mean(),
        "group_clf_acc"     : scores_g.mean(),
    }


def compute_memory_coefficient(trajectories):
    print("\n" + "="*60)
    print("METRIC: Memory Coefficient (α)")
    print("="*60)

    group_alphas = defaultdict(list)

    for (conv_id, pid), data in trajectories.items():
        traj  = data["trajectory"]
        group = PERSONA_BY_ID[pid]["group"]

        if traj.shape[0] < 2:
            continue

        for t in range(1, traj.shape[0]):
            h_curr = traj[t]
            h_prev = traj[t-1]
            n_curr = np.linalg.norm(h_curr)
            n_prev = np.linalg.norm(h_prev)

            if n_curr > 0 and n_prev > 0:
                alpha = np.dot(h_curr, h_prev) / (n_curr * n_prev)
                group_alphas[group].append(alpha)

    overall = []
    for group in ["suppressive","balanced","positive","negative","reactive"]:
        alphas = group_alphas[group]
        overall.extend(alphas)
        print(f"  {group:<12}: α = {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")

    overall_alpha = np.mean(overall)
    print(f"\n  Overall α : {overall_alpha:.4f}")

    return {"overall_alpha": overall_alpha}


def print_final_comparison(sil, clf, mem):
    print("\n" + "="*60)
    print("MAMBA INIT — COMPLETE METRICS")
    print("="*60)
    print(f"""
┌─────────────────────────────────────────────────────┐
│         Mamba Init (h₀ only) — Full Results          │
├──────────────────────────────┬──────────────────────┤
│ Metric                       │ Value                │
├──────────────────────────────┼──────────────────────┤
│ Drift Variance Ratio (H/L)   │ 3.8514              │
│ High Volatility Drift        │ 0.4820              │
│ Low Volatility Drift         │ 0.1252              │
├──────────────────────────────┼──────────────────────┤
│ Silhouette (5 groups)        │ {sil['silhouette_group']:.4f}              │
│ Silhouette (3 volatility)    │ {sil['silhouette_vol']:.4f}              │
├──────────────────────────────┼──────────────────────┤
│ Volatility Classification    │ {clf['volatility_clf_acc']*100:.1f}%             │
│ Group Classification         │ {clf['group_clf_acc']*100:.1f}%             │
├──────────────────────────────┼──────────────────────┤
│ Memory Coefficient (α)       │ {mem['overall_alpha']:.4f}              │
└──────────────────────────────┴──────────────────────┘
""")

    print("="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(f"""
┌──────────────────────┬─────────────┬─────────────┐
│ Metric               │ Mamba Mod   │ Mamba Init  │
│                      │ (h₀+FiLM)   │ (h₀ only)   │
├──────────────────────┼─────────────┼─────────────┤
│ Drift Ratio (H/L)    │ 3.1634      │ 3.8514 ✅   │
│ Volatility Clf       │ 75.2%       │ {clf['volatility_clf_acc']*100:.1f}%       │
│ Group Clf            │ 67.2%       │ {clf['group_clf_acc']*100:.1f}%       │
│ Silhouette (group)   │ 0.0752      │ {sil['silhouette_group']:.4f}       │
│ Memory α             │ 0.9988      │ {mem['overall_alpha']:.4f}       │
│ Parameters           │ 1,787,840   │ 1,524,672 ✅│
│ Best Epoch           │ 1 of 5      │ 3 of 3 ✅   │
└──────────────────────┴─────────────┴─────────────┘
""")


if __name__ == "__main__":
    trajectories, summary_df = load_data()

    sil = compute_silhouette(trajectories)
    clf = compute_classification(trajectories)
    mem = compute_memory_coefficient(trajectories)

    print_final_comparison(sil, clf, mem)

    # Save
    all_metrics = {**sil, **clf, **mem}
    pd.DataFrame([all_metrics]).to_csv(
        os.path.join(TRAJ_DIR, "metrics.csv"), index=False
    )
    print(f"\n✅ Metrics saved → {TRAJ_DIR}/metrics.csv")
