"""
extract_init.py
---------------
Extracts trajectories from Mamba Init best checkpoint.
Saves to outputs/trajectories_init/
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
from training.dataset    import ConversationTripletDataset, collate_triplets
from training.model_init import PersonaInitMamba
from personas.personas   import PERSONAS, PERSONA_BY_ID

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def extract():
    device = torch.device("cuda")

    # Load model
    print("Loading Mamba Init best checkpoint...")
    ckpt  = torch.load(
        "outputs/checkpoints_init/checkpoint-best.pt",
        map_location=device, weights_only=False
    )
    model = PersonaInitMamba().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, "
          f"drift_ratio={ckpt['best_drift_ratio']:.4f}")

    # Load test dataset
    print("\n📦 Loading test dataset...")
    test_dataset = ConversationTripletDataset(
        csv_path="data/splits/test.csv",
        max_turns=10, seed=44
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_triplets
    )
    print(f"Test conversations: {len(test_dataset.conv_ids)}")

    # Extract
    all_trajectories = {}
    rows = []

    with torch.no_grad():
        for anchor, pos, neg in tqdm(test_loader, desc="Extracting"):
            out = model(
                anchor["persona_input_ids"].to(device),
                anchor["persona_attention_mask"].to(device),
                anchor["utt_input_ids"].to(device),
                anchor["utt_attention_mask"].to(device),
                anchor["turn_mask"].to(device),
            )

            traj        = out["trajectory"].cpu().numpy()
            hidden      = out["hidden"].cpu().numpy()
            persona_vec = out["persona_vec"].cpu().numpy()

            for i in range(len(anchor["conversation_id"])):
                conv_id   = anchor["conversation_id"][i]
                pid       = anchor["persona_id"][i]
                pid_int   = pid.item() if isinstance(pid, torch.Tensor) else int(pid)
                num_turns = anchor["num_turns"][i].item()
                pinfo     = PERSONA_BY_ID[pid_int]

                t   = traj[i, :num_turns, :]
                key = (conv_id, pid_int)

                all_trajectories[key] = {
                    "trajectory"         : t,
                    "hidden"             : hidden[i, :num_turns, :],
                    "persona_vec"        : persona_vec[i],
                    "num_turns"          : num_turns,
                    "persona_name"       : pinfo["name"],
                    "persona_group"      : pinfo["group"],
                    "persona_volatility" : pinfo["volatility"],
                }

                if t.shape[0] > 1:
                    drift      = np.diff(t, axis=0)
                    drift_mag  = np.linalg.norm(drift, axis=1)
                    rows.append({
                        "conversation_id"    : conv_id,
                        "persona_id"         : pid_int,
                        "persona_name"       : pinfo["name"],
                        "persona_group"      : pinfo["group"],
                        "persona_volatility" : pinfo["volatility"],
                        "num_turns"          : num_turns,
                        "avg_drift"          : drift_mag.mean(),
                        "total_distance"     : drift_mag.sum(),
                        "drift_variance"     : drift_mag.var(),
                    })

    # Save
    os.makedirs("outputs/trajectories_init", exist_ok=True)
    np.save("outputs/trajectories_init/all_trajectories.npy",
            all_trajectories, allow_pickle=True)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/trajectories_init/trajectory_summary.csv", index=False)

    # Print summary
    print(f"\nExtracted {len(all_trajectories)} trajectories")
    print("\n=== Per-Persona Drift (Mamba Init) ===")
    by_p = df.groupby(
        ["persona_name", "persona_volatility"]
    )["avg_drift"].mean().sort_values(ascending=False)

    for (name, vol), val in by_p.items():
        print(f"  {name:<25} [{vol:<6}]: {val:.4f}")

    high  = df[df["persona_volatility"]=="high"]["avg_drift"].mean()
    low   = df[df["persona_volatility"]=="low"]["avg_drift"].mean()
    ratio = high / low

    print(f"\n{'='*50}")
    print(f"High volatility : {high:.4f}")
    print(f"Low  volatility : {low:.4f}")
    print(f"Drift ratio     : {ratio:.4f}")
    print(f"{'='*50}")
    print(f"\n✅ Saved to outputs/trajectories_init/")


if __name__ == "__main__":
    extract()
