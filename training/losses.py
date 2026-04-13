"""
losses.py
---------
Four loss functions implementing your core research idea:

  "Emotion is not just what is felt, but HOW it evolves
   over time — and that evolution depends on the individual."

Loss 1: Contrastive  — different personas → different trajectories
Loss 2: Smoothness   — emotion changes gradually (memory preserved)
Loss 3: Drift        — volatile personas drift MORE than stoic
Loss 4: Separation   — persona groups are separable in trajectory space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
from personas.personas import (
    HIGH_VOLATILITY_IDS,
    LOW_VOLATILITY_IDS,
    PERSONA_BY_ID,
)


class TrajectoryLosses(nn.Module):
    """
    Combined loss for trajectory learning.
    No classification — purely trajectory-based.
    """

    def __init__(
        self,
        lambda_contrastive : float = 1.0,
        lambda_smoothness  : float = 0.5,
        lambda_drift       : float = 0.3,
        lambda_separation  : float = 0.2,
        triplet_margin     : float = 0.5,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_smoothness  = lambda_smoothness
        self.lambda_drift       = lambda_drift
        self.lambda_separation  = lambda_separation
        self.triplet_margin     = triplet_margin

    def contrastive_loss(
        self,
        traj_anchor  : torch.Tensor,
        traj_positive: torch.Tensor,
        traj_negative: torch.Tensor,
        turn_mask    : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Triplet contrastive loss on trajectory embeddings.

        Pulls together: same persona, different conversation
        Pushes apart  : different persona, same conversation

        traj_anchor  : [batch, turns, d_traj]
        traj_positive: [batch, turns, d_traj]
        traj_negative: [batch, turns, d_traj]
        turn_mask    : [batch, turns]
        """
        # Summarize trajectory → single embedding per conversation
        if turn_mask is not None:
            mask = turn_mask.unsqueeze(-1).float()
            # Mean over valid turns only
            emb_anchor   = (traj_anchor   * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb_positive = (traj_positive * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb_negative = (traj_negative * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            emb_anchor   = traj_anchor.mean(dim=1)    # [batch, d_traj]
            emb_positive = traj_positive.mean(dim=1)
            emb_negative = traj_negative.mean(dim=1)

        # L2 distances
        dist_pos = F.pairwise_distance(emb_anchor, emb_positive, p=2)
        dist_neg = F.pairwise_distance(emb_anchor, emb_negative, p=2)

        # Triplet loss: anchor closer to positive than negative
        loss = F.relu(dist_pos - dist_neg + self.triplet_margin)
        return loss.mean()

    def smoothness_loss(
        self,
        trajectory: torch.Tensor,
        turn_mask : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Temporal smoothness loss.

        Your formula: mₜ = α·mₜ₋₁ + (1-α)·eₜ
        → current state relates to previous state
        → emotion changes gradually not randomly

        Penalizes: large sudden jumps in trajectory
        Encourages: smooth emotional evolution

        trajectory: [batch, turns, d_traj]
        """
        # driftₜ = hₜ - hₜ₋₁  (your drift formula)
        drift = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        # drift: [batch, turns-1, d_traj]

        # Smoothness = variance of drift magnitude
        # Low variance = smooth emotional flow
        # High variance = erratic jumps
        drift_magnitude = torch.norm(drift, dim=-1)  # [batch, turns-1]

        if turn_mask is not None:
            # Only count valid transitions
            valid_mask = (turn_mask[:, 1:] & turn_mask[:, :-1]).float()
            drift_magnitude = drift_magnitude * valid_mask

        # Second derivative — acceleration of emotion change
        # Large acceleration = sudden emotional spike
        if drift.size(1) > 1:
            acceleration = drift[:, 1:, :] - drift[:, :-1, :]
            accel_magnitude = torch.norm(acceleration, dim=-1).mean()
        else:
            accel_magnitude = torch.tensor(0.0, device=trajectory.device)

        # Penalize high acceleration (sudden spikes)
        return torch.clamp(accel_magnitude, max=5.0)

    def drift_ordering_loss(
        self,
        trajectory_volatile: torch.Tensor,
        trajectory_stoic   : torch.Tensor,
        turn_mask_volatile : torch.Tensor = None,
        turn_mask_stoic    : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Drift ordering loss.

        Your key insight:
          driftₜ = eₜ - eₜ₋₁
          Volatile personas should drift MORE than Stoic personas

        This directly validates your research hypothesis:
          "Different personas produce different emotional trajectories"

        trajectory_volatile: [batch, turns, d_traj]
        trajectory_stoic   : [batch, turns, d_traj]
        """
        def compute_drift_magnitude(traj, mask=None):
            # driftₜ = hₜ - hₜ₋₁
            drift = traj[:, 1:, :] - traj[:, :-1, :]
            # drift: [batch, turns-1, d_traj]

            mag = torch.norm(drift, dim=-1)  # [batch, turns-1]

            if mask is not None:
                valid = (mask[:, 1:] & mask[:, :-1]).float()
                mag   = mag * valid
                return mag.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            return mag.mean(dim=1)  # [batch]

        mag_volatile = compute_drift_magnitude(
            trajectory_volatile, turn_mask_volatile
        )
        mag_stoic = compute_drift_magnitude(
            trajectory_stoic, turn_mask_stoic
        )

        # Volatile MUST drift more than Stoic
        # If not — penalize proportionally
        loss = F.relu(mag_stoic.mean() - mag_volatile.mean() + 0.1)
        return loss.mean()

    def persona_separation_loss(
        self,
        persona_vecs: torch.Tensor,
        persona_ids : torch.Tensor,
    ) -> torch.Tensor:
        """
        Persona embedding separation loss.

        Ensures persona vectors are spread in embedding space.
        Different persona groups should be distinguishable.

        persona_vecs: [batch, persona_dim]
        persona_ids : [batch]
        """
        if persona_vecs.size(0) < 2:
            return torch.tensor(0.0, device=persona_vecs.device)

        # Normalize persona vectors
        normed = F.normalize(persona_vecs, dim=-1)

        # Compute pairwise similarities
        sim_matrix = torch.mm(normed, normed.t())  # [batch, batch]

        # Build same/different persona mask
        ids        = persona_ids.unsqueeze(1)       # [batch, 1]
        same_mask  = (ids == ids.t()).float()        # 1 where same persona
        diff_mask  = 1.0 - same_mask

        # Remove diagonal
        eye        = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        same_mask  = same_mask  * (1 - eye)
        diff_mask  = diff_mask  * (1 - eye)

        # Same persona pairs should be similar
        if same_mask.sum() > 0:
            loss_same = (1 - sim_matrix) * same_mask
            loss_same = loss_same.sum() / same_mask.sum()
        else:
            loss_same = torch.tensor(0.0, device=persona_vecs.device)

        # Different persona pairs should be dissimilar
        if diff_mask.sum() > 0:
            loss_diff = F.relu(sim_matrix - 0.3) * diff_mask
            loss_diff = loss_diff.sum() / diff_mask.sum()
        else:
            loss_diff = torch.tensor(0.0, device=persona_vecs.device)

        return loss_same + loss_diff

    def forward(
        self,
        # Triplet outputs
        out_anchor  : dict,
        out_positive: dict,
        out_negative: dict,
        # Batch metadata
        turn_mask_anchor  : torch.Tensor,
        turn_mask_positive: torch.Tensor,
        turn_mask_negative: torch.Tensor,
        persona_ids_anchor: torch.Tensor,
    ) -> dict:
        """
        Compute all losses and return combined loss + breakdown.
        """
        traj_a = out_anchor["trajectory"]
        traj_p = out_positive["trajectory"]
        traj_n = out_negative["trajectory"]

        # ── Loss 1: Contrastive ────────────────────────────────────────────────
        L_contrastive = self.contrastive_loss(
            traj_a, traj_p, traj_n, turn_mask_anchor
        )

        # ── Loss 2: Smoothness (on anchor trajectories) ────────────────────────
        L_smoothness = self.smoothness_loss(traj_a, turn_mask_anchor)

        # ── Loss 3: Drift ordering ─────────────────────────────────────────────
        # Only compute if we have both volatile and stoic in batch
        volatile_mask = torch.zeros(
            traj_a.size(0), dtype=torch.bool, device=traj_a.device
        )
        stoic_mask = torch.zeros(
            traj_a.size(0), dtype=torch.bool, device=traj_a.device
        )

        for i, pid in enumerate(persona_ids_anchor):
            pid_int = pid.item() if isinstance(pid, torch.Tensor) else int(pid)
            if pid_int in HIGH_VOLATILITY_IDS:
                volatile_mask[i] = True
            elif pid_int in LOW_VOLATILITY_IDS:
                stoic_mask[i] = True

        if volatile_mask.sum() > 0 and stoic_mask.sum() > 0:
            L_drift = self.drift_ordering_loss(
                traj_a[volatile_mask], traj_a[stoic_mask],
                turn_mask_anchor[volatile_mask] if turn_mask_anchor is not None
                else None,
                turn_mask_anchor[stoic_mask]    if turn_mask_anchor is not None
                else None,
            )
        else:
            L_drift = torch.tensor(0.0, device=traj_a.device)

        # ── Loss 4: Persona separation ─────────────────────────────────────────
        persona_ids_tensor = torch.tensor(
            persona_ids_anchor if not isinstance(persona_ids_anchor, torch.Tensor)
            else persona_ids_anchor,
            device=traj_a.device
        )
        L_separation = self.persona_separation_loss(
            out_anchor["persona_vec"], persona_ids_tensor
        )

        # ── Combined loss ──────────────────────────────────────────────────────
        total = (
            self.lambda_contrastive * L_contrastive +
            self.lambda_smoothness  * L_smoothness  +
            self.lambda_drift       * L_drift        +
            self.lambda_separation  * L_separation
        )

        return {
            "loss"           : total,
            "L_contrastive"  : L_contrastive.item(),
            "L_smoothness"   : L_smoothness.item(),
            "L_drift"        : L_drift.item(),
            "L_separation"   : L_separation.item(),
        }


if __name__ == "__main__":
    print("=== Testing Losses ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    losses = TrajectoryLosses(
        lambda_contrastive=1.0,
        lambda_smoothness=0.5,
        lambda_drift=0.3,
        lambda_separation=0.2,
        triplet_margin=0.5,
    )

    B, T, D = 8, 10, 64
    persona_ids = torch.tensor([0, 0, 4, 4, 0, 4, 0, 4])

    # Simulate model outputs
    out_a = {
        "trajectory" : torch.randn(B, T, D).to(device),
        "hidden"     : torch.randn(B, T, 256).to(device),
        "persona_vec": torch.randn(B, 256).to(device),
    }
    out_p = {
        "trajectory" : torch.randn(B, T, D).to(device),
        "hidden"     : torch.randn(B, T, 256).to(device),
        "persona_vec": torch.randn(B, 256).to(device),
    }
    out_n = {
        "trajectory" : torch.randn(B, T, D).to(device),
        "hidden"     : torch.randn(B, T, 256).to(device),
        "persona_vec": torch.randn(B, 256).to(device),
    }

    turn_mask = torch.ones(B, T, dtype=torch.bool).to(device)

    result = losses(
        out_a, out_p, out_n,
        turn_mask, turn_mask, turn_mask,
        persona_ids,
    )

    print("Loss breakdown:")
    for k, v in result.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k:20s}: {val:.4f}")

    print("\n✅ Loss test passed!")
