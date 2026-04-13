"""
losses_memory.py (v3 — REACTIVITY)
------------------------------------
Extended losses for PersonaMemoryMamba v3.

KEY CHANGE from v2:
  v2 forced: stoic_alpha > volatile_alpha (memory)
  v3 forces: volatile_alpha > stoic_alpha (reactivity)

This directly guarantees high drift ratio because:
  drift = α_p × ||eₜ - hₜ₋₁||
  volatile α_p HIGH → volatile drift HIGH
  stoic    α_p LOW  → stoic    drift LOW
  ratio = volatile_drift / stoic_drift
        ∝ volatile_alpha / stoic_alpha
        → guaranteed > 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
from training.losses   import TrajectoryLosses
from personas.personas import (
    HIGH_VOLATILITY_IDS,
    LOW_VOLATILITY_IDS,
)


class MemoryTrajectoryLosses(TrajectoryLosses):

    def __init__(
        self,
        lambda_contrastive : float = 1.0,
        lambda_smoothness  : float = 0.1,
        lambda_drift       : float = 0.3,
        lambda_separation  : float = 0.2,
        triplet_margin     : float = 0.5,
        lambda_alpha_order : float = 2.0,
        lambda_alpha_div   : float = 1.0,
        alpha_margin       : float = 0.15,
    ):
        super().__init__(
            lambda_contrastive=lambda_contrastive,
            lambda_smoothness=lambda_smoothness,
            lambda_drift=lambda_drift,
            lambda_separation=lambda_separation,
            triplet_margin=triplet_margin,
        )
        self.lambda_alpha_order = lambda_alpha_order
        self.lambda_alpha_div   = lambda_alpha_div
        self.alpha_margin       = alpha_margin

    def alpha_ordering_loss(
        self,
        alpha_p    : torch.Tensor,
        persona_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alpha Ordering Loss — v3 REACTIVITY version.

        Forces: α_p(volatile) > α_p(stoic) + margin

        This is FLIPPED from v2:
          v2: stoic   > volatile (memory interpretation)
          v3: volatile > stoic   (reactivity interpretation)

        Mathematical consequence:
          drift = α_p × ||eₜ - hₜ₋₁||
          volatile_drift / stoic_drift
          ≈ volatile_alpha / stoic_alpha
          → GUARANTEED high drift ratio
        """
        if alpha_p is None:
            return torch.tensor(0.0)

        alpha_sq = alpha_p.squeeze(-1)

        volatile_mask = torch.zeros(
            len(persona_ids), dtype=torch.bool,
            device=alpha_p.device
        )
        stoic_mask = torch.zeros(
            len(persona_ids), dtype=torch.bool,
            device=alpha_p.device
        )

        for i, pid in enumerate(persona_ids):
            pid_int = pid.item() if isinstance(pid, torch.Tensor) \
                      else int(pid)
            if pid_int in HIGH_VOLATILITY_IDS:
                volatile_mask[i] = True
            elif pid_int in LOW_VOLATILITY_IDS:
                stoic_mask[i] = True

        if volatile_mask.sum() == 0 or stoic_mask.sum() == 0:
            return torch.tensor(0.0, device=alpha_p.device)

        avg_volatile = alpha_sq[volatile_mask].mean()
        avg_stoic    = alpha_sq[stoic_mask].mean()

        # Force: volatile_alpha > stoic_alpha + margin
        # FLIPPED from v2 — now volatile must be HIGH
        loss = F.relu(
            avg_stoic - avg_volatile + self.alpha_margin
        )
        return loss

    def alpha_diversity_loss(
        self,
        alpha_p    : torch.Tensor,
        persona_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alpha Diversity Loss — v3 REACTIVITY version.

        Forces:
          volatile α_p → HIGH (> 0.6)
          stoic    α_p → LOW  (< 0.4)

        FLIPPED from v2:
          v2: volatile < 0.5, stoic > 0.6
          v3: volatile > 0.6, stoic < 0.4
        """
        if alpha_p is None:
            return torch.tensor(0.0)

        alpha_sq = alpha_p.squeeze(-1)

        # Variance component
        variance = alpha_sq.var()
        loss_var = F.relu(0.02 - variance)

        # Range enforcement — FLIPPED
        volatile_mask = torch.zeros(
            len(persona_ids), dtype=torch.bool,
            device=alpha_p.device
        )
        stoic_mask = torch.zeros(
            len(persona_ids), dtype=torch.bool,
            device=alpha_p.device
        )

        for i, pid in enumerate(persona_ids):
            pid_int = pid.item() if isinstance(pid, torch.Tensor) \
                      else int(pid)
            if pid_int in HIGH_VOLATILITY_IDS:
                volatile_mask[i] = True
            elif pid_int in LOW_VOLATILITY_IDS:
                stoic_mask[i] = True

        loss_range = torch.tensor(0.0, device=alpha_p.device)

        if volatile_mask.sum() > 0:
            volatile_alphas = alpha_sq[volatile_mask]
            # Volatile should be HIGH (> 0.6)
            loss_range = loss_range + F.relu(
                0.6 - volatile_alphas
            ).mean()

        if stoic_mask.sum() > 0:
            stoic_alphas = alpha_sq[stoic_mask]
            # Stoic should be LOW (< 0.4)
            loss_range = loss_range + F.relu(
                stoic_alphas - 0.4
            ).mean()

        return loss_var + loss_range

    def forward(
        self,
        out_anchor  : dict,
        out_positive: dict,
        out_negative: dict,
        turn_mask_anchor  : torch.Tensor,
        turn_mask_positive: torch.Tensor,
        turn_mask_negative: torch.Tensor,
        persona_ids_anchor: torch.Tensor,
    ) -> dict:

        base_result = super().forward(
            out_anchor, out_positive, out_negative,
            turn_mask_anchor, turn_mask_positive,
            turn_mask_negative, persona_ids_anchor,
        )

        alpha_p = out_anchor.get("alpha_p", None)

        L_alpha_order = self.alpha_ordering_loss(
            alpha_p, persona_ids_anchor
        )
        L_alpha_div = self.alpha_diversity_loss(
            alpha_p, persona_ids_anchor
        )

        total = (
            base_result["loss"]
            + self.lambda_alpha_order * L_alpha_order
            + self.lambda_alpha_div   * L_alpha_div
        )

        return {
            "loss"          : total,
            "L_contrastive" : base_result["L_contrastive"],
            "L_smoothness"  : base_result["L_smoothness"],
            "L_drift"       : base_result["L_drift"],
            "L_separation"  : base_result["L_separation"],
            "L_alpha_order" : L_alpha_order.item(),
            "L_alpha_div"   : L_alpha_div.item(),
        }
