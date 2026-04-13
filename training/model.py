"""
model.py
--------
Pure Trajectory Mamba Model — No Classification Head.

Architecture:
  persona_text → RoBERTa(frozen) → h₀ (initial emotional state)
  utterances   → RoBERTa(frozen) → [x_1..x_T]
  FiLM         → persona modulates HOW inputs are processed
  Mamba SSM    → hₜ = A·hₜ₋₁ + B·xₜ  (emotional memory)
  Trajectory   → [h_1..h_T] in 64-dim emotion space

No labels. No cross-entropy. No accuracy metric.
Output is a continuous emotional trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from mamba_ssm import Mamba


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Persona vector modulates utterance representations.
    x_modulated = gamma * x + beta
    where gamma, beta are learned from persona vector.
    """
    def __init__(self, d_model: int, persona_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(persona_dim, d_model)
        self.beta_proj  = nn.Linear(persona_dim, d_model)

    def forward(self, x: torch.Tensor, persona_vec: torch.Tensor) -> torch.Tensor:
        """
        x          : [batch, turns, d_model]
        persona_vec: [batch, persona_dim]
        returns    : [batch, turns, d_model]
        """
        gamma = self.gamma_proj(persona_vec).unsqueeze(1)  # [batch, 1, d_model]
        beta  = self.beta_proj(persona_vec).unsqueeze(1)   # [batch, 1, d_model]
        return gamma * x + beta


class PersonaTrajectoryMamba(nn.Module):
    """
    Persona-conditioned Mamba model for emotional trajectory learning.

    Two persona conditioning mechanisms:
      1. h₀ initialization: persona sets the starting emotional state
      2. FiLM modulation  : persona modulates how each utterance is processed

    Together these implement:
      mₜ = α·mₜ₋₁ + (1-α)·eₜ  (your emotional memory formula)
    where α and dynamics are persona-conditioned.
    """

    def __init__(
        self,
        encoder_name    : str   = "roberta-base",
        encoder_hidden  : int   = 768,
        d_model         : int   = 256,
        d_state         : int   = 64,
        d_conv          : int   = 4,
        expand          : int   = 2,
        num_layers      : int   = 2,
        persona_proj_dim: int   = 256,
        d_trajectory    : int   = 64,
        dropout         : float = 0.1,
        freeze_encoder  : bool  = True,
    ):
        super().__init__()

        # ── RoBERTa Encoder (shared for utterances and persona) ────────────────
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("RoBERTa encoder frozen ✅")

        # ── Projections ───────────────────────────────────────────────────────
        self.utt_proj     = nn.Linear(encoder_hidden, d_model)
        self.persona_proj = nn.Linear(encoder_hidden, persona_proj_dim)
        self.h0_proj      = nn.Linear(persona_proj_dim, d_model)

        # ── FiLM layers (one per Mamba layer) ────────────────────────────────
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model, persona_proj_dim)
            for _ in range(num_layers)
        ])

        # ── Mamba SSM layers ──────────────────────────────────────────────────
        # hₜ = A·hₜ₋₁ + B·xₜ  ← your emotional memory equation
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # ── Layer norms ───────────────────────────────────────────────────────
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # ── Trajectory head ───────────────────────────────────────────────────
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_trajectory),
        )

        self.dropout = nn.Dropout(dropout)

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters()
                               if p.requires_grad)
        print(f"Total params    : {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

    def encode_persona(
        self,
        persona_input_ids     : torch.Tensor,
        persona_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode persona description to persona vector.
        persona_input_ids: [batch, seq_len]
        returns          : [batch, persona_proj_dim]
        """
        with torch.no_grad():
            out = self.encoder(
                input_ids      = persona_input_ids,
                attention_mask = persona_attention_mask,
            )
        cls = out.last_hidden_state[:, 0, :]  # [batch, 768]
        return self.persona_proj(cls)          # [batch, persona_proj_dim]

    def encode_utterances(
        self,
        utt_input_ids     : torch.Tensor,
        utt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all utterances in a conversation.
        utt_input_ids: [batch, turns, seq_len]
        returns      : [batch, turns, d_model]
        """
        batch, turns, seq_len = utt_input_ids.shape
        flat_ids  = utt_input_ids.view(batch * turns, seq_len)
        flat_mask = utt_attention_mask.view(batch * turns, seq_len)

        with torch.no_grad():
            out = self.encoder(
                input_ids      = flat_ids,
                attention_mask = flat_mask,
            )

        cls = out.last_hidden_state[:, 0, :]  # [batch*turns, 768]
        cls = cls.view(batch, turns, -1)       # [batch, turns, 768]
        return self.utt_proj(cls)              # [batch, turns, d_model]

    def forward(
        self,
        persona_input_ids     : torch.Tensor,
        persona_attention_mask: torch.Tensor,
        utt_input_ids         : torch.Tensor,
        utt_attention_mask    : torch.Tensor,
        turn_mask             : torch.Tensor = None,
    ) -> dict:
        """
        Forward pass — returns emotional trajectory.

        Returns:
            trajectory : [batch, turns, d_trajectory] ← the emotion video
            hidden     : [batch, turns, d_model]      ← Mamba hidden states
            persona_vec: [batch, persona_proj_dim]    ← persona embedding
        """
        # Step 1: Encode persona → persona vector
        persona_vec = self.encode_persona(
            persona_input_ids, persona_attention_mask
        )

        # Step 2: Encode utterances → sequence of emotion inputs
        x = self.encode_utterances(utt_input_ids, utt_attention_mask)

        # Step 3: Add persona initial state h₀
        # Different personas start from different emotional states
        h0 = self.h0_proj(persona_vec).unsqueeze(1)  # [batch, 1, d_model]
        x  = x + h0                                   # broadcast across turns

        x = self.dropout(x)

        # Step 4: Persona-modulated Mamba layers
        # FiLM conditions HOW emotion evolves per persona
        # Mamba implements hₜ = A·hₜ₋₁ + B·xₜ (your memory formula)
        hidden = x
        for film, mamba, norm in zip(
            self.film_layers, self.mamba_layers, self.layer_norms
        ):
            modulated = film(hidden, persona_vec)
            mamba_out = mamba(modulated)
            hidden    = norm(hidden + mamba_out)

        # Step 5: Project to compact trajectory space
        trajectory = self.trajectory_head(hidden)
        # trajectory: [batch, turns, 64] ← the emotional trajectory

        # Mask padding turns
        if turn_mask is not None:
            mask       = turn_mask.unsqueeze(-1).float()
            trajectory = trajectory * mask
            hidden     = hidden     * mask

        return {
            "trajectory" : trajectory,
            "hidden"     : hidden,
            "persona_vec": persona_vec,
        }


if __name__ == "__main__":
    print("=== Testing Model ===\n")

    # ── IMPORTANT: Mamba requires CUDA ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Mamba requires CUDA. Please run on GPU.")
        exit(1)

    model = PersonaTrajectoryMamba(
        encoder_name ="roberta-base",
        d_model      =256,
        d_state      =64,
        d_conv       =4,
        expand       =2,
        num_layers   =2,
        d_trajectory =64,
    ).to(device)

    model.eval()

    # Dummy batch — all on GPU
    B, T, S = 4, 10, 64
    dummy = {
        "persona_input_ids"      : torch.ones(B, S, dtype=torch.long).to(device),
        "persona_attention_mask" : torch.ones(B, S, dtype=torch.long).to(device),
        "utt_input_ids"          : torch.ones(B, T, S, dtype=torch.long).to(device),
        "utt_attention_mask"     : torch.ones(B, T, S, dtype=torch.long).to(device),
        "turn_mask"              : torch.ones(B, T, dtype=torch.bool).to(device),
    }

    with torch.no_grad():
        out = model(**dummy)

    print(f"\nOutput shapes:")
    print(f"  trajectory : {out['trajectory'].shape}")   # [4, 10, 64]
    print(f"  hidden     : {out['hidden'].shape}")        # [4, 10, 256]
    print(f"  persona_vec: {out['persona_vec'].shape}")   # [4, 256]

    print(f"\nTrajectory value range:")
    print(f"  min: {out['trajectory'].min().item():.4f}")
    print(f"  max: {out['trajectory'].max().item():.4f}")
    print(f"  mean: {out['trajectory'].mean().item():.4f}")

    print("\n✅ Model test passed!")
