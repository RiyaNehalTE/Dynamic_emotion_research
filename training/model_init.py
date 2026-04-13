"""
model_init.py
-------------
Mamba Init Model — h₀ initialization ONLY, no FiLM.

Difference from model.py:
  model.py    : h₀ + FiLM (both persona conditioning mechanisms)
  model_init.py: h₀ ONLY  (persona only sets starting state)

This is the ablation model for comparison:
  "Does FiLM modulation add value beyond h₀ initialization?"

Architecture:
  persona_text → RoBERTa(frozen) → h₀ (initial state only)
  utterances   → RoBERTa(frozen) → [x_1..x_T]
  Mamba SSM    → hₜ = A·hₜ₋₁ + B·xₜ (same dynamics for all personas)
  Trajectory   → [h_1..h_T] in 64-dim emotion space

NO FiLM — persona ONLY affects starting point, not dynamics.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from mamba_ssm import Mamba


class PersonaInitMamba(nn.Module):
    """
    Mamba Init — persona conditions only the initial hidden state h₀.
    Same transition dynamics for all personas.
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

        # ── RoBERTa Encoder (frozen) ───────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("RoBERTa encoder frozen ✅")

        # ── Projections ───────────────────────────────────────────────────────
        self.utt_proj     = nn.Linear(encoder_hidden, d_model)
        self.persona_proj = nn.Linear(encoder_hidden, persona_proj_dim)

        # Persona → h₀ ONLY (no FiLM)
        self.h0_proj = nn.Linear(persona_proj_dim, d_model)

        # ── Mamba SSM layers (SAME for all personas — no FiLM) ────────────────
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
        print(f"Model type      : Mamba Init (h₀ only, no FiLM)")

    def encode_persona(self, persona_input_ids, persona_attention_mask):
        with torch.no_grad():
            out = self.encoder(
                input_ids      = persona_input_ids,
                attention_mask = persona_attention_mask,
            )
        cls = out.last_hidden_state[:, 0, :]
        return self.persona_proj(cls)

    def encode_utterances(self, utt_input_ids, utt_attention_mask):
        batch, turns, seq_len = utt_input_ids.shape
        flat_ids  = utt_input_ids.view(batch * turns, seq_len)
        flat_mask = utt_attention_mask.view(batch * turns, seq_len)
        with torch.no_grad():
            out = self.encoder(
                input_ids      = flat_ids,
                attention_mask = flat_mask,
            )
        cls = out.last_hidden_state[:, 0, :]
        cls = cls.view(batch, turns, -1)
        return self.utt_proj(cls)

    def forward(
        self,
        persona_input_ids     : torch.Tensor,
        persona_attention_mask: torch.Tensor,
        utt_input_ids         : torch.Tensor,
        utt_attention_mask    : torch.Tensor,
        turn_mask             : torch.Tensor = None,
    ) -> dict:

        # Step 1: Encode persona → persona vector
        persona_vec = self.encode_persona(
            persona_input_ids, persona_attention_mask
        )

        # Step 2: Encode utterances
        x = self.encode_utterances(utt_input_ids, utt_attention_mask)

        # Step 3: Add h₀ from persona (ONLY conditioning mechanism)
        h0 = self.h0_proj(persona_vec).unsqueeze(1)  # [batch, 1, d_model]
        x  = x + h0                                   # shift starting point

        x = self.dropout(x)

        # Step 4: Mamba layers — NO FiLM, same dynamics for all personas
        hidden = x
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            mamba_out = mamba(hidden)
            hidden    = norm(hidden + mamba_out)

        # Step 5: Project to trajectory space
        trajectory = self.trajectory_head(hidden)

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
    print("=== Testing Mamba Init Model ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PersonaInitMamba().to(device)
    model.eval()

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
    print(f"  trajectory : {out['trajectory'].shape}")
    print(f"  hidden     : {out['hidden'].shape}")
    print(f"  persona_vec: {out['persona_vec'].shape}")
    print("\n✅ Mamba Init model test passed!")
