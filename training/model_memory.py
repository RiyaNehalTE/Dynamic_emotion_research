"""
model_memory.py (v2 — MEMORY — FINAL)
---------------------------------------
PersonaMemoryMamba — Model 3 for paper.

Formula: mₜ = α_p·mₜ₋₁ + (1-α_p)·eₜ
α_p = Emotional Inertia Coefficient (MEMORY)

Stoic    → HIGH α_p (strong memory, low drift)
Volatile → LOW  α_p (weak memory,  high drift)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class PersonaMemoryMamba(nn.Module):

    def __init__(
        self,
        encoder_name    : str   = "roberta-base",
        encoder_hidden  : int   = 768,
        d_model         : int   = 256,
        persona_proj_dim: int   = 256,
        d_trajectory    : int   = 64,
        dropout         : float = 0.1,
        freeze_encoder  : bool  = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("RoBERTa encoder frozen ✅")

        self.persona_proj = nn.Linear(encoder_hidden, persona_proj_dim)

        self.h0_proj = nn.Sequential(
            nn.Linear(persona_proj_dim, d_model),
            nn.Tanh(),
        )
        self.h0_scale = nn.Parameter(torch.tensor(0.1))

        # α_p: Emotional Inertia (Memory)
        # Stoic → HIGH, Volatile → LOW
        self.alpha_proj = nn.Sequential(
            nn.Linear(persona_proj_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.alpha_proj[5].bias, 0.5)
        nn.init.normal_(self.alpha_proj[5].weight, std=1.0)

        self.utt_proj = nn.Linear(encoder_hidden, d_model)

        self.emotion_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

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
        print(f"Model type      : PersonaMemoryMamba v2 (Memory)")
        print(f"Formula         : mₜ = α_p·mₜ₋₁ + (1-α_p)·eₜ")

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
        flat_ids  = utt_input_ids.reshape(batch * turns, seq_len)
        flat_mask = utt_attention_mask.reshape(batch * turns, seq_len)
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

        n_turns = utt_input_ids.size(1)

        persona_vec = self.encode_persona(
            persona_input_ids, persona_attention_mask
        )

        # α_p = Emotional Inertia
        alpha_p = self.alpha_proj(persona_vec)  # [batch, 1]

        # h₀ scaled down
        h = self.h0_proj(persona_vec) * self.h0_scale
        h = self.dropout(h)

        x = self.encode_utterances(utt_input_ids, utt_attention_mask)

        # mₜ = α_p·mₜ₋₁ + (1-α_p)·eₜ  ← v2 MEMORY formula
        trajectories = []
        for t in range(n_turns):
            e_t = self.emotion_proj(x[:, t, :])
            h   = alpha_p * h + (1 - alpha_p) * e_t
            trajectories.append(h)

        hidden     = torch.stack(trajectories, dim=1)
        trajectory = self.trajectory_head(hidden)

        if turn_mask is not None:
            mask       = turn_mask.unsqueeze(-1).float()
            trajectory = trajectory * mask
            hidden     = hidden     * mask

        return {
            "trajectory" : trajectory,
            "hidden"     : hidden,
            "persona_vec": persona_vec,
            "alpha_p"    : alpha_p,
        }

    def get_persona_alphas(self, persona_input_ids, persona_attention_mask):
        with torch.no_grad():
            persona_vec = self.encode_persona(
                persona_input_ids, persona_attention_mask
            )
            return self.alpha_proj(persona_vec)
