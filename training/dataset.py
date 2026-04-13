"""
dataset.py
----------
Loads conversations and creates triplets for contrastive learning.

Triplet structure:
  anchor   = conversation C with persona A
  positive = different conversation with same persona A
  negative = same conversation C with different persona B

No emotion labels used — pure trajectory learning.
"""

import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict
import sys
sys.path.append(".")
from personas.personas import PERSONAS, get_persona_text


class ConversationTripletDataset(Dataset):
    def __init__(
        self,
        csv_path        : str,
        tokenizer_name  : str  = "roberta-base",
        max_utt_length  : int  = 64,
        max_persona_length: int = 64,
        max_turns       : int  = 10,
        seed            : int  = 42,
    ):
        random.seed(seed)

        self.max_utt_length     = max_utt_length
        self.max_persona_length = max_persona_length
        self.max_turns          = max_turns
        self.num_personas       = len(PERSONAS)

        # ── Load tokenizer ─────────────────────────────────────────────────────
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # ── Load CSV ───────────────────────────────────────────────────────────
        print(f"Loading data: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.sort_values(["conversation_id", "turn_index"])

        # ── Build conversation index ───────────────────────────────────────────
        # {conv_id: [utt_1, utt_2, ..., utt_T]}
        self.conversations = {}
        for conv_id, group in df.groupby("conversation_id"):
            utts = group.sort_values("turn_index")["text"].tolist()
            self.conversations[conv_id] = utts

        self.conv_ids = list(self.conversations.keys())
        print(f"Total conversations: {len(self.conv_ids)}")

        # ── Pre-tokenize all persona descriptions ──────────────────────────────
        print("Pre-tokenizing persona descriptions...")
        self.persona_tokens = {}
        for p in PERSONAS:
            tokens = self.tokenizer(
                p["description"],
                padding="max_length",
                truncation=True,
                max_length=self.max_persona_length,
                return_tensors="pt",
            )
            self.persona_tokens[p["id"]] = {
                "input_ids"      : tokens["input_ids"].squeeze(0),
                "attention_mask" : tokens["attention_mask"].squeeze(0),
            }

        # ── Pre-tokenize all utterances ────────────────────────────────────────
        print("Pre-tokenizing utterances...")
        self.utt_tokens = {}
        all_utts = set()
        for utts in self.conversations.values():
            all_utts.update(utts)

        for utt in all_utts:
            tokens = self.tokenizer(
                utt,
                padding="max_length",
                truncation=True,
                max_length=self.max_utt_length,
                return_tensors="pt",
            )
            self.utt_tokens[utt] = {
                "input_ids"      : tokens["input_ids"].squeeze(0),
                "attention_mask" : tokens["attention_mask"].squeeze(0),
            }

        print("Dataset ready ✅")

    def _encode_conversation(self, conv_id: str, persona_id: int) -> dict:
        """
        Encodes a single (conversation, persona) pair.
        Returns padded utterance tokens + persona tokens.
        """
        utts     = self.conversations[conv_id]
        num_turns = min(len(utts), self.max_turns)

        # Stack utterance tokens — [max_turns, seq_len]
        utt_input_ids      = []
        utt_attention_mask = []

        for t in range(self.max_turns):
            if t < num_turns:
                utt = utts[t]
                utt_input_ids.append(self.utt_tokens[utt]["input_ids"])
                utt_attention_mask.append(self.utt_tokens[utt]["attention_mask"])
            else:
                # Padding for shorter conversations
                utt_input_ids.append(
                    torch.zeros(self.max_utt_length, dtype=torch.long)
                )
                utt_attention_mask.append(
                    torch.zeros(self.max_utt_length, dtype=torch.long)
                )

        # Turn mask — which turns are real (not padding)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask[:num_turns] = True

        return {
            "persona_input_ids"      : self.persona_tokens[persona_id]["input_ids"],
            "persona_attention_mask" : self.persona_tokens[persona_id]["attention_mask"],
            "utt_input_ids"          : torch.stack(utt_input_ids),       # [max_turns, seq_len]
            "utt_attention_mask"     : torch.stack(utt_attention_mask),  # [max_turns, seq_len]
            "turn_mask"              : turn_mask,                         # [max_turns]
            "num_turns"              : torch.tensor(num_turns),
            "conversation_id"        : conv_id,
            "persona_id"             : persona_id,
        }

    def __len__(self):
        # Each (conversation, persona) pair is one anchor
        return len(self.conv_ids) * self.num_personas

    def __getitem__(self, idx):
        # Determine anchor conversation and persona
        conv_idx   = idx // self.num_personas
        persona_id = idx  % self.num_personas
        conv_id    = self.conv_ids[conv_idx]

        # ── ANCHOR ────────────────────────────────────────────────────────────
        anchor = self._encode_conversation(conv_id, persona_id)

        # ── POSITIVE: same persona, different conversation ─────────────────────
        other_convs = [c for c in self.conv_ids if c != conv_id]
        pos_conv_id = random.choice(other_convs)
        positive    = self._encode_conversation(pos_conv_id, persona_id)

        # ── NEGATIVE: same conversation, different persona ─────────────────────
        other_personas = [p for p in range(self.num_personas) if p != persona_id]
        neg_persona_id = random.choice(other_personas)
        negative       = self._encode_conversation(conv_id, neg_persona_id)

        return anchor, positive, negative


def collate_triplets(batch):
    """
    Collates a list of (anchor, positive, negative) triplets into batches.
    """
    anchors, positives, negatives = zip(*batch)

    def stack(samples, key):
        vals = [s[key] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            return torch.stack(vals)
        return vals  # strings stay as lists

    def collate_one(samples):
        return {
            "persona_input_ids"      : stack(samples, "persona_input_ids"),
            "persona_attention_mask" : stack(samples, "persona_attention_mask"),
            "utt_input_ids"          : stack(samples, "utt_input_ids"),
            "utt_attention_mask"     : stack(samples, "utt_attention_mask"),
            "turn_mask"              : stack(samples, "turn_mask"),
            "num_turns"              : stack(samples, "num_turns"),
            "conversation_id"        : stack(samples, "conversation_id"),
            "persona_id"             : stack(samples, "persona_id"),
        }

    return collate_one(anchors), collate_one(positives), collate_one(negatives)


if __name__ == "__main__":
    print("=== Testing Dataset ===")
    ds = ConversationTripletDataset(
        csv_path="data/splits/train.csv",
        max_turns=10,
    )
    print(f"\nDataset size: {len(ds)}")
    print(f"(6349 conversations × 20 personas = {6349*20} samples)")

    # Test one sample
    anchor, positive, negative = ds[0]
    print(f"\nAnchor:")
    print(f"  conv_id         : {anchor['conversation_id']}")
    print(f"  persona_id      : {anchor['persona_id']}")
    print(f"  num_turns       : {anchor['num_turns']}")
    print(f"  utt_input_ids   : {anchor['utt_input_ids'].shape}")
    print(f"  persona_input_ids: {anchor['persona_input_ids'].shape}")
    print(f"  turn_mask       : {anchor['turn_mask']}")

    print(f"\nPositive (same persona={positive['persona_id']}, "
          f"diff conv={positive['conversation_id'] != anchor['conversation_id']})")
    print(f"Negative (diff persona={negative['persona_id']}, "
          f"same conv={negative['conversation_id'] == anchor['conversation_id']})")

    print("\n✅ Dataset test passed!")
