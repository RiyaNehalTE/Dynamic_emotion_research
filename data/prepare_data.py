"""
prepare_data.py
---------------
Filters DeepDialogue to 8+ turn conversations only.
Creates train/val/test splits by conversation_id (no leakage).
Saves clean CSVs ready for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load raw parquet ───────────────────────────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet("data/train-00000-of-00001.parquet")

# Keep only text columns — ignore ALL audio columns
df = df[['conversation_id', 'turn_index', 'speaker', 'text', 'emotion', 'domain']]
print(f"Raw rows: {len(df)}")
print(f"Raw conversations: {df['conversation_id'].nunique()}")

# ── Filter to 8+ turn conversations ───────────────────────────────────────────
print("\nFiltering to 8+ turn conversations...")
conv_lengths = df.groupby('conversation_id')['turn_index'].max() + 1
valid_convs  = conv_lengths[conv_lengths >= 8].index
df_filtered  = df[df['conversation_id'].isin(valid_convs)].copy()

print(f"Filtered conversations : {df_filtered['conversation_id'].nunique()}")
print(f"Filtered rows         : {len(df_filtered)}")

# Verify lengths
lengths = df_filtered.groupby('conversation_id')['turn_index'].max() + 1
print(f"Min turns : {lengths.min()}")
print(f"Max turns : {lengths.max()}")
print(f"Avg turns : {lengths.mean():.1f}")

# ── Check emotion distribution ─────────────────────────────────────────────────
print("\nEmotion distribution in filtered data:")
print(df_filtered['emotion'].value_counts(normalize=True).round(3))

# ── Split by conversation_id (prevents leakage) ────────────────────────────────
print("\nCreating train/val/test splits...")
unique_convs = df_filtered['conversation_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_convs)

n           = len(unique_convs)
n_train     = int(0.80 * n)
n_val       = int(0.10 * n)

train_convs = unique_convs[:n_train]
val_convs   = unique_convs[n_train:n_train + n_val]
test_convs  = unique_convs[n_train + n_val:]

train_df = df_filtered[df_filtered['conversation_id'].isin(train_convs)].copy()
val_df   = df_filtered[df_filtered['conversation_id'].isin(val_convs)].copy()
test_df  = df_filtered[df_filtered['conversation_id'].isin(test_convs)].copy()

# ── Verify no leakage ──────────────────────────────────────────────────────────
train_ids = set(train_df['conversation_id'].unique())
val_ids   = set(val_df['conversation_id'].unique())
test_ids  = set(test_df['conversation_id'].unique())

assert len(train_ids & val_ids)  == 0, "Train-Val leakage!"
assert len(train_ids & test_ids) == 0, "Train-Test leakage!"
assert len(val_ids   & test_ids) == 0, "Val-Test leakage!"
print("No data leakage confirmed ✅")

# ── Print split stats ──────────────────────────────────────────────────────────
print(f"\nTrain: {train_df['conversation_id'].nunique()} convs, {len(train_df)} rows")
print(f"Val  : {val_df['conversation_id'].nunique()} convs, {len(val_df)} rows")
print(f"Test : {test_df['conversation_id'].nunique()} convs, {len(test_df)} rows")

# ── Save ───────────────────────────────────────────────────────────────────────
Path("data/splits").mkdir(exist_ok=True)
Path("data/raw").mkdir(exist_ok=True)

df_filtered.to_csv("data/raw/conversations_8plus.csv", index=False)
train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/val.csv",     index=False)
test_df.to_csv("data/splits/test.csv",   index=False)

# ── Save stats ─────────────────────────────────────────────────────────────────
stats = f"""
Dataset Statistics
==================
Total conversations : {df_filtered['conversation_id'].nunique()}
Total rows          : {len(df_filtered)}
Min turns           : {lengths.min()}
Max turns           : {lengths.max()}
Avg turns           : {lengths.mean():.1f}

Splits
======
Train: {train_df['conversation_id'].nunique()} conversations
Val  : {val_df['conversation_id'].nunique()} conversations
Test : {test_df['conversation_id'].nunique()} conversations

Emotion Classes (18)
====================
{df_filtered['emotion'].value_counts().to_string()}
"""

with open("data/raw/dataset_stats.txt", "w") as f:
    f.write(stats)

print("\n✅ All files saved!")
print("   data/raw/conversations_8plus.csv")
print("   data/splits/train.csv")
print("   data/splits/val.csv")
print("   data/splits/test.csv")
print("   data/raw/dataset_stats.txt")
