"""
STEP 1: prepare_splits.py
Run this ONCE before any training.
Splits all data deterministically and saves index files to disk.
The test set is NEVER touched until final_benchmark.py is run.

Split strategy:
  - Nucleosome (MOESM4): 70% train | 15% val | 15% test
  - Random     (MOESM6): 70% train | 15% val | 15% test
  - Stratified: class balance is preserved in every split

Output files (in ./splits/):
  train_indices.pt   val_indices.pt   test_indices.pt
  split_manifest.txt  ← human-readable record of exactly what went where
"""

import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# ── CONFIG ─────────────────────────────────────────────────────────────────────
NUCLEOSOME_DATA = r"D:\exprement_16\data\41586_2020_3052_MOESM4_ESM.txt"
RANDOM_DATA     = r"D:\exprement_16\data\41586_2020_3052_MOESM6_ESM.txt"
SPLIT_DIR       = "./splits"
SEED            = 42          # fixed forever — never change this
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# TEST_RATIO is the remainder = 0.15

os.makedirs(SPLIT_DIR, exist_ok=True)

# ── LOAD ───────────────────────────────────────────────────────────────────────
def load_sequences(path, label):
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.strip()
    if 'Sequence' not in df.columns:
        merged_col = [c for c in df.columns if 'Sequence' in c][0]
        df['Sequence'] = df[merged_col].astype(str).str.split().str[-1]
    seqs = df['Sequence'].dropna().tolist()
    print(f"  Loaded {len(seqs)} sequences (label={label}) from {os.path.basename(path)}")
    return seqs

print("Loading datasets...")
seqs_nuc  = load_sequences(NUCLEOSOME_DATA, label=1)
seqs_rand = load_sequences(RANDOM_DATA,     label=0)

all_seqs   = seqs_nuc  + seqs_rand
all_labels = [1] * len(seqs_nuc) + [0] * len(seqs_rand)
all_idx    = list(range(len(all_seqs)))

# ── SPLIT ──────────────────────────────────────────────────────────────────────
# First cut: train vs (val + test), stratified on label
train_idx, temp_idx = train_test_split(
    all_idx,
    test_size  = 1.0 - TRAIN_RATIO,
    stratify   = all_labels,
    random_state = SEED
)

# Labels for the temp pool
temp_labels = [all_labels[i] for i in temp_idx]

# Second cut: val vs test (50/50 of temp = 15/15 of total), stratified
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size    = 0.5,
    stratify     = temp_labels,
    random_state = SEED
)

# ── VERIFY NO OVERLAP ──────────────────────────────────────────────────────────
assert len(set(train_idx) & set(val_idx))  == 0, "LEAK: train/val overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "LEAK: train/test overlap!"
assert len(set(val_idx)   & set(test_idx)) == 0, "LEAK: val/test overlap!"
assert len(train_idx) + len(val_idx) + len(test_idx) == len(all_seqs), "Size mismatch!"

# ── SAVE INDICES ───────────────────────────────────────────────────────────────
torch.save(train_idx, os.path.join(SPLIT_DIR, "train_indices.pt"))
torch.save(val_idx,   os.path.join(SPLIT_DIR, "val_indices.pt"))
torch.save(test_idx,  os.path.join(SPLIT_DIR, "test_indices.pt"))

# Also save the full sequence+label list so training never reloads from CSV differently
torch.save({'sequences': all_seqs, 'labels': all_labels},
           os.path.join(SPLIT_DIR, "full_dataset.pt"))

# ── CLASS COUNTS PER SPLIT ─────────────────────────────────────────────────────
def class_counts(idx_list):
    labs = [all_labels[i] for i in idx_list]
    return labs.count(1), labs.count(0)

tr_pos, tr_neg = class_counts(train_idx)
va_pos, va_neg = class_counts(val_idx)
te_pos, te_neg = class_counts(test_idx)

# ── MANIFEST ───────────────────────────────────────────────────────────────────
manifest = f"""
CircleDot-Former — Deterministic Data Split Manifest
=====================================================
Generated with SEED = {SEED}

Source files:
  Nucleosome (pos=1) : {NUCLEOSOME_DATA}
  Random     (neg=0) : {RANDOM_DATA}

Total sequences      : {len(all_seqs):,}
  Nucleosome (+)     : {len(seqs_nuc):,}
  Random     (-)     : {len(seqs_rand):,}

Split sizes:
  TRAIN : {len(train_idx):,}  ({tr_pos:,} pos  |  {tr_neg:,} neg)
  VAL   : {len(val_idx):,}   ({va_pos:,} pos  |  {va_neg:,} neg)
  TEST  : {len(test_idx):,}   ({te_pos:,} pos  |  {te_neg:,} neg)

Overlap checks       : PASSED (zero leakage confirmed)

WARNING: TEST SET IS SEALED. Do not evaluate on test_indices.pt until
   training is fully complete and final model is chosen.
"""

with open(os.path.join(SPLIT_DIR, "split_manifest.txt"), "w", encoding="utf-8") as f:
    f.write(manifest)

print(manifest)
print(f"Splits saved to: {SPLIT_DIR}/")
print(">>> TEST SET IS NOW SEALED. Do not touch test_indices.pt until final benchmark.")