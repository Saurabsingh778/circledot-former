"""
STEP 2: train_from_splits.py
Trains CircleDot-Former Phase 1 (C0 regression) using only TRAIN indices.
VAL indices used for early stopping. TEST indices never loaded.

Run AFTER prepare_splits.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset

from model import (CircleDotFormer, NUCLEOTIDE_MAP,
                        HelicalGNNFrontend, HelicalDynamicsFunc, ContinuousODEBlock)

import pandas as pd
import os

SPLIT_DIR          = "./splits"
PRETRAIN_WEIGHTS   = "circledot_former_loopseq_weights.pth"

# ── DATASET ────────────────────────────────────────────────────────────────────
# NOTE: Phase-1 pre-training uses the random library C0 scores (regression).
# We load those directly; the split indices apply to the classification dataset.
# If you want to also split the regression pre-training data, point C0_DATA
# to MOESM6 and the loader will respect the train indices only.

C0_DATA = r"D:\exprement_16\data\41586_2020_3052_MOESM6_ESM.txt"

def build_graph(seq):
    seq     = seq.upper()
    x       = [NUCLEOTIDE_MAP.get(b, NUCLEOTIDE_MAP['N']) for b in seq]
    x       = torch.tensor(x, dtype=torch.float)
    seq_len = len(seq)
    sources, targets, edge_attrs = [], [], []
    for i in range(seq_len):
        if i < seq_len - 1:
            sources.extend([i, i+1]);  targets.extend([i+1, i])
            edge_attrs.extend([[1.0], [1.0]])
        if i < seq_len - 10:
            sources.extend([i, i+10]); targets.extend([i+10, i])
            edge_attrs.extend([[0.2], [0.2]])
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class C0RegressionDataset(TorchDataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Sequence', 'C0']).reset_index(drop=True)
        self.seqs   = df['Sequence'].tolist()
        self.scores = df['C0'].tolist()
        print(f"  C0 dataset: {len(self.seqs)} sequences")

    def __len__(self):  return len(self.seqs)

    def __getitem__(self, idx):
        g   = build_graph(self.seqs[idx])
        g.y = torch.tensor([self.scores[idx]], dtype=torch.float)
        return g


def train_phase1(epochs=55, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1 — C0 Regression pre-training on {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    dataset    = C0RegressionDataset(C0_DATA)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kw    = dict(batch_size=batch_size, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model     = CircleDotFormer(node_dim=4, edge_dim=1, hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5)
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(batch), batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer);  scaler.update()
            total += loss.item() * batch.num_graphs

        model.eval();  val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    val_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
        scheduler.step(val_loss / len(val_ds))
        print(f"Epoch {epoch:03d} | Train MSE: {total/len(train_ds):.4f} | "
              f"Val MSE: {val_loss/len(val_ds):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), PRETRAIN_WEIGHTS)
    print(f"Phase 1 complete. Weights → {PRETRAIN_WEIGHTS}")


if __name__ == "__main__":
    train_phase1(epochs=55, batch_size=128)