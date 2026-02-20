"""
STEP 3: finetune_from_splits.py
Phase 2 classification fine-tuning using ONLY train/val indices from splits/.
Test indices are never loaded here.

Run AFTER prepare_splits.py and train_from_splits.py (or your existing training).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset, Subset
from sklearn.metrics import roc_auc_score
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from train_fast import NUCLEOTIDE_MAP, HelicalGNNFrontend, HelicalDynamicsFunc, ContinuousODEBlock

SPLIT_DIR        = "./splits"
PRETRAIN_WEIGHTS = "circledot_former_loopseq_weights.pth"
BEST_WEIGHTS     = "circledot_classifier_best.pth"


# ── MODEL ──────────────────────────────────────────────────────────────────────
class CircleDotFormerClassifier(nn.Module):
    def __init__(self, node_dim=4, edge_dim=1, hidden_dim=64):
        super().__init__()
        self.gnn        = HelicalGNNFrontend(node_dim, edge_dim, hidden_dim)
        self.ode        = ContinuousODEBlock(HelicalDynamicsFunc(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),             nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x          = self.gnn(data.x, data.edge_index, data.edge_attr)
        x_evolved  = self.ode(x)
        graph_repr = torch.cat([global_mean_pool(x_evolved, data.batch),
                                 global_max_pool(x_evolved, data.batch)], dim=1)
        return self.classifier(graph_repr).squeeze(-1)


# ── DATASET ────────────────────────────────────────────────────────────────────
def build_graph(seq):
    seq     = seq.upper()
    x       = torch.tensor([NUCLEOTIDE_MAP.get(b, NUCLEOTIDE_MAP['N']) for b in seq],
                            dtype=torch.float)
    seq_len = len(seq)
    sources, targets, edge_attrs = [], [], []
    for i in range(seq_len):
        if i < seq_len - 1:
            sources.extend([i, i+1]);  targets.extend([i+1, i])
            edge_attrs.extend([[1.0], [1.0]])
        if i < seq_len - 10:
            sources.extend([i, i+10]); targets.extend([i+10, i])
            edge_attrs.extend([[0.2], [0.2]])
    return Data(x=x,
                edge_index=torch.tensor([sources, targets], dtype=torch.long),
                edge_attr =torch.tensor(edge_attrs, dtype=torch.float))


class FullDataset(TorchDataset):
    """Wraps the full sequence+label list saved by prepare_splits.py"""
    def __init__(self):
        saved          = torch.load(os.path.join(SPLIT_DIR, "full_dataset.pt"))
        self.sequences = saved['sequences']
        self.labels    = saved['labels']

    def __len__(self):  return len(self.sequences)

    def __getitem__(self, idx):
        g   = build_graph(self.sequences[idx])
        g.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return g


# ── FINE-TUNE ──────────────────────────────────────────────────────────────────
def finetune(epochs=25, batch_size=128, lr=2e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 2 fine-tuning on {device}")

    # Load the full dataset then immediately Subset to train/val only
    full_ds    = FullDataset()
    train_idx  = torch.load(os.path.join(SPLIT_DIR, "train_indices.pt"))
    val_idx    = torch.load(os.path.join(SPLIT_DIR, "val_indices.pt"))
    # test_indices.pt is deliberately NOT loaded here

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  "
          f"Test: SEALED (not loaded)")

    loader_kw    = dict(batch_size=batch_size, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # Build model and load pre-trained backbone
    model      = CircleDotFormerClassifier(node_dim=4, edge_dim=1, hidden_dim=64).to(device)
    ckpt       = torch.load(PRETRAIN_WEIGHTS, map_location=device)
    backbone   = {k: v for k, v in ckpt.items()
                  if k.startswith('gnn.') or k.startswith('ode.')}
    model.load_state_dict(backbone, strict=False)
    print(f"  Loaded {len(backbone)} backbone tensors from {PRETRAIN_WEIGHTS}")

    # Freeze physics backbone
    for name, p in model.named_parameters():
        if name.startswith('gnn.') or name.startswith('ode.'):
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    # Cosine schedule — smooth decay over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp    = device.type == 'cuda'
    scaler     = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_auroc = 0.0

    print("\n--- Classification Fine-Tuning (train/val only) ---")
    for epoch in range(1, epochs + 1):
        model.train();  total_loss = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(batch), batch.y.squeeze(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer);  scaler.update()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        model.eval();  preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(batch)
                preds.extend(torch.sigmoid(logits).cpu().float().tolist())
                labels.extend(batch.y.squeeze(-1).cpu().tolist())

        auroc     = roc_auc_score(labels, preds)
        avg_loss  = total_loss / len(train_ds)
        sota_flag = " *** BEATS SOTA 0.78 ***" if auroc > 0.78 else ""
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val AUROC: {auroc:.4f}"
              f" | LR: {optimizer.param_groups[0]['lr']:.6f}{sota_flag}")

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), BEST_WEIGHTS)
            print(f"           -> New best val AUROC — weights saved")

    print(f"\nBest val AUROC: {best_auroc:.4f}")
    print(f"Weights saved : {BEST_WEIGHTS}")
    print(">>> Training complete. Now run final_benchmark.py to unseal the test set.")


if __name__ == "__main__":
    finetune(epochs=25, batch_size=128, lr=2e-3)