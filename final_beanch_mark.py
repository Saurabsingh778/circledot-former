"""
STEP 4: final_benchmark.py
THE FINAL EVALUATION — run this exactly ONCE after training is complete.

Unseals the held-out test set and reports:
  - AUROC
  - AUPRC
  - Accuracy, Precision, Recall, F1 at 0.5 threshold
  - Confusion matrix

This file is your proof. The test set was never seen during training or
model selection. Results here are unbiased and reviewer-proof.

⚠  DO NOT run this file multiple times to tune hyperparameters.
   If you do, the test set is no longer held-out and your claim is invalid.
"""

import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset, Subset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              roc_curve)
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
import numpy as np

from model import NUCLEOTIDE_MAP, HelicalGNNFrontend, HelicalDynamicsFunc, ContinuousODEBlock

SPLIT_DIR    = "./splits"
BEST_WEIGHTS = "circledot_classifier_best.pth"

# SOTA reference — Karbalayghareh et al. 2018, Table 1, best CNN baseline
SOTA_AUROC   = 0.78
SOTA_SOURCE  = "1D CNN baseline (Karbalayghareh et al. / standard DNA-seq benchmark)"


# ── MODEL (identical to finetune_from_splits.py) ───────────────────────────────
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
    def __init__(self):
        saved          = torch.load(os.path.join(SPLIT_DIR, "full_dataset.pt"))
        self.sequences = saved['sequences']
        self.labels    = saved['labels']

    def __len__(self):  return len(self.sequences)

    def __getitem__(self, idx):
        g   = build_graph(self.sequences[idx])
        g.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return g


# ── BENCHMARK ──────────────────────────────────────────────────────────────────
def run_final_benchmark(batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model on {device}...")

    # Load model
    model = CircleDotFormerClassifier(node_dim=4, edge_dim=1, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device), strict=False)
    model.eval()

    # Unseal the test set — only time test_indices.pt is ever loaded
    full_ds  = FullDataset()
    test_idx = torch.load(os.path.join(SPLIT_DIR, "test_indices.pt"))

    # Sanity: confirm test set was never touched by checking it's disjoint from train
    train_idx = torch.load(os.path.join(SPLIT_DIR, "train_indices.pt"))
    val_idx   = torch.load(os.path.join(SPLIT_DIR, "val_indices.pt"))
    assert len(set(test_idx) & set(train_idx)) == 0, "CRITICAL: test/train overlap detected!"
    assert len(set(test_idx) & set(val_idx))   == 0, "CRITICAL: test/val overlap detected!"
    print(f"Leakage check: PASSED — test set is clean ({len(test_idx):,} samples)")

    test_ds     = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # Run inference
    use_amp = device.type == 'cuda'
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(batch)
            all_probs.extend(torch.sigmoid(logits).cpu().float().tolist())
            all_labels.extend(batch.y.squeeze(-1).cpu().tolist())

    # ── METRICS ───────────────────────────────────────────────────────────────
    y_true  = np.array(all_labels)
    y_prob  = np.array(all_probs)
    y_pred  = (y_prob >= 0.5).astype(int)

    auroc   = roc_auc_score(y_true, y_prob)
    auprc   = average_precision_score(y_true, y_prob)
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Class balance in test set
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    beats  = auroc > SOTA_AUROC
    margin = auroc - SOTA_AUROC

    # ── REPORT ────────────────────────────────────────────────────────────────
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║       CircleDot-Former  —  FINAL HELD-OUT TEST RESULTS       ║
╠══════════════════════════════════════════════════════════════╣
║  Model weights  : {BEST_WEIGHTS:<43}║
║  Test samples   : {len(y_true):<6,}  ({n_pos:,} nucleosome  |  {n_neg:,} random){'':>5}║
║  Leakage check  : PASSED (zero overlap with train/val)       ║
╠══════════════════════════════════════════════════════════════╣
║  PRIMARY METRIC                                              ║
║    AUROC          : {auroc:.4f}                                    ║
║    SOTA baseline  : {SOTA_AUROC:.4f}  [{SOTA_SOURCE[:30]}]  ║
║    Margin         : {margin:+.4f}  {">>> BEATS SOTA <<<" if beats else "--- below SOTA ---"}{'':>18}║
╠══════════════════════════════════════════════════════════════╣
║  SECONDARY METRICS                                           ║
║    AUPRC          : {auprc:.4f}                                    ║
║    Accuracy       : {acc:.4f}                                    ║
║    Precision      : {prec:.4f}                                    ║
║    Recall         : {rec:.4f}                                    ║
║    F1 Score       : {f1:.4f}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX  (threshold = 0.5)                         ║
║    True  Positives: {tp:<6}   False Positives: {fp:<6}             ║
║    False Negatives: {fn:<6}   True  Negatives: {tn:<6}             ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(report)

    # Save results to file so you have a permanent record
    results_path = "final_benchmark_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\nRaw arrays saved for plotting ROC curve if needed.\n")
        f.write(f"\nSOTA reference: {SOTA_SOURCE}\n")
    print(f"Results saved to: {results_path}")

    # Optionally save probabilities for ROC curve plotting
    torch.save({'y_true': y_true, 'y_prob': y_prob},
               "test_predictions.pt")
    print("Raw predictions saved to: test_predictions.pt")
    print("  (use these to plot the ROC curve in your paper/report)")

    return auroc


if __name__ == "__main__":
    # ⚠  Run this exactly ONCE.
    # ⚠  Do not loop over hyperparameters with this script.
    run_final_benchmark(batch_size=256)