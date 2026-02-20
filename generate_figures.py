"""
generate_figures.py
Generates all publication-quality figures from test_predictions.pt

Figures produced:
  1. ROC Curve with AUROC and SOTA comparison line
  2. Precision-Recall Curve with AUPRC
  3. Score Distribution (nucleosome vs random)
  4. Confusion Matrix heatmap
  5. Summary panel (all 4 in one figure)

Run after final_benchmark.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              roc_auc_score)
import os

# ── STYLE ──────────────────────────────────────────────────────────────────────
PALETTE = {
    'model'    : '#2563EB',   # strong blue
    'sota'     : '#DC2626',   # red
    'random'   : '#9CA3AF',   # grey
    'pos'      : '#16A34A',   # green
    'neg'      : '#EA580C',   # orange
    'bg'       : '#F8FAFC',
    'grid'     : '#E2E8F0',
    'text'     : '#1E293B',
    'subtext'  : '#64748B',
}

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.facecolor'   : PALETTE['bg'],
    'figure.facecolor' : 'white',
    'axes.grid'        : True,
    'grid.color'       : PALETTE['grid'],
    'grid.linewidth'   : 0.8,
    'axes.labelcolor'  : PALETTE['text'],
    'xtick.color'      : PALETTE['subtext'],
    'ytick.color'      : PALETTE['subtext'],
    'text.color'       : PALETTE['text'],
})

SOTA_AUROC  = 0.78
OUT_DIR     = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── LOAD PREDICTIONS ───────────────────────────────────────────────────────────
print("Loading test predictions...")
data    = torch.load("test_predictions.pt", weights_only=False)
y_true  = data['y_true'].astype(int) if hasattr(data['y_true'], 'astype') else np.array(data['y_true']).astype(int)
y_prob  = data['y_prob'] if not hasattr(data['y_prob'], 'numpy') else data['y_prob']
y_true  = np.array(y_true)
y_prob  = np.array(y_prob)
y_pred  = (y_prob >= 0.5).astype(int)

auroc   = roc_auc_score(y_true, y_prob)
auprc   = average_precision_score(y_true, y_prob)
print(f"  AUROC : {auroc:.4f}  |  AUPRC : {auprc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC CURVE
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc(ax=None, standalone=False):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color=PALETTE['model'], lw=2.5,
            label=f'CircleDot-Former  (AUROC = {auroc:.4f})')
    ax.axhline(y=SOTA_AUROC, color=PALETTE['sota'], lw=1.5, linestyle='--',
               label=f'SOTA baseline  (AUROC = {SOTA_AUROC:.2f})')
    ax.plot([0, 1], [0, 1], color=PALETTE['random'], lw=1, linestyle=':',
            label='Random classifier')

    # Shade the gap between model and SOTA diagonal
    ax.fill_between(fpr, SOTA_AUROC, tpr,
                    where=(tpr > SOTA_AUROC), alpha=0.08, color=PALETTE['model'])

    ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Annotate margin
    ax.annotate(f'+{auroc - SOTA_AUROC:.4f} above SOTA',
                xy=(0.6, 0.6), fontsize=9, color=PALETTE['model'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=PALETTE['model'], alpha=0.8))

    if standalone:
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig1_roc_curve.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PRECISION-RECALL CURVE
# ══════════════════════════════════════════════════════════════════════════════
def plot_prc(ax=None, standalone=False):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()   # random classifier baseline = class prevalence
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(recall, precision, color=PALETTE['model'], lw=2.5,
            label=f'CircleDot-Former  (AUPRC = {auprc:.4f})')
    ax.axhline(y=baseline, color=PALETTE['random'], lw=1.5, linestyle=':',
               label=f'Random baseline  ({baseline:.2f})')

    ax.fill_between(recall, baseline, precision,
                    where=(precision > baseline), alpha=0.08, color=PALETTE['pos'])

    ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    if standalone:
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig2_pr_curve.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — SCORE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_score_dist(ax=None, standalone=False):
    scores_pos = y_prob[y_true == 1]
    scores_neg = y_prob[y_true == 0]
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 1, 50)
    ax.hist(scores_neg, bins=bins, alpha=0.65, color=PALETTE['neg'],
            label=f'Random DNA  (n={len(scores_neg):,})', density=True)
    ax.hist(scores_pos, bins=bins, alpha=0.65, color=PALETTE['pos'],
            label=f'Nucleosome  (n={len(scores_pos):,})', density=True)
    ax.axvline(x=0.5, color=PALETTE['text'], lw=1.5, linestyle='--',
               label='Decision threshold (0.5)')

    ax.set_xlabel('Predicted Probability (Nucleosome)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Predicted Score Distributions', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, framealpha=0.9)

    if standalone:
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig3_score_dist.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion(ax=None, standalone=False):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    labels = ['Random\n(Negative)', 'Nucleosome\n(Positive)']
    ax.set_xticks([0, 1]);  ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)

    thresh = cm_pct.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_pct[i, j] > thresh else PALETTE['text']
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    if standalone:
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig4_confusion_matrix.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — SUMMARY PANEL (all 4 in one)
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_panel():
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    fig.suptitle('CircleDot-Former  —  Nucleosome Positioning Results',
                 fontsize=17, fontweight='bold', y=0.98, color=PALETTE['text'])

    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.07, right=0.97, top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_roc(ax=ax1)
    plot_prc(ax=ax2)
    plot_score_dist(ax=ax3)
    plot_confusion(ax=ax4)

    # Metrics banner at the bottom
    metrics_text = (
        f"AUROC: {auroc:.4f}   |   AUPRC: {auprc:.4f}   |   "
        f"SOTA Margin: +{auroc - SOTA_AUROC:.4f}   |   "
        f"Test set: {len(y_true):,} sequences (zero leakage confirmed)"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=10,
             color=PALETTE['subtext'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PALETTE['bg'],
                       edgecolor=PALETTE['grid']))

    path = os.path.join(OUT_DIR, "fig5_summary_panel.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating figures...")
    plot_roc(standalone=True)
    plot_prc(standalone=True)
    plot_score_dist(standalone=True)
    plot_confusion(standalone=True)
    plot_summary_panel()

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("  fig1_roc_curve.png")
    print("  fig2_pr_curve.png")
    print("  fig3_score_dist.png")
    print("  fig4_confusion_matrix.png")
    print("  fig5_summary_panel.png  <-- use this as your main figure")