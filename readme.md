# CircleDot-Former

**A Physics-Informed Graph Neural ODE for Nucleosome Positioning Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

> Pre-print: *CircleDot-Former: A Physics-Informed Graph Neural ODE for Genome-Scale Nucleosome Positioning Prediction* — Saurab Singh (2025) [[arXiv link when available]]

---

## Overview

CircleDot-Former predicts whether a 50-bp DNA sequence will be occupied by a nucleosome *in vivo*, using only raw sequence as input. Rather than treating DNA as a linear token string like conventional CNNs, it encodes DNA as a **helical graph** — capturing both the covalent backbone and 3D helical geometry — and models continuous mechanical dynamics via a **Neural Ordinary Differential Equation (Neural ODE)**.

The model is trained in two stages:
1. **Physics pre-training**: regress intrinsic cyclizability scores (C₀) from Loop-seq, learning DNA bending mechanics directly from biophysical data
2. **Classification fine-tuning**: frozen backbone + lightweight classification head trained to distinguish in-vivo nucleosomal sequences from synthetic random DNA

**Key result: AUROC = 0.8663 on a sealed held-out test set — with 44K parameters and under 600 MB of GPU memory on a consumer laptop.**

---

## Architecture

```
DNA Sequence (50 bp)
       │
       ▼
Helical Graph Construction
  • Nodes: one-hot nucleotides ∈ ℝ⁴
  • Backbone edges (weight 1.0): covalent bonds i↔i+1
  • Helical edges  (weight 0.2): spatial contacts i↔i+10
       │
       ▼
GATv2 Message Passing (2 layers, hidden dim 64)
  • Dynamic attention weights conditioned on source + target
  • Edge features encode backbone vs. helical bond type
       │
       ▼
Neural ODE Block
  • Continuously evolves node representations t: 0→1
  • dH/dt = f_θ(H), solved by RK4
  • Backprop via adjoint method → O(1) memory
       │
       ▼
Mean + Max Pooling → ℝ¹²⁸
       │
       ▼
Classification Head (3-layer MLP, dropout 0.3/0.2)
       │
       ▼
Nucleosome probability ∈ [0, 1]
```

---

## Results

Evaluated on a held-out test set of **4,857 sequences** (sealed before training, evaluated exactly once):

| Metric | Value |
|--------|-------|
| **AUROC** | **0.8663** |
| **AUPRC** | **0.9073** |
| Accuracy | 0.7904 |
| Precision | 0.8081 |
| Recall | 0.8644 |
| F1 Score | 0.8353 |
| Val / Test AUROC gap | 0.0004 |
| Leakage check | ✅ PASSED (zero overlap) |

**Hardware:** Consumer laptop GPU · Peak VRAM: < 600 MB · Training time: ~63 min total

---

## Dataset

All data are from **Basu et al., *Nature* 589, 462–467 (2021)**
DOI: [10.1038/s41586-020-03052-3](https://doi.org/10.1038/s41586-020-03052-3)

| File | Description | Role | Sequences |
|------|-------------|------|-----------|
| MOESM6 (Dataset 3) | Random Library — synthetic 50-bp sequences with C₀ scores | Phase 1 pre-training + negative class | 12,472 |
| MOESM4 (Dataset 1) | Cerevisiae Nucleosomal Library — in-vivo nucleosome dyad sequences | Positive class | 19,907 |

Download the supplementary data files from the Nature paper and place them in `data/`:
```
data/
  41586_2020_3052_MOESM4_ESM.txt
  41586_2020_3052_MOESM6_ESM.txt
```

---

## Repository Structure

```
circledot-former/
├── train_fast.py          # GPU-optimised Phase 1 pre-training (C₀ regression)
├── prepare_split.py       # Deterministic 70/15/15 stratified split (seed 42)
├── fine_tune.py           # Phase 2 classification fine-tuning (frozen backbone)
├── final_beanch_mark.py   # Sealed test-set evaluation (run exactly once)
├── generate_figures.py    # Publication-quality ROC/PR/distribution figures
├── splits/
│   ├── train_indices.pt   # Deterministic split indices (seed 42)
│   ├── val_indices.pt
│   ├── test_indices.pt    # Sealed — do not load until final evaluation
│   └── split_manifest.txt # Split statistics and overlap assertions
├── figures/
│   ├── fig1_roc_curve.png
│   ├── fig2_pr_curve.png
│   ├── fig3_score_dist.png
│   ├── fig4_confusion_matrix.png
│   └── fig5_summary_panel.png
├── data/                  # Place MOESM4 and MOESM6 here (not tracked by git)
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/circledot-former.git
cd circledot-former

# Create and activate environment
python -m venv newenv
newenv\Scripts\activate       # Windows
# source newenv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torchdiffeq scikit-learn pandas numpy matplotlib
```

---

## Reproducing the Results

Run the scripts in this exact order:

**Step 1 — Create deterministic splits**
```bash
python prepare_split.py
# Output: splits/train_indices.pt, splits/val_indices.pt, splits/test_indices.pt
# The test set is now sealed. Do not open test_indices.pt until Step 4.
```

**Step 2 — Phase 1: Pre-train on C₀ cyclizability regression**
```bash
python train_fast.py
# Output: circledot_former_loopseq_weights.pth
# Expected: Val MSE ≈ 0.132 after 55 epochs
```

**Step 3 — Phase 2: Fine-tune classification head**
```bash
python fine_tune.py
# Output: circledot_classifier_best.pth
# Expected: Best Val AUROC ≈ 0.8667
```

**Step 4 — Final benchmark (run exactly once)**
```bash
python final_beanch_mark.py
# Unseals test set for the first time
# Output: final_benchmark_results.txt, test_predictions.pt
# Expected: Test AUROC ≈ 0.8663
```

**Step 5 — Generate figures**
```bash
python generate_figures.py
# Output: figures/fig1_roc_curve.png ... fig5_summary_panel.png
```

> ⚠️ **Reproducibility note:** All random seeds are fixed (NumPy seed 42, PyTorch seed via `torch.manual_seed`). Split indices are provided in `splits/` so results are exactly reproducible without re-running `prepare_split.py`.

---

## Why This Approach is Novel

| Aspect | Prior CNN approaches | CircleDot-Former |
|--------|---------------------|-----------------|
| DNA representation | 1D token sequence | Helical graph (backbone + 10-bp stacking edges) |
| Dynamics model | Discrete layers | Continuous Neural ODE |
| Training signal | Classification labels only | C₀ biophysics → transfer → classification |
| Parameters | ~500K | **44,289** |
| Peak VRAM | ~2 GB | **< 600 MB** |
| AUROC (this benchmark) | ~0.78 | **0.8663** |

The helical edges (connecting nucleotides 10 bp apart) encode one complete turn of B-form DNA — the fundamental spatial frequency of nucleosome-positioning signals. The Neural ODE models the continuous redistribution of elastic strain energy along the helix, consistent with the worm-like chain model of DNA mechanics.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@article{singh2025circledotformer,
  title   = {CircleDot-Former: A Physics-Informed Graph Neural ODE
             for Genome-Scale Nucleosome Positioning Prediction},
  author  = {Singh, Saurab},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://arxiv.org/abs/[ID when available]}
}
```

Please also cite the original Loop-seq dataset:

```bibtex
@article{basu2021measuring,
  title   = {Measuring DNA mechanics on the genome scale},
  author  = {Basu, Aakash and Bobrovnikov, Dmitriy G and Qureshi, Zan and
             Kayikcioglu, Tunc and Ngo, Thuy T M and Ranjan, Anand and
             Eustermann, Sebastian and Cieza, Basilio and Morgan, Michael T and
             Hejna, Miroslav and Rube, H Tomas and Hopfner, Karl-Peter and
             Wolberger, Cynthia and Song, Jun S and Ha, Taekjip},
  journal = {Nature},
  volume  = {589},
  pages   = {462--467},
  year    = {2021},
  doi     = {10.1038/s41586-020-03052-3}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Data (MOESM4, MOESM6) is from Basu et al. (2021) and subject to the terms of the original Nature publication. Please refer to the original paper for data usage terms.

---

## Contact

**Saurab Singh** · Independent Researcher  
📧 saurabsingh778@gmail.com  
🔗 [arXiv preprint](#) · [GitHub](https://github.com/[username]/circledot-former)