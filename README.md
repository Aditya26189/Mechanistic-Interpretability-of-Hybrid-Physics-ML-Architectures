# D4-LensPINN v4

### Physics-Informed Gravitational Lens Substructure Classifier with D4 Equivariance & Mechanistic Interpretability

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![escnn](https://img.shields.io/badge/escnn-Equivariant_CNN-00B140?style=for-the-badge)](https://github.com/QUVA-Lab/escnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## What This Is

This repository contains a **physics-informed neural network** (PINN) for classifying gravitational lens images by dark-matter substructure type, paired with a rigorous **mechanistic interpretability (MI)** study of how the model's internal representations behave under the D4 symmetry group.

The core novelty is twofold:

1. **Differentiable Physics Pipeline** — The gravitational lensing equation `β = θ − α(θ)` is implemented as a parameter-free, fully differentiable PyTorch module. The network simultaneously learns to predict the convergence map κ̂ and is penalised if it violates the Poisson equation `∇²Ψ = 2κ̂`.

2. **D4 Group Equivariance** — The convergence estimator (Stage 1) is built using `escnn` equivariant layers under the dihedral group D4 = {4 rotations × 1 flip}. This is the correct symmetry for gravitational lensing, which is invariant under both rotation and parity reflection.

---

## Task

Classify 150×150 grayscale gravitational lens images into three dark-matter substructure classes:

| Label | Physical Meaning |
|:---:|:---|
| `no` | Smooth lens — no dark matter substructure |
| `sphere` | Spherical CDM subhalo mass clumps |
| `vort` | Vortex / WDM substructure |

**30,000 images** total · **Metric:** Macro-average one-vs-rest AUC-ROC

---

## Results

| Model | Macro AUC (No TTA) | Macro AUC (D4-TTA) | Params |
|:---|:---:|:---:|:---:|
| **D4LensPINN** | **0.9786** | **0.9809** | 20.3M |
| VanillaLensPINN | 0.9776 | — | 20.3M |
| ResNet18 Baseline | 0.9182 | — | 11.2M |

---

## Architecture

```
Input I : (B, 1, 150, 150)
    │
    ▼  Stage 0 · PhysicsPreprocess            [0 params]
    │   log-ratio map + Sobel saddle-point feature
    │   OUT: (B, 2, 150, 150)
    │
    ▼  Stage 1 · EfficientD4UNet              [0.46M params]
    │   D4-equivariant U-Net → convergence map κ̂ ≥ 0
    │   OUT: (B, 1, 150, 150)
    │
    ▼  Stage 2 · DifferentiablePhysicsEngine  [0 params]
    │   PoissonSolverFFT:  ∇²Ψ = 2κ̂  (spectral)
    │   DeflectionField:   α = ∇Ψ
    │   InverseLensLayer:  Ŝ, R = β=θ-α(θ), I-Ŝ
    │   OUT: Ŝ, R  each (B, 1, 150, 150)
    │
    ▼  Stage 3 · EfficientNetV2Head           [19.87M trainable]
    │   input_proj: Conv2d(4→3, 1×1) → BN → SiLU
    │   EfficientNetV2-S (features.0–2 frozen, features.3–7 fine-tuned)
    │   avgpool → Dropout(0.2) → Linear(1280→3)
    │   OUT: logits (B, 3)
    │
    ▼  Output: logits (B, 3) + κ̂ (B, 1, 150, 150)
```

---

## Repository Contents

> Files and directories listed here are the ones **tracked in version control**. Training runs, checkpoints, virtual environments, and cache files are excluded via `.gitignore`.

```
d7/
├── D4-PINN + Differentiable Lensing Engine.ipynb   # Main model: architecture, training, evaluation
├── d4-PINN-MI notebook.ipynb                       # MI experiment for D4LensPINN vs ResNet18
├── vanilla-notebook.ipynb                          # VanillaLensPINN control experiment
│
├── d4-PINN-MI results/                             # Compiled MI outputs for D4LensPINN
│   ├── mi_results_full.csv                         # Full intervention results (4.5 MB)
│   ├── probe_results.csv                           # Linear probe CV accuracy per hook
│   ├── mechanistic_probe_comparison.csv            # D4 vs Vanilla probe comparison
│   ├── findings_summary.json                       # Top-level MI findings
│   ├── figure1.png, figure4.png, ...               # Publication-ready figures
│   └── [+ 100 more incremental checkpoint CSVs]
│
├── vanilla results/                                # MI outputs for VanillaLensPINN control
│   ├── mi_results_full.csv
│   ├── vanilla_probe_summary.csv
│   └── vanilla_figure1.png
│
├── docs/                                           # Technical documentation
│   ├── ARCHITECTURE.md                             # All classes, modules, design decisions
│   ├── TRAINING.md                                 # Loss functions, optimizers, training phases
│   └── MECHANISTIC_INTERPRETABILITY.md             # MI methodology, hooks, analysis outputs
│
├── README.md
└── .gitignore
```

---

## Documentation

| Document | Contents |
|:---|:---|
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Full class-by-class breakdown of every module — PhysicsPreprocess, EfficientD4UNet, PoissonSolverFFT, DeflectionField, InverseLensLayer, EfficientNetV2Head, D4LensPINN, ResNet18Baseline, VanillaLensPINN |
| **[TRAINING.md](docs/TRAINING.md)** | Dataset split, loss functions (CE + 4-term physics loss), optimizer, warmup+cosine scheduler, Mixup, Optuna hyperparameter search, multi-GPU setup, TTA evaluation |
| **[MECHANISTIC_INTERPRETABILITY.md](docs/MECHANISTIC_INTERPRETABILITY.md)** | Hook table (17 layers), activation patching methodology, `cache_activations`, `intervention_pass`, sanity checks (SC1/SC3/SC5), linear probing, chain property failure, probe-delta dissociation, full output file reference |
| **[RESULTS.md](docs/RESULTS.md)** | All experimental results with real numbers — AUC table with TTA accuracy regression, 5 experiments run, Δ-L2 hook profiles, linear probe accuracy (D4 vs Vanilla), Kruskal-Wallis tests, bootstrap CIs, spectral analysis, ablation study, full output file index |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy==1.26.4 escnn gdown \
            pandas matplotlib scikit-learn scipy optuna pillow
```

> `numpy==1.26.4` is pinned — `escnn` is not compatible with numpy 2.x. The notebooks include a runtime guard that auto-installs and restarts the kernel if the wrong version is detected.

### 2. Run the Main Model

Open `D4-PINN + Differentiable Lensing Engine.ipynb` and run cells in order:

| Cell | Action |
|:---:|:---|
| 1 | Install `numpy==1.26.4`, `escnn`, `gdown` — restarts kernel if needed |
| 2 | Imports, global hyperparameters, TF32, seeds |
| 3 | Multi-GPU device routing |
| 4 | Dataset download via `gdown` (Google Drive) + extraction |
| 5 | Data loading, stratified split, DataLoaders |
| 6–10 | Model class definitions + shape assertions |
| 11 | ResNet18 baseline definition |
| 12 | Loss, optimizer, `train_model` |
| 13 | TTA prediction functions |
| 14 | Sample grid visualisation |
| 15 | Optuna search (commented out — use stored best params) |
| 16 | Phase 1 training (15 epochs, CE only) |
| 17 | Phase 2 training (40 epochs, CE + physics loss) |
| 18 | ResNet18 baseline training |
| 19 | Equivariance error verification |
| 20 | AUC, confusion matrix, ROC curves |

### 3. Run the Control Experiment

Open `vanilla-notebook.ipynb`. Runs `VanillaLensPINN` (identical pipeline, non-equivariant U-Net) with the same data split and training protocol.

### 4. Run the MI Experiment

Open `d4-PINN-MI notebook.ipynb`. Requires trained checkpoints from Steps 2–3 mounted as Kaggle datasets at:
- `PINN_CKPT = /kaggle/input/datasets/Aditya26189/d4-pinn-and-resnet/d4_phase2_best.pth`
- `RESNET_CKPT = /kaggle/input/datasets/Aditya26189/d4-pinn-and-resnet/resnet18_baseline_best.pth`

Run the pre-flight checks (Cells 1–2) before executing the main MI loop.

---

## Physics Background

### Why D4?

The gravitational lensing equation **β = θ − α(θ)** is symmetric under:
- **Rotation** — the lens mass distribution is statistically isotropic
- **Parity reflection** — the lensing equation is unchanged by mirroring

D4 (dihedral group of order 8) encodes both. C8 (8 rotations, no flip) would encode rotation invariance but **not** the parity symmetry, making it physically incorrect for this task.

### Physics Loss Terms

The 4-term physics loss penalises κ̂ for violating physical priors:

| Term | Physical Prior |
|:---|:---|
| Total Variation | Convergence maps should be spatially smooth |
| L1 sparsity | Subhalos are compact — κ̂ should be sparse |
| Centre penalty `κ̂·r²` | Mass should concentrate at the lens centre |
| Poisson residual `(∇²Ψ − 2κ̂)²` | Self-consistency with the lensing potential |

---

## Notebook Environment

These notebooks were developed and executed on **Kaggle** with:
- GPU: NVIDIA Tesla T4 (16 GB VRAM) or P100
- Python: 3.10
- PyTorch: 2.x
- CUDA: 11.x / 12.x
- numpy: 1.26.4 (pinned)

Dataset is hosted on Google Drive and downloaded via `gdown` at runtime.
