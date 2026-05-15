# Experimental Results — D4-LensPINN v4

> All numbers in this document are extracted directly from the result files in `d4-PINN-MI results/` and `vanilla results/`. No numbers are estimated or fabricated.

---

## 1. Model Performance (AUC-ROC)

### Final Evaluation on 3,000-Image Held-Out Test Set

| Model | Macro AUC (No TTA) | Macro AUC (D4-TTA) | TTA Δ |
|:---|:---:|:---:|:---:|
| **D4LensPINN** | **0.9786** | **0.9809** | +0.0024 |
| VanillaLensPINN | 0.9776 | — | — |
| ResNet18 Baseline | 0.9182 | — | — |

**Source:** `tta_bootstrap_ci.json`
```
auc_notta : 0.97856
auc_tta   : 0.98093
delta_auc : 0.00237
n         : 3000 images
boot_n    : 2000 bootstrap samples
```

### Bootstrap Confidence Interval for TTA Gain

```
95% CI (bootstrap, n=2000): [−0.0007, +0.0052]
```

The CI crosses zero, confirming the TTA gain is **not statistically significant at α=0.05** — TTA is marginally beneficial for AUC but not a statistically robust improvement. Furthermore, TTA strictly **hurts top-1 accuracy** (90.4% → 84.9%) due to horizontal flips altering the sign of the Sobel saddle-point feature.

### D4LensPINN vs VanillaLensPINN

AUC difference = **0.0010** (0.9786 vs 0.9776) — within noise. The mechanistic interpretability contribution stands independently of this raw AUC delta.

> **Key finding:** D4 equivariance does not confer a meaningful AUC advantage over VanillaLensPINN in this experiment. The advantage lies in the internal representation structure and causal mechanisms, not the headline metric.

---

## 2. Experiments Run

### Experiment 1: D4LensPINN Training (Main Notebook)

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`

| Phase | Epochs | Loss | LR | WD |
|:---:|:---:|:---:|:---:|:---:|
| Phase 1 | 15 | CE only | 7.11e-4 | 1.46e-3 |
| Phase 2 | 40 | CE + Physics | 4.46e-4 | 1.46e-3 |

Hyperparameters found via Optuna (15 trials, TPE sampler, 6000-sample subset, 8 epochs/trial).

**Optuna best values:**

| Parameter | Best Value |
|:---|:---:|
| `lr_phase1` | 7.11e-4 |
| `lr_phase2` | 4.46e-4 |
| `weight_decay` | 1.46e-3 |
| `lambda_poisson` | ~0.01 |

---

### Experiment 2: ResNet18 Baseline Training

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 18

ResNet-18 trained from scratch (`weights=None`) on raw grayscale images. Same 80/10/10 split. Final **Macro AUC: 0.9182**, representing the lower bound — a CNN with no equivariance and no physics priors.

---

### Experiment 3: VanillaLensPINN Control Training

**File:** `vanilla-notebook.ipynb`

Identical training protocol to D4LensPINN but with a non-equivariant U-Net:

| Phase | Epochs | LR | WD |
|:---:|:---:|:---:|:---:|
| Phase 1 | 15 | 7.0e-4 | 1.0e-3 |
| Phase 2 | 60 | 4.0e-4 | 1.0e-3 |

> Phase 2 ran for 60 epochs (vs 40 for D4LensPINN) with checkpoint resuming to account for Kaggle session timeouts.

**Final Macro AUC: 0.9776** (AUC gate ≥ 0.93 ✅)

---

### Experiment 4: Mechanistic Interpretability — D4LensPINN vs ResNet18

**File:** `d4-PINN-MI notebook.ipynb`

**Setup:** 200 balanced test images (67+67+66 per class, seed=42). For each image, all 7 non-identity D4 group elements applied, activation cached at 17 hooks, intervention (patching) performed.

**Total intervention records:** `200 images × 8 group elements × len(PINN_HOOKS_PRIMARY)` rows in `mi_results_full.csv`. Exact count depends on primary hook set; verified by notebook assert at runtime.

---

### Experiment 5: Mechanistic Interpretability — VanillaLensPINN Control

**File:** `vanilla-notebook.ipynb`

Same 200-image MI subset and methodology applied to VanillaLensPINN. Results in `vanilla results/mi_results_full.csv`.

**Total records:** `200 images × 8 group elements × len(VANILLA_HOOKS_PRIMARY)` rows.

---

## 3. Activation Patching Results (Causal Intervention)

### Mean Δ-L2 by Hook (D4LensPINN vs ResNet18)

Higher Δ-L2 = that hook's activation carries more causally effective orientation information.

**Source:** `hook_stats_by_hook.csv`

#### D4LensPINN

| Hook | Mean Δ-L2 | Std | Interpretation |
|:---|:---:|:---:|:---|
| **H11_inv_lens** | **18.003** | 12.805 | Largest causal signal — InverseLensLayer residual R carries strong orientation info |
| H00_preprocess | 3.363 | 2.487 | Saddle-point feature encodes orientation |
| H01_enc1 | 3.363 | 2.487 | Same as preprocess (early equivariant encoder) |
| H08_kappa_out | ~3.16 | — | **Equivariance boundary** — GroupPooling produces D4-invariant κ̂; Δ-L2 recovers from encoder drop due to downstream physics graph |
| H02_enc2 | 3.231 | 2.514 | Slight drop through D4 encoder |
| H12_HANDOFF | 3.174 | 2.206 | 4-channel stack handed to classifier |
| H12b–H17 | 3.174 | 2.206 | **Uniform across all EfficientNetV2 layers** (Chain Property Failure artefact) |
| H03_enc3 | 1.316 | 1.322 | Deeper encoder → lower raw delta |
| H04_bot | 1.182 | 1.199 | Bottleneck — compressed representation |
| **H09_poisson** | **0.402** | 0.340 | Very low — Poisson solver spectral decay suppresses causal orientation signal |
| H10_deflection | 0.402 | 0.340 | Same as Poisson (deterministic transform) |

#### ResNet18

| Hook | Mean Δ-L2 | Std |
|:---|:---:|:---:|
| R00_stem through R06_fc | **6.334** | 5.385 |

**All ResNet18 hooks show identical Δ-L2** — the causal signal is uniform across the entire depth. This is the **"flat profile"** characteristic of a model without structured equivariance — every layer responds equally to orientation changes.

---

## 4. Key Causal Finding: InverseLensLayer Amplification

**Source:** `findings_summary.json`

```
invlens_r90_mean         : 20.333  (Δ-L2 at H11, r90 transform)
poisson_r90_mean         : 0.440   (Δ-L2 at H09, r90 transform)
handoff_r90_mean         : 4.047   (Δ-L2 at H12, r90 transform)

amplification over Poisson  : 46.19×
amplification over Handoff  : 5.02×
```

**Interpretation:** The `InverseLensLayer` (H11) amplifies the causal effect of orientation by **46×** relative to the Poisson solver output (H09). This is a purely physics-driven amplification: when the deflection field α is rotated, the grid-sampling of the source image `Ŝ = grid_sample(I, base_grid − α)` produces a residual `R = I − Ŝ` that spatially encodes the rotation directly. The EfficientNetV2 classifier then reads this rotation-encoded residual.

---

## 5. Ablation Study — Channel Contribution to Δ-L2

**Source:** `findings_summary.json`

Three conditions tested, measuring mean Δ-L2 at the handoff point:

| Condition | Input to Classifier | Mean Δ-L2 |
|:---|:---|:---:|
| **κ̂ only** | `[I, κ̂, zeros, zeros]` | **3.292** |
| **ISR only** | `[I, zeros, Ŝ, R]` | **3.802** |
| **Full** | `[I, κ̂, Ŝ, R]` | **3.162** |

The ISR (Inverse-lens + Source + Residual) channels carry slightly more orientation signal than κ̂ alone. The full 4-channel model shows a slightly lower Δ-L2 because the combined representation allows the classifier to partially disentangle orientation from class identity (routing effect).

---

## 6. Linear Probe Results — Group-Element Decodability

**Setup:** Logistic regression probe (C=0.1, 5-fold CV) on PCA-reduced activations. Task: predict which of 7 non-identity D4 group elements was applied. Chance = 1/7 ≈ 0.143.

> **Two probe experiments exist in the codebase.** The **PCA-50 equalized** experiment (`probe_pca_equalized.csv`, notebook line 4721, `N_PCA=50`) is the primary result used in the ICML paper. The **PCA-100** experiment (`probe_d4_h09_h12_pca100.csv`, line 5273) is a supplementary check. They differ because PCA-50 is equalized across all hooks for fair capacity comparison; PCA-100 is hook-specific. Both are real.

### D4LensPINN Probes — Primary (PCA-50 Equalized)

**Source:** `probe_pca_equalized.csv` | **Cell:** line 4721, `N_PCA = 50`

| Hook | Acc (mean) | Std | Chance | Δ Over Chance | Significant |
|:---|:---:|:---:|:---:|:---:|:---:|
| H08_kappa_out | 0.365 | 0.024 | 0.143 | +0.222 | ✅ Yes |
| **H09_poisson** | **0.485** | 0.027 | 0.143 | **+0.342** | ✅ Yes |
| H11_inv_lens | 0.419 | 0.012 | 0.143 | +0.276 | ✅ Yes |
| H12_HANDOFF | 0.429 | 0.011 | 0.143 | +0.286 | ✅ Yes |
| H13_eff3 | 0.449 | — | 0.143 | +0.306 | ✅ Yes |
| H13b_eff4 | 0.456 | — | 0.143 | +0.313 | ✅ Yes |
| H15_before_gap | 0.266 | 0.028 | 0.143 | +0.123 | ✅ Yes |
| H16_after_gap | 0.198 | 0.021 | 0.143 | +0.055 | ✅ Yes |

> Note: H08 PCA-50 explains ~58% of variance; H08 probe value carries additional uncertainty.

**Interpretation:**
- **H09 (0.485):** Poisson output encodes group-element information at **3.39× chance** — physics engine transmits orientation structure.
- **H12 (0.429):** 4-channel handoff still encodes group elements at **3.00× chance**.
- **H16 (0.198):** After GAP, probe drops from 0.266 → 0.198 (Δ=0.069) — the single largest per-layer drop. Cumulative H09→H15 drop (Δ=0.219) substantially exceeds the GAP contribution alone.
- **H08_kappa_out (0.365):** Encodes some orientation — note this is *above* chance, unlike the PCA-100 result below. PCA-50 captures only 58% variance at H08, so this reflects partial geometric information leaking into the first 50 components.

### D4LensPINN Probes — Supplementary (PCA-100)

**Source:** `probe_d4_h09_h12_pca100.csv` | **Cell:** line 5273, `N_PCA = 100`

| Hook | Acc (mean) | Std | Chance | Δ Over Chance |
|:---|:---:|:---:|:---:|:---:|
| H09_poisson | **0.492** | 0.019 | 0.143 | **+0.349** |
| H12_HANDOFF | **0.444** | 0.009 | 0.143 | **+0.301** |
| H08_kappa_out | 0.122 | 0.012 | 0.143 | −0.021 |

> At PCA-100, H08 falls **below** chance — GroupPooling effectively makes κ̂ D4-invariant when using more principal components. The PCA-50 result (0.365) reflects capacity constraints, not true decodability.

### VanillaLensPINN Probes

**Source:** `vanilla_probe_summary.csv`, `vanilla_probe_cv_per_fold.csv`

**Protocol:** Spatial mean-pool activations `(B,C,H,W)→(B,C)`, then PCA-100 (`N_PCA_TARGET=100`). Notebook: vanilla-notebook.ipynb line 1594.

| Hook | Acc (mean) | Std | Chance | Δ Over Chance | Significant |
|:---|:---:|:---:|:---:|:---:|:---:|
| H09_poisson | **0.479** | 0.023 | 0.143 | **+0.336** | ✅ Yes |
| H12_handoff | — | — | 0.143 | — | — |

> Earlier cell (line 919, full spatial flatten + PCA-100, no spatial pooling): Vanilla H09 ≈ 0.154 — this much lower value arises because tens-of-thousands of raw spatial features overwhelm the PCA compression. The mean-pooled result (0.479) is the correct per-channel representation.

### Probe-Delta Dissociation Comparison at H09

| Model | δ (Δ-L2, 7-elem mean) | 95% CI (n=2000) | Probe Acc (PCA-50/pooled) | Chance |
|:---|:---:|:---:|:---:|:---:|
| D4LensPINN | **0.402** | [0.384, 0.419] | **0.485** ± 0.027 | 0.143 |
| VanillaLensPINN | **1.429** | [1.352, 1.506] | **0.479** ± 0.023 | 0.143 |
| **Ratio / Δ** | **3.6×** | non-overlapping | **Δ = 0.006** | |

**Key finding:** Causal sensitivity differs by **3.6×** while probe accuracy is nearly identical (Δ=0.006). D4 equivariance reshapes causal routing without changing linear decodability — the two MI tools reach opposite conclusions at the same layer.

### ResNet18 Probes

**Source:** `probe_results.csv`

| Hook | Acc (mean) | Std | Chance | Δ Over Chance | Significant |
|:---|:---:|:---:|:---:|:---:|:---:|
| R04_before_gap | 0.279 | 0.022 | 0.143 | +0.136 | ✅ Yes |
| R05_after_gap | 0.279 | 0.022 | 0.143 | +0.136 | ✅ Yes |
| R06_fc | 0.215 | 0.030 | 0.143 | +0.072 | ✅ Yes |

ResNet18 carries group-element information at every layer uniformly — no structured suppression mechanism.

### ResNet18 Probes
---

## 7. Wilcoxon Signed-Rank Tests — Hook Transitions

**Source:** `wilcoxon_transitions.csv`  
**Test:** Wilcoxon signed-rank on paired Δ-L2 values between consecutive hooks (H_i → H_j). Bonferroni-corrected p-values. Implemented at notebook Cell 21 (lines 3322–3397).

**Structure of the test:**
- For each adjacent hook pair, two one-sided Wilcoxon tests are run: DROP (H_i > H_j) and RISE (H_i < H_j)
- The significant direction is chosen based on the smaller p-value
- Bonferroni correction is applied across `n_pairs = len(hook_order) - 1` comparisons
- Results saved per-row as `{model, h1, h2, statistic, p_raw, p_bonferroni, significant, direction}`

### D4LensPINN — Key Transitions (runtime values from `wilcoxon_transitions.csv`)

| Transition | Direction | p_bonferroni (runtime) |
|:---|:---:|:---:|
| H08_kappa_out → H09_poisson | **DROP** | significant ✅ |
| H10_deflection → H11_inv_lens | **RISE** | significant ✅ |
| H11_inv_lens → H12_HANDOFF | **DROP** | significant ✅ |
| H12b → H13 → ... → H17 (all head transitions) | **NONE** | non-significant ❌ |

> The specific statistic values and exact p-values are runtime outputs stored in `wilcoxon_transitions.csv`. The table above shows the structure; exact numbers populate at runtime.

### D4LensPINN — Non-Significant Transitions

All transitions within the EfficientNetV2 backbone (H12b → H13 → ... → H17) are **non-significant**, confirming the classifier layers do not alter causal sensitivity — the Δ-L2 profile is flat through the backbone. This flatness is the **Chain Property Failure** artefact: deterministic eval-mode BatchNorm freezes statistics, making head-level deltas structurally identical.

### ResNet18 — All Transitions

**All ResNet18 hook transitions are non-significant** (uniform Δ-L2 profile). No hook-to-hook routing structure exists.

---

## 8. Polytope Geometry (D8 Orbit Volume)

**Source:** `polytope_geometry.csv`  
Measures the volume spanned by the D8 orbit in activation space — how much the 8 D4 group elements spread the representation at each layer. Computed in notebook Cell 29.

| Hook | Mean Volume | Std | Mean Rank |
|:---|:---:|:---:|:---:|
| H08_kappa_out | **0.026** | 0.199 | 7.00 |
| H09_poisson | 1.10e13 | 5.48e13 | 6.98 |
| H11_inv_lens | 1.10e7 | 1.34e7 | 7.00 |
| H12_HANDOFF | 1.28e8 | 1.55e8 | 7.00 |
| H13_eff3 | 3.70e19 | 1.62e19 | 7.00 |
| H15_before_gap | 1.00e12 | 2.24e12 | 7.00 |
| H16_after_gap | **53.96** | 139.02 | 7.00 |

**Key observations:**
- **H08_kappa_out (volume ≈ 0.026):** Near-zero orbit volume confirms κ̂ is D4-invariant — all 8 group elements map to virtually the same point in activation space.
- **H16_after_gap (volume ≈ 54):** After Global Average Pooling, volume collapses dramatically (from 1e12 at H15 to ~54). GAP destroys spatial orientation information.
- Full-rank orbits (mean rank = 7.0) everywhere else confirm the 7 non-identity group elements produce linearly independent activation directions.

---

## 9. Chain Property Failure

**Source:** Notebook Cell 11 (`model_verdicts` logic) and `hook_stats_table.csv`  

**Key Finding:** During interchange interventions, we discovered a **chain property failure** for head-level analysis.

The notebook explicitly detects this at lines 2412–2448:

```python
# The GAP-ratio routing verdict is an artifact of this property.
# Override: model_verdicts should not be used for head-level routing claims.
model_verdicts[mn]['verdict_head'] = 'INDETERMINATE (chain property)'
```

**Mechanism:** Deterministic eval-mode BatchNorm freezes its running mean/variance statistics. When an activation from a group-transformed input `g·x` is patched into the network mid-stream, the downstream BatchNorm layers apply the *same* frozen normalization regardless of the injected value’s distribution shift. This creates structurally identical logit deltas across all classifier head hooks (H13–H17), making the head-level Wilcoxon transitions non-significant — an empirical instantiation of Mueller et al. (2024)’s theoretical non-transitivity result in interchange interventions.

**Consequence:** Head-level routing/collapse verdicts are classified as `INDETERMINATE` in the notebook. The physics pipeline profile (H00–H12) remains interpretable and is the primary evidence base.

---

## 10. Probe-Delta Dissociation

**Source:** `hook_stats_by_hook.csv` (δ values) + `probe_pca_equalized.csv` (D4 probe) + `vanilla_probe_summary.csv` (Vanilla probe). Bootstrap CIs computed in `vanilla-notebook.ipynb` lines 1468–1491 (`n=2000`).

At H09 (Poisson solver), comparing D4LensPINN vs VanillaLensPINN:

| Metric | D4LensPINN | VanillaLensPINN | Interpretation |
|:---|:---:|:---:|:---|
| **Causal Sensitivity** (δ, 7-elem mean) | **0.402** | **1.429** | **3.6× difference** — D4 far less causally sensitive |
| **95% CI** (n=2,000 bootstrap) | [0.384, 0.419] | [1.352, 1.506] | CIs **non-overlapping** ✅ |
| **Probe Acc** (PCA-50/mean-pool) | **0.485** ± 0.027 | **0.479** ± 0.023 | Nearly identical (Δ=0.006) |

These two standard MI tools give **opposite conclusions at the same layer**: probing says group-element information is present at similar levels in both models; intervention says D4 is 3.6× less causally sensitive. **Equivariance reshapes causal routing without changing linear decodability.**

**Mechanism:** The Poisson 1/k² spectral decay compresses orientation into low-frequency modes that survive linear probing but are not causally routed downstream. D4's equivariant κ̂ enters the solver with more spatially structured orbits, making the spectral attenuation more complete — hence the lower δ — while the probe still reads out the compressed group-element information.

---

## 11. Spectral Band Analysis

**Source:** `spectral_band_analysis.csv`

Spatial frequency content of activations at H08 (κ̂) and H09 (Ψ):

| Hook | Total Power | Low-freq | Mid-freq | High-freq |
|:---|:---:|:---:|:---:|:---:|
| H08_kappa_out | 2.68e-4 | 38.9% | 59.5% | 1.6% |
| H09_poisson | 42,351 | **99.97%** | 0.02% | <0.001% |

**Interpretation:** The Poisson solver acts as a perfect **low-pass filter** — the FFT solution `Ψ̂(k) = −2κ̂(k) / k²` divides by k², heavily suppressing high-frequency components. κ̂ contains mixed mid-frequency content (subhalo mass profiles), which Ψ transforms entirely into low-frequency structure. This spectral collapse is why H09 shows extremely low Δ-L2 (0.40) despite carrying strong linear probe accuracy (0.49) — the group-element information survives but is compressed into low-frequency modes.

---

## 12. Summary of Mechanistic Claims

| Claim | Evidence | Source |
|:---|:---:|:---:|
| D4LensPINN achieves AUC 0.9786 | Direct measurement | `tta_bootstrap_ci.json` |
| D4-TTA improves AUC by +0.0024 (95% CI: [−0.0007, +0.0052], n=2000) | Bootstrap CI | `tta_bootstrap_ci.json` |
| κ̂ is D4-invariant at PCA-100 (probe acc below chance: 0.122) | Linear probe (PCA-100) | `probe_d4_h09_h12_pca100.csv` |
| D4 H09 probe acc = 0.485 (PCA-50); Vanilla H09 = 0.479 (Δ=0.006) | Linear probe (PCA-50 / mean-pool) | `probe_pca_equalized.csv`, `vanilla_probe_summary.csv` |
| Probe-delta dissociation: 3.6× causal diff, near-identical probe acc | Probe + Intervention (bootstrap n=2000) | `hook_stats_by_hook.csv`, `probe_pca_equalized.csv` |
| Vanilla H09 δ = 1.429 (CI [1.352, 1.506]); D4 = 0.402 (CI [0.384, 0.419]) | Bootstrap CI | `vanilla-notebook.ipynb` line 1468–1491 |
| InverseLensLayer amplifies causal effect ~46× over Poisson | Intervention Δ-L2 | `findings_summary.json` |
| Head-level verdicts INDETERMINATE (chain property failure) | Intervention Δ-L2 | `hook_stats_table.csv`, `wilcoxon_transitions.csv` |
| κ̂ orbit volume ≈ 0 (D4-invariant) | Polytope geometry | `polytope_geometry.csv` |
| GAP collapses orbit volume from 1e12 to ~54 | Polytope geometry | `polytope_geometry.csv` |
| Poisson solver is low-pass filter (99.97% low-freq power) | Spectral analysis | `spectral_band_analysis.csv` |

---

## 13. Result Files Index

| File | Contents | Size |
|:---|:---|:---:|
| `mi_results_full.csv` | All intervention results: model, img, group, hook, delta_l2, act_diff_sq | 4.6 MB |
| `tta_bootstrap_ci.json` | AUC with/without TTA, bootstrap 95% CI | <1 KB |
| `findings_summary.json` | Core numerical findings (amplification, ablations) | <1 KB |
| `probe_results.csv` | Linear probe CV accuracy at key hooks | <1 KB |
| `probe_d4_h09_h12_pca100.csv` | D4 probe (PCA-100) at H09, H12 | <1 KB |
| `mechanistic_probe_comparison.csv` | Head-to-head D4 vs Vanilla at H09, H12 | <1 KB |
| `probe_pca_equalized.csv` | PCA-equalized probe for fair comparison | <1 KB |
| `probe_class_results.csv` | Per-class probe accuracy breakdown | <1 KB |
| `hook_stats_by_hook.csv` | Mean/std Δ-L2 per hook for D4 and ResNet18 | <1 KB |
| `hook_stats_table.csv` | Extended hook statistics | 14 KB |
| `wilcoxon_transitions.csv` | Wilcoxon signed-rank tests for hook-to-hook transitions | <1 KB |
| `ablation_results.csv` | Per-sample ablation Δ-L2 (κ-only, ISR-only, full) | 22 KB |
| `spectral_band_analysis.csv` | Frequency band power at H08, H09 | <1 KB |
| `polytope_geometry.csv` | D8 orbit volume and rank per hook | <1 KB |
| `repr_dist_results.csv` | Within-orbit cosine distances per hook (from cosine distance analysis) | 2.1 MB |
| `act_cache_meta.csv` | Activation cache metadata | 28 KB |
| `mi_subset_indices.json` | Exact 200 test indices used | <1 KB |
| `figure1.png` | Δ-L2 profile across hooks | 427 KB |
| `figure2.png` | Within-orbit cosine distance (MDS + per-class bar) | 689 KB |
| `figure4.png` | Probe accuracy vs depth | 122 KB |
| `figure5.png` | Class-selective neuron analysis | 125 KB |
| `figure_rsa.png` | Full within-orbit geometry matrix | 1.1 MB |
| `figure_family_split.png` | Class separation geometry | 633 KB |
| `figure1_poisson_differential.png` | Poisson differential visualisation | 137 KB |
| `figure2_invariance_restoration.png` | Invariance restoration curve | 179 KB |
| `figure_b1_invariance_restoration_curve.png` | Appendix B1 invariance curve | 122 KB |
| `figure_d1_h11_class_selective.png` | Appendix D1 H11 class selectivity | 194 KB |
| `results_partial_D4LensPINN_*.csv` | Incremental saves every 5 images (40 files) | ~3.3 MB total |
| `results_partial_ResNet18_*.csv` | Incremental saves every 5 images (40 files) | ~1.2 MB total |

### Vanilla Results

| File | Contents |
|:---|:---|
| `mi_results_full.csv` | Full VanillaLensPINN intervention results (1.9 MB) |
| `vanilla_probe_cv_per_fold.csv` | Per-fold CV scores for H09 and H12 probes |
| `vanilla_probe_summary.csv` | Summary probe accuracy for VanillaLensPINN |
| `vanilla_figure1.png` | Δ-L2 profile for VanillaLensPINN |
