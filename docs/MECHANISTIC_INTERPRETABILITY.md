# Mechanistic Interpretability Reference — D4-LensPINN v4

## Overview

The MI experiment answers: **Does D4 equivariance causally alter how the network processes orientation information?**

It runs on both `D4LensPINN` (the equivariant model) and `VanillaLensPINN` (non-equivariant control), comparing internal representations using four complementary methods:

1. **Activation Patching / Causal Intervention** — inject activations from group-transformed inputs; measure logit Δ-L2
2. **Linear Probing** — train logistic classifiers on activations to detect group-element information (7-class, chance=0.143)
3. **Within-Orbit Cosine Distance Analysis** — measure within-orbit cosine distance at H08/H11/H16; MDS scatter + per-class bar (implemented in notebook Cell RSA). Note: `figure_rsa.png`, `repr_dist_results.csv`
4. **Polytope Geometry (D8 Orbit Volume)** — compute convex hull volume of the 8-point D4 orbit in activation space per hook; `polytope_geometry.csv`

**Key findings from these methods:**
- **Chain Property Failure:** Deterministic eval-mode BatchNorm causes structurally identical logit deltas across head hooks — head-level routing/collapse is INDETERMINATE (Mueller et al. 2024)
- **Probe-Delta Dissociation at H09:** High probe decodability (0.492) contradicts very low causal sensitivity (Δ-L2 ≈ 0.40) at the same layer

---

## Experimental Setup

**MI Subset:** 200 balanced test images (67 + 67 + 66 per class), sampled with `SEED=42`.

```python
counts = {0: 67, 1: 67, 2: 66}
for cls, n in counts.items():
    cls_idx = np.where(labels_all == cls)[0]
    subset_indices.extend(np.random.choice(cls_idx, n, replace=False))
```

**D4 Group Elements:**
```python
GROUP_ELEMS = ['e', 'r90', 'r180', 'r270', 'flip_h', 'flip_h_r90', 'flip_h_r180', 'flip_h_r270']

D4_TRANSFORMS = {
    'e':           lambda x: x.contiguous(),
    'r90':         lambda x: torch.rot90(x, k=1, dims=[2,3]).contiguous(),
    'r180':        lambda x: torch.rot90(x, k=2, dims=[2,3]).contiguous(),
    'r270':        lambda x: torch.rot90(x, k=3, dims=[2,3]).contiguous(),
    'flip_h':      lambda x: torch.flip(x, dims=[3]).contiguous(),
    'flip_h_r90':  lambda x: torch.rot90(torch.flip(x, dims=[3]), k=1, dims=[2,3]).contiguous(),
    # ...
}
```

> ⚠️ `.contiguous()` is required on all transforms (Bug C fix) — `torch.rot90` returns a non-contiguous view that crashes downstream conv ops.

---

## Hook Architecture

Both notebooks register forward hooks at 17 specific layers, spanning the entire computation graph from raw input to final pre-classification representation.

### D4LensPINN Hooks (`d4-PINN-MI notebook.ipynb`)

| Hook ID | Layer | Module Path | Stage |
|:---:|:---|:---|:---:|
| H00 | Physics Preprocess | `pinn_model.preprocess` | 0 |
| H01 | D4UNet Enc1 | `pinn_model.d4unet.enc1` | 1 |
| H02 | D4UNet Enc2 | `pinn_model.d4unet.enc2` | 1 |
| H03 | D4UNet Enc3 | `pinn_model.d4unet.enc3` | 1 |
| H04 | D4UNet Bottleneck | `pinn_model.d4unet.bot` | 1 |
| H08 | κ̂ output | `pinn_model.d4unet.kappa_out` | 1 |
| H09 | Poisson solver | `pinn_model.poisson` | 2 |
| H10 | Deflection field | `pinn_model.deflection` | 2 |
| H11 | Inverse lens | `pinn_model.inv_lens` | 2 |
| H12 | Handoff probe | `pinn_model.handoff_probe` | 2→3 |
| H13 | EfficientNetV2 block 3 | `pinn_model.classifier.features[3]` | 3 |
| H13b | EfficientNetV2 block 4 | `pinn_model.classifier.features[4]` | 3 |
| H14 | EfficientNetV2 block 5 | `pinn_model.classifier.features[5]` | 3 |
| H14b | EfficientNetV2 block 6 | `pinn_model.classifier.features[6]` | 3 |
| H15 | EfficientNetV2 block 7 | `pinn_model.classifier.features[7]` | 3 |
| H16 | After Global AvgPool | `pinn_model.classifier.avgpool` | 3 |
| H17 | Pre-linear (Dropout) | `pinn_model.classifier.head[0]` | 3 |

### VanillaLensPINN Hooks (`vanilla-notebook.ipynb`)

Identical hook IDs to D4 model. `d4unet` attribute refers to `VanillaUNet` (plain CNN, not equivariant).

```python
VANILLA_HOOKS = {
    'H00_preprocess':  vanilla_model.preprocess,
    'H01_enc1':        vanilla_model.d4unet.enc1,
    # ... same structure ...
    'H08_kappa_out':   vanilla_model.d4unet.kappa_out,
    'H09_poisson':     vanilla_model.poisson,
    # ... through H17_pre_linear
}
```

---

## Core MI Functions

### `cache_activations(model, x, hook_specs, device)`

**File:** Both MI notebooks

Performs a single forward pass and captures activations at all specified hooks.

```python
def cache_activations(model, x, hook_specs, device):
    """Returns (logits, cache_dict)."""
    cache = {}
    hooks = []

    for name, module in hook_specs.items():
        def _make_hook(n):
            def _hook(mod, inp, output):
                if isinstance(output, tuple):
                    cache[n] = tuple(t.detach().clone() for t in output)
                    cache[n + '__is_tuple'] = True
                elif isinstance(output, GeometricTensor):
                    cache[n] = output.tensor.detach().clone()
                    cache[n + '__gtype'] = copy.deepcopy(output.type)
                else:
                    cache[n] = output.detach().clone()
            return _hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

    for h in hooks:
        h.remove()   # always clean up hooks
    return logits.cpu(), cache
```

**Cache metadata keys:**
- `cache['H09_poisson']` — activation tensor
- `cache['H09_poisson__is_tuple']` — True if output was a tuple (e.g., InverseLensLayer returning `(S_hat, R)`)
- `cache['H01_enc1__gtype']` — escnn `FieldType` for equivariant tensor reconstruction

### `intervention_pass(model, x_clean, cache_gx, target_hook_name, hook_specs, device)`

**File:** Both MI notebooks

Performs a **causal intervention**: injects the cached activation from a group-transformed input `g·x` at a specific hook, then runs the rest of the computation graph on the original clean input `x`.

```
x_clean  ──────────────────────► [hook: INJECTED with act(g·x)] ──► ... ──► logits_patched
                                     ↑
g·x   → cache_activations() → cache_gx[target_hook_name]
```

**Implementation detail — `injected` flag:**
```python
injected = [False]   # mutable list used as closure variable

def _inject_hook(mod, inp, output):
    if injected[0]:
        return output       # on subsequent calls, pass through
    injected[0] = True
    # ... return cached activation ...
```
This prevents the hook from firing on downstream recursive module calls within the same forward pass.

**Measuring causal effect:**
```python
delta_l2 = (logits_patched - logits_clean).norm(p=2).item()
```
A large `delta_l2` means the activation at that hook carries causally relevant orientation information.

---

## Sanity Checks

Three sanity checks must pass before running the full MI loop.

### SC1: Identity Patch (delta < 1e-3)

Inject `cache_x` (activations from `x` itself) back into `x`. Logits must be unchanged:
```python
logit_clean, cache_x = cache_activations(model, s0['img'], HOOKS, DEVICE)
for h in HOOKS[:5]:
    lp = intervention_pass(model, s0['img'], cache_x, h, HOOKS, DEVICE)
    delta = (lp - logit_clean).norm(p=2).item()
    assert delta < 1e-3, f"Identity check FAILED at {h}"
```

### SC3: Non-Uniform Delta Profile (CV > 0.1)

Inject `cache_r90` (activations from r90-rotated image) into `x_clean`. Deltas across hooks must be **non-uniform** — if all hooks produce the same delta, the causal graph is contaminated:
```python
deltas = [intervention_pass(model, x, cache_r90, h, HOOKS, DEVICE) ... for h in HOOKS]
cv = std(deltas) / (mean(deltas) + 1e-8)
assert cv > 0.1
```

### SC5: VanillaUNet κ̂ Non-Invariance

The VanillaUNet's κ̂ MUST NOT be D4-equivariant (it's a standard CNN). This confirms the model is a valid control:
```python
kappa_x  = vanilla_model.d4unet(preprocess(x))
kappa_gx = vanilla_model.d4unet(preprocess(rot90(x)))
g_kappa_x = rot90(kappa_x)  # what an equivariant model would produce
diff = (kappa_gx - g_kappa_x).abs().mean().item()
assert diff > 1e-3, "VanillaUNet MUST be non-equivariant for this control"
```

---

## Linear Probing

**Goal:** How much **group-element information** (which of the 7 non-identity D4 transforms was applied) is recoverable from each hook's activation via a linear classifier?

**Method:** For each hook, run 5-fold stratified cross-validation with `LogisticRegression(C=0.1)`:

```python
N_PCA = 100   # reduce dimensionality before probing

for h in HOOKS:
    X = np.stack(all_feats[h])         # (N_samples, D)
    if X.shape[1] > N_PCA:
        X = PCA(n_components=N_PCA).fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = [accuracy_score(labels[te],
            LogisticRegression(C=0.1, max_iter=500)
            .fit(X[tr], labels[tr]).predict(X[te]))
            for tr, te in cv.split(X, labels)]
    print(f"{h}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
```

**Chance level:** 1/7 ≈ 0.143 (7 non-identity group elements)

**Interpretation:**
- High probe accuracy at H01–H04 (D4 encoder): expected — equivariant features encode group structure by construction
- High probe accuracy at H09 (Poisson): group information transmitted through physics engine
- Low probe accuracy at H16 (after GAP): Global Average Pooling collapses spatial structure → group info destroyed

---

## Results Summary

| Model | AUC | Computed Verdict (GAP Ratio) | Head Verdict | Mechanism |
|:---|:---:|:---:|:---:|:---|
| D4LensPINN | **0.9786** | ROUTING (runtime) | INDETERMINATE | Physics pipeline routes orientation through InverseLensLayer (46× amplification); head-level verdict overridden by chain property failure |
| VanillaLensPINN | 0.9776 | AMBIGUOUS (runtime) | INDETERMINATE | Non-equivariant κ̂ fails to structure representations; probe at H09 near-chance (0.154) |
| ResNet18 | 0.9182 | COLLAPSE (runtime) | INDETERMINATE | GAP destroys all spatial info; all head hooks show chain property failure (flat Δ-L2) |

**Verdict explanation:** The `ROUTING`/`COLLAPSE`/`AMBIGUOUS` verdict is computed from the GAP-ratio vote in notebook Cell 11. `INDETERMINATE` is the notebook’s explicit override (`verdict_head = 'INDETERMINATE (chain property)'`) applied to all models because frozen eval-mode BatchNorm makes head-level transitions non-informative (lines 2444–2446).

**AUC note:** D4LensPINN (0.9786) and VanillaLensPINN (0.9776) differ by only 0.001 — within noise. The mechanistic contribution (equivariance circuit routing) is independent of the raw AUC delta.

---

## Output Files

### `d4-PINN-MI results/`

| File | Description |
|:---|:---|
| `mi_results_full.csv` | Full intervention results: `(model, img_idx, true_class, group, hook, delta_l2, act_diff_sq)` |
| `ablation_results.csv` | Ablation study results across hook subsets |
| `act_cache_meta.csv` | Metadata for activation cache entries |
| `probe_results.csv` | Linear probe cross-validation summary per hook |
| `probe_class_results.csv` | Per-class breakdown of probe accuracy |
| `probe_d4_h09_h12_pca100.csv` | D4 probe at H09 and H12 with PCA-100 reduction |
| `probe_pca_equalized.csv` | PCA-equalized probe results for fair comparison |
| `mechanistic_probe_comparison.csv` | D4 vs Vanilla probe accuracy comparison |
| `hook_stats_by_hook.csv` | Summary statistics of delta_l2 grouped by hook |
| `hook_stats_table.csv` | Full hook statistics table |
| `wilcoxon_transitions.csv` | Wilcoxon signed-rank tests for consecutive hook transitions |
| `polytope_geometry.csv` | D8 orbit volume and mean rank per hook |
| `repr_dist_results.csv` | Within-orbit cosine distances (N_hooks × 200 images) |
| `spectral_band_analysis.csv` | Spectral analysis of activation frequency content |
| `tta_bootstrap_ci.json` | Bootstrap confidence intervals for TTA AUC |
| `findings_summary.json` | Top-level MI findings in machine-readable format |
| `mi_subset_indices.json` | Exact indices of the 200-image MI subset |
| `figure1.png` | Main MI figure: delta_l2 profiles across hooks |
| `figure2.png` | Within-orbit cosine distance: MDS scatter + per-class bar |
| `figure4.png` | Probe accuracy vs hook depth |
| `figure5.png` | Class-selective neuron analysis |
| `figure_rsa.png` | Full within-orbit geometry plot (H08/H11/H16) |
| `figure_family_split.png` | Class separation by group-element family |
| `results_partial_D4LensPINN_*.csv` | Incremental save checkpoints (every 5 samples) |
| `results_partial_ResNet18_*.csv` | Incremental save checkpoints for baseline |
| `probs_notta.npy` | Raw probability outputs without TTA |
| `probs_tta.npy` | Raw probability outputs with D4-TTA |

### `vanilla results/`

| File | Description |
|:---|:---|
| `mi_results_full.csv` | Full intervention results for VanillaLensPINN |
| `vanilla_probe_cv_per_fold.csv` | Per-fold cross-validation scores |
| `vanilla_probe_summary.csv` | Summary probe accuracy at H09 and H12 |
| `vanilla_figure1.png` | Delta_l2 profile for VanillaLensPINN |

---

## Pre-flight Checklist (`d4-PINN-MI notebook.ipynb`)

Before running the main MI loop, cells 1–2 verify:

- [ ] `pinn_model.preprocess` → `PhysicsPreprocess`
- [ ] d4unet attribute names: `enc1, enc2, enc3, bot, dec3, dec2, dec1, kappa_out`
- [ ] `pinn_model.handoff_probe` exists (nn.Identity())
- [ ] `pinn_model.classifier.features[N]` — no `.backbone.` prefix
- [ ] `pinn_model.classifier.avgpool` / `.head[0]`
- [ ] `resnet_model.net.layer1` — `.net.` prefix required
- [ ] EfficientNetV2 features block count ≥ 8
- [ ] `d4unet` forward order: `d4unet` appears before `poisson` in source
- [ ] Checkpoint `d4_phase2_best.pth` loads without key errors
