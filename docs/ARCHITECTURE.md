# Architecture Reference — D4-LensPINN v4

## Overview

D4-LensPINN is a **4-stage hybrid model** combining group-theoretic equivariant networks with a fully differentiable physical optics simulation. The key insight is that gravitational lensing obeys the **D4 symmetry group** (4 rotations × 1 reflection = 8 group elements), and so the convergence map estimator is built to be D4-equivariant by construction.

```
Input I : (B, 1, 150, 150)  ← grayscale .npy images
    │
    ▼  STAGE 0 — PhysicsPreprocess                  [0 params]
    │   Saddle-point physics feature extraction
    │   OUT: (B, 2, 150, 150)
    │
    ▼  STAGE 1 — EfficientD4UNet                    [0.46M params]
    │   D4-equivariant U-Net estimating κ̂ ≥ 0
    │   OUT: κ̂ (B, 1, 150, 150)
    │
    ▼  STAGE 2 — DifferentiablePhysicsEngine        [0 params]
    │   PoissonSolverFFT  → Ψ
    │   DeflectionField   → α = ∇Ψ
    │   InverseLensLayer  → Ŝ = unlensed source, R = I − Ŝ
    │   OUT: Ŝ (B, 1, 150, 150),  R (B, 1, 150, 150)
    │
    ▼  STAGE 3 — EfficientNetV2Head                 [19.87M trainable]
    │   X_cls = cat([I, κ̂, Ŝ, R])   (B, 4, H, W)
    │   1×1 conv projection → ImageNet norm → EfficientNetV2-S
    │   OUT: logits (B, 3)
    │
    ▼  OUTPUT
       logits (B, 3) + κ̂ (B, 1, 150, 150)
```

**Total model parameters:** 20.65M total / 20.33M trainable

---

## Stage 0: `PhysicsPreprocess`

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 6

### Motivation

Gravitational lensing deflects photon paths along saddle directions of the lensing potential Ψ. The cross-product of log-intensity gradients is:
- **Large** at saddle points → characteristic of perturbed lenses (sphere/vort classes)
- **Near-zero** on smooth uniform backgrounds → characteristic of `no` class

This provides a **parameter-free, physics-motivated second channel** invisible to vanilla CNNs.

### Implementation

```python
class PhysicsPreprocess(nn.Module):
    def __init__(self):
        # Sobel kernels stored as buffers (auto-moves to GPU with .to(device))
        kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]   # horizontal gradient
        ky = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]   # vertical gradient

    def forward(self, I):
        I_max  = I.amax(dim=(-2,-1), keepdim=True).clamp(min=1e-6)
        I_log  = log(I_max / (I + 1e-6))   # log-ratio: highlights faint arcs
        gx     = Sobel_x(I_log)             # ∂_x log(I)
        gy     = Sobel_y(I_log)             # ∂_y log(I)
        P      = gx * gy                    # raw saddle-point product (no tanh!)
        P      = P / std(P)                 # per-image normalisation
        return cat([I, P], dim=1)           # (B, 2, H, W)
```

### Design Decisions

| Decision | Rationale |
|:---|:---|
| **No `tanh` on P** | Raw magnitude preserves stronger saddle signatures of subhalo lenses |
| **Per-image std normalisation** | Makes feature scale-invariant across exposure levels |
| **`register_buffer` for Sobel kernels** | Kernels move to GPU automatically when model moves |

**Output shape assertion:** `(B, 2, 150, 150)` ✅

---

## Stage 1: `EfficientD4UNet`

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 8

### D4 Group Equivariance

The D4 group has 8 elements: `{e, r90, r180, r270, flip, flip·r90, flip·r180, flip·r270}`.

A function `f` is D4-equivariant if: **`f(g·x) = g·f(x)` for all g ∈ D4**

The lensing equation `β = θ − α(θ)` is symmetric under both rotation **and** parity reflection — D4 (`flipRot2dOnR2(N=4)`) encodes this directly. C8 (8 rotations, no flip) would NOT encode the reflection symmetry of gravitational lensing.

### Encoder–Decoder Architecture

| Stage | Spatial | Equivariant Channels | Notes |
|:---:|:---:|:---:|:---|
| enc1 (R2Conv 7×7, stride=2) | 150→75 | regular×8 (64 ch) | Initial feature extraction |
| enc2 (D4ResBlock, stride=2) | 75→38  | regular×16 (128 ch) | — |
| enc3 (D4ResBlock, stride=2) | 38→19  | regular×32 (256 ch) | — |
| bottleneck (D4ResBlock, stride=1) | 19→19 | regular×48 (384 ch) | — |
| dec3 (cat + R2Conv) | 19→19 | regular×32 | Skip from enc3 |
| dec2 (upsample + cat + R2Conv) | 19→38 | regular×16 | Skip from enc2 |
| dec1 (upsample + cat + R2Conv) | 38→75 | regular×8  | Skip from enc1 |
| out (GroupPool + Conv1×1 + Softplus) | 75→150 | scalar (1 ch) | κ̂ ≥ 0 enforced |

### `D4ResBlock`

A standard residual block using equivariant `escnn` layers:
```
R2Conv(stride) → InnerBatchNorm → ReLU → R2Conv → InnerBatchNorm
      + shortcut projection (1×1 R2Conv if stride≠1 or channels change)
→ ReLU
```

### `GroupPooling` Output Gate

`GroupPooling` collapses the equivariant feature representation to D4-**invariant** scalars while preserving full spatial resolution. This makes κ̂ a rotation/reflection-independent scalar field — which is physically correct since convergence is an observable quantity.

`Softplus` enforces κ̂ ≥ 0 everywhere (convergence is positive-definite by definition).

### Critical Bug Fixes

**Bug 1: `gs` as instance variable (not global)**
```python
# CORRECT — each model gets its own isolated escnn cache
class EfficientD4UNet(nn.Module):
    def __init__(self):
        gs = gspaces.flipRot2dOnR2(N=4)  # INSTANCE variable

# WRONG — all instances share one cache, corrupted by GPU moves
gs = gspaces.flipRot2dOnR2(N=4)  # global variable
```

**Bug 2: Override `_apply`, not `to()`**
```python
def _apply(self, fn):
    # PyTorch calls _apply() recursively during .to(device)
    # Overriding to() only fires on DIRECT calls, bypassed when nested
    super()._apply(fn)
    for mod in self.modules():
        if hasattr(mod, 'sampled_basis') and isinstance(mod.sampled_basis, torch.Tensor):
            mod.sampled_basis = fn(mod.sampled_basis)  # migrate escnn basis tensors
    return self
```

**Total params:** 0.46M ✅

---

## Stage 2: Differentiable Physics Engine

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 7

This stage implements the gravitational lensing equations as a **parameter-free, fully differentiable** PyTorch module. Gradients flow through the entire physics simulation back to the U-Net.

### `PoissonSolverFFT` — ∇²Ψ = 2κ̂

Solves the lensing potential Ψ from the convergence κ̂ in the frequency domain:

```
κ̂  →  zero-pad to 256×256  →  rfft2
   →  Ψ̂(k) = −2·κ̂(k) / k²    (spectral Green's function)
   →  irfft2  →  crop to 150×150
```

| Design Decision | Reason |
|:---|:---|
| Pad to next power-of-2 (256) | Avoids circular FFT wraparound artifacts at image edges |
| `k²[0,0] = 1.0` (DC gauge) | Prevents division-by-zero at zero frequency |
| `Ψ̂(0,0) = 0` | Sets zero-mean gauge for the potential (arbitrary constant removed) |
| All frequency grids as `register_buffer` | Auto-GPU migration with `.to(device)` |

### `DeflectionField` — α = ∇Ψ

Central finite differences with periodic wrap (via `torch.roll`):
```
αx = (Ψ[x+1] − Ψ[x−1]) / 2
αy = (Ψ[y+1] − Ψ[y−1]) / 2
OUT: α = cat([αx, αy], dim=1)   → (B, 2, H, W)
```

### `InverseLensLayer` — β = θ − α(θ)

Applies the inverse lens mapping to recover the unlensed source:

```
αx, αy   →  normalise to [-1,1]
         →  clamp to ±0.95  (prevents grid_sample out-of-bounds)
grid     = base_grid − delta
Ŝ        = grid_sample(I, grid, bilinear, zeros, align_corners=False)
R        = I − Ŝ   (lensing residual)
```

The `±0.95` clamp prevents `F.grid_sample` from sampling outside the image boundaries, which would produce zero-padding artifacts that contaminate the physics signal.

**Shape assertions:**
- `psi`: `(B, 1, 150, 150)` ✅
- `alpha`: `(B, 2, 150, 150)` ✅
- `S_hat`, `R`: `(B, 1, 150, 150)` ✅

---

## Stage 3: `EfficientNetV2Head`

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 9

Receives the 4-channel physics stack `X_cls = [I, κ̂, Ŝ, R]` and outputs class logits.

### Channel Projection

```
(B, 4, H, W) → Conv2d(4→3, 1×1) → BatchNorm2d → SiLU → (B, 3, H, W)
```

A learnable 1×1 convolution maps the 4 physics-derived channels to the 3-channel RGB domain expected by EfficientNetV2-S. This layer is **always fine-tuned** regardless of freezing schedule.

### Backbone Freezing Schedule

| Layers | Status | Reason |
|:---|:---:|:---|
| `features.0–2` | **Frozen** | Low-level edge/texture detectors transfer directly from ImageNet |
| `features.3–7` | **Fine-tuned** | Higher-level patterns need adaptation to astrophysical images |
| `classifier` | **Replaced** | `Dropout(0.2) → Linear(1280→3)` for 3-class output |

ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) is applied **inside** the module as registered buffers — always correct regardless of call context.

**Trainable: 19.87M / Total: 20.18M** ✅

---

## Full Model: `D4LensPINN`

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 10

```python
class D4LensPINN(nn.Module):
    def forward(self, I):
        X_in         = self.preprocess(I)              # (B,1,H,W) → (B,2,H,W)
        kappa        = self.d4unet(X_in)               # (B,2,H,W) → (B,1,H,W)  κ̂≥0
        psi          = self.poisson(kappa)             # ∇²Ψ = 2κ̂
        alpha        = self.deflection(psi)            # α = ∇Ψ → (B,2,H,W)
        S_hat, R     = self.inv_lens(I, alpha)         # β=θ-α(θ), R=I-Ŝ
        X_cls        = cat([I, kappa, S_hat, R], dim=1) # (B,4,H,W)
        logits       = self.classifier(X_cls)          # → (B,3)
        return logits, kappa                           # kappa for PhysicsLoss
```

`kappa` is returned alongside `logits` to avoid a redundant second forward pass during loss computation in the training loop.

**Total: 20.33M trainable / 20.65M** ✅

---

## Baseline: `ResNet18Baseline`

**File:** `D4-PINN + Differentiable Lensing Engine.ipynb`, Cell 11

Standard ResNet-18 trained **from scratch** (`weights=None`) on raw grayscale images. Two surgical patches:

```python
self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Original: Conv2d(3→64) for RGB. Patched to (1→64) for single-channel grayscale.

self.net.fc = nn.Linear(512, num_classes)
# Original: Linear(512→1000) for ImageNet. Replaced with 3-class head.
```

**Purpose:** Receives only raw pixel intensity — no physics preprocessing, no D4 equivariance, no residual lensing channels. `weights=None` ensures zero ImageNet prior. This is the true from-scratch baseline against which all D4LensPINN AUC improvements are measured.

**Params: 11.17M**

---

## Vanilla Control: `VanillaLensPINN`

**File:** `vanilla-notebook.ipynb`

Identical pipeline to `D4LensPINN` but Stage 1 (`EfficientD4UNet`) is replaced by `VanillaUNet` — a standard (non-equivariant) convolutional U-Net with identical channel counts and topology.

```python
class VanillaUNet(nn.Module):
    # Encoder: Conv2d → BatchNorm → ReLU → _VanillaResBlock (stride=2) ×2
    # Decoder: bilinear upsample + skip cat + Conv2d ×3
    # Output: Conv2d(64→1) + Softplus  →  κ̂  (NOT equivariant)
```

**Purpose:** Causal control for the Mechanistic Interpretability experiment. If D4 equivariance causes a measurable difference in internal representations, `VanillaLensPINN`'s κ̂ will NOT be D4-invariant (verified by Sanity Check 5 in the notebook).

Key difference from `D4LensPINN.d4unet`:
- `escnn` `R2Conv` layers → replaced by plain `nn.Conv2d`
- `InnerBatchNorm` → replaced by `nn.BatchNorm2d`
- `GroupPooling` → removed; direct `Conv2d(64→1)` output
