# Training Reference — D4-LensPINN v4

## Dataset

**Source:** 30,000 grayscale `.npy` images (150×150 pixels), 3 perfectly balanced classes.

```
Total 30,000  →  Train 24,000  |  Val 3,000  |  Test 3,000
Per class:         8,000       |   1,000     |   1,000
```

**Split strategy:** Stratified 80/10/10 using `StratifiedShuffleSplit` applied twice (seed=42). Class weights compute to `[1.0, 1.0, 1.0]` — perfect balance, no weighted sampling required.

**Loader:** `npy_loader` reads `.npy` → `np.squeeze` → `float32` → `PIL.Image`. `DatasetFolder(extensions=('.npy',))` auto-discovers class folders. `transforms.ToTensor()` normalises pixels to [0, 1].

**DataLoader settings:** `num_workers=2, pin_memory=True, persistent_workers=True`  
> ⚠️ Reset `persistent_workers=False` before Optuna to prevent worker-process leaks between trials.

---

## Loss Functions

### Classification Loss: `nn.CrossEntropyLoss`

Plain cross-entropy — no focal loss weighting. Applied to logits output by `EfficientNetV2Head`.

### Physics Loss: `PhysicsLoss` (4 terms)

```
L_phys = λ_tv · (TV_x + TV_y)
       + λ_l1 · E[κ̂]
       + λ_ctr · E[κ̂ · r²]
       + λ_poisson · E[(∇²Ψ − 2κ̂)²]
```

| Term | Default λ | Purpose |
|:---|:---:|:---|
| Total Variation on κ̂ | `0.005` | Spatial smoothness of convergence map |
| L1 on κ̂ | `0.001` | Sparsity — encodes compact subhalo prior |
| Centre penalty `κ̂·r²` | `0.002` | Biases mass toward lens centre (physically correct) |
| Poisson residual | `~0.01` (Optuna-tuned) | Physical self-consistency of Ψ |

**Poisson residual implementation:**  
Uses a 5-point finite-difference Laplacian kernel (registered as buffer):
```
lap = [[0, 1, 0],
       [1,-4, 1],
       [0, 1, 0]]
poisson_residual = mean((conv2d(psi, lap) - 2·kappa)²)
```

### Combined Training Loss

- **Phase 1:** `L = L_CE` (classification only — build structurally valid κ̂)
- **Phase 2:** `L = L_CE + L_phys` (impose physical consistency)

Mixed-precision note: `L_CE` runs inside `autocast('cuda')` returning `float16`. `L_phys` runs in `float32` (escnn and FFT operations incompatible with `float16`). They are combined as `cls_loss.float() + phys_v`.

---

## Optimizer and Scheduler

### `get_optimizer_and_scheduler`

```python
AdamW(trainable_params, lr=lr, weight_decay=wd)
```

Scheduler: **Warmup + CosineAnnealing** via `SequentialLR`:

```
Epochs 1–3:   LinearLR(start_factor=1/3, end_factor=1.0)
              ← stabilises escnn basis-expansion gradients at init
Epochs 4–end: CosineAnnealingLR(T_max=epochs-3, eta_min=1e-7)
```

### Mixed-Precision Training

```python
scaler = torch.amp.GradScaler('cuda')

# Per-batch:
with torch.amp.autocast('cuda'):
    logits = model.classifier(X_cls)
    cls_loss = criterion(logits, labels)
loss = cls_loss.float() + phys_v      # upcast before adding physics loss
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
scaler.step(optimizer)
scaler.update()
```

---

## Training Phases

### Phase 1 — Classification Only (15 epochs)

```python
train_model(
    model, train_loader, val_loader,
    device=DEVICE_PINN,
    epochs=15,
    is_pinn=True,
    use_phys_loss=False,      # ← physics loss OFF
    label='d4_phase1',
    lr=7.11e-4,               # Optuna best
    wd=1.46e-3,               # Optuna best
)
```

**Goal:** Build a structurally valid κ̂ by training only on classification loss. This establishes a convergence map that carries physically meaningful signal before the physics constraint is imposed.

### Phase 2 — Full Physics Training (40 epochs)

```python
train_model(
    model, train_loader, val_loader,
    device=DEVICE_PINN,
    epochs=40,
    is_pinn=True,
    use_phys_loss=True,       # ← physics loss ON
    label='d4_phase2',
    lr=4.46e-4,               # Optuna best (lower than P1 — see note)
    wd=1.46e-3,
    lambda_poisson=~0.01,     # Optuna best
)
```

**Why `LR_P2 < LR_P1`:** A lower Phase 2 learning rate prevents the physics loss transient from destroying the classification-useful κ̂ structure built in Phase 1.

### Checkpoint Strategy

Best checkpoint is saved per phase:
```python
if val_loss < best_val:
    torch.save(model.state_dict(), f'{label}_best.pth')
# At end of training, best checkpoint is restored automatically
model.load_state_dict(torch.load(f'{label}_best.pth', weights_only=True))
```

> ⚠️ The `vanilla-notebook.ipynb` training adds `start_epoch` and `ckpt_path` parameters for resuming interrupted long-run sessions on Kaggle.

---

## Data Augmentation

### Mixup (`Beta(α, α)`)

```python
def mixup_batch(imgs, labels, alpha=0.2):
    lam = Beta(alpha, alpha).sample()
    idx = randperm(batch_size)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam
```

Loss is computed as: `lam · CE(logits, la) + (1−lam) · CE(logits, lb)`  
Accuracy tracking uses `dominant = la if lam >= 0.5 else lb`.

---

## Hyperparameter Search: Optuna

**Strategy:** 15 trials, TPE sampler, MedianPruner (startup=5 trials, warmup=3 steps), 6,000-sample stratified subset, 8 epochs per trial.

**Search space:**

| Parameter | Distribution | Range | Best Value |
|:---|:---:|:---:|:---:|
| `lr_phase1` | log-uniform | [3e-4, 3e-3] | **7.11e-4** |
| `lr_phase2` | log-uniform | [5e-5, 5e-4] | **4.46e-4** |
| `weight_decay` | log-uniform | [5e-5, 5e-3] | **1.46e-3** |
| `lambda_poisson` | log-uniform | [0.001, 0.05] | **~0.01** |

`WD ≈ 1.46e-3` is ~10× the typical default, appropriate for regularising 20M parameters on 24,000 training samples.

---

## Evaluation

### `predict_no_tta` — Single Pass

```python
@torch.no_grad()
def predict_no_tta(model, loader, device, is_pinn=True):
    # single forward pass per batch → softmax → (N, 3) probs + labels
```

### `predict_with_tta` — D4 Test-Time Augmentation (8 transforms)

Averages softmax over all 8 D4 group elements:
```
{ rot0, rot90, rot180, rot270 } × { no flip, horizontal flip }
```

| Metric | No TTA | D4-TTA | Delta |
|:---|:---:|:---:|:---:|
| Macro AUC | 0.9786 | **0.9809** | +0.0023 |
| Test Accuracy | **90.4%** | 84.9% | −5.5 pp |

> ⚠️ **TTA accuracy regression:** Horizontal flip changes the *sign* of the Sobel saddle-point feature in `PhysicsPreprocess`, generating spurious substructure signals for smooth-lens (`no`) images. **Use TTA for AUC-optimised evaluation. Disable TTA if accuracy is the primary metric.**

### Evaluation Metric

**Macro-average one-vs-rest AUC-ROC** across 3 classes:
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
```

---

## Reproducibility

All randomness is seeded at the top of each notebook:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # required for full determinism
```

TF32 precision is enabled for speed:
```python
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision  = 'tf32'
```

---

## Multi-GPU Routing

```python
n_gpu = torch.cuda.device_count()
if n_gpu >= 2:
    DEVICE_PINN = 'cuda:0'   # D4LensPINN on GPU 0 (needs more VRAM for escnn)
    DEVICE_R18  = 'cuda:1'   # ResNet18 on GPU 1
    BATCH = 64
elif n_gpu == 1:
    DEVICE_PINN = DEVICE_R18 = 'cuda:0'
    BATCH = 32               # reduced batch to fit both models
else:
    DEVICE_PINN = DEVICE_R18 = 'cpu'
```
