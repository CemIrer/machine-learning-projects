"""
COMP4350 - Midterm Exam Project
Part A: MLP (Multi-Layer Perceptron) Neural Network — built from scratch
=========================================================================
Goal          : Predict house prices (regression)
Architecture  : 3 inputs → 20 → 10 → 5 neurons (tanh) → 1 output (linear)
Optimizer     : Mini-Batch SGD with momentum + step-based LR decay
Normalization : Min-Max scaling applied to ALL features and the target
Allowed libs  : numpy, pandas  (for the algorithm itself)
Reproducible  : Fixed random seed ensures identical results every run
=========================================================================
"""

import sys
import numpy as np
import pandas as pd

# ── Reproducibility (fixed seed → same results every run) ─────────────────────
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

# ── Network & training hyper-parameters ───────────────────────────────────────
LAYER_DIMS     = [3, 20, 10, 5, 1] # [input, hidden1, hidden2, hidden3, output]
INIT_LR        = 0.010            # initial learning rate
MOMENTUM_COEF  = 0.88             # fraction of previous velocity to carry forward
MAX_EPOCHS     = 2000             # maximum training epochs
BATCH_SIZE     = 16               # mini-batch size (samples per gradient update)
LR_DECAY_EVERY = 200              # reduce LR every N epochs
LR_DECAY_MULT  = 0.92             # LR is multiplied by this factor at each decay step
GRAD_CLIP_VAL  = 4.0              # clip gradient magnitude to prevent exploding gradients
EARLY_STOP_EPS = 1e-7             # stop early if normalised MSE drops below this


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(train_path, test_path):
    """
    Read the Excel files and split into feature matrices and target vectors.
    Feature columns : Neighborhood, Age (Years), Net Square Meters (m2)
    Target column   : Price (TRY)
    Returns numpy float64 arrays.
    """
    train_df = pd.read_excel(train_path)
    test_df  = pd.read_excel(test_path)

    feat_cols  = ['Neighborhood', 'Age (Years)', 'Net Square Meters (m2)']
    label_col  = 'Price (TRY)'

    X_tr = train_df[feat_cols].values.astype(np.float64)
    y_tr = train_df[label_col].values.astype(np.float64)
    X_te = test_df[feat_cols].values.astype(np.float64)
    y_te = test_df[label_col].values.astype(np.float64)

    return X_tr, y_tr, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# 2. MIN-MAX NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def minmax_normalize(X_tr, y_tr, X_te, y_te):
    """
    Fit normalization statistics on training data ONLY, then apply to both
    train and test sets (avoids data leakage).
    All values are mapped to [0, 1] range.
    Returns scaled arrays and a dict of scaling parameters for inverse-transform.
    """
    # --- feature scaling ---
    feat_min   = X_tr.min(axis=0)
    feat_max   = X_tr.max(axis=0)
    feat_range = feat_max - feat_min + 1e-9        # avoid division by zero

    X_tr_n = (X_tr - feat_min) / feat_range
    X_te_n = (X_te - feat_min) / feat_range

    # --- target scaling ---
    y_min   = float(y_tr.min())
    y_max   = float(y_tr.max())
    y_range = y_max - y_min + 1e-9

    y_tr_n = (y_tr - y_min) / y_range
    y_te_n = (y_te - y_min) / y_range

    scale = {
        'feat_min': feat_min, 'feat_range': feat_range,
        'y_min':    y_min,    'y_max':      y_max,
        'y_range':  y_range,
    }
    return X_tr_n, y_tr_n, X_te_n, y_te_n, scale


def inverse_y(y_scaled, scale):
    """Map normalised predictions back to original TRY scale."""
    return y_scaled * scale['y_range'] + scale['y_min']


# ══════════════════════════════════════════════════════════════════════════════
# 3. MLP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MLP:
    """
    Multi-Layer Perceptron for regression, built entirely from scratch.

    Architecture (configurable via `dims` list):
        Hidden layers  — tanh activation (smooth, bounded, zero-centred)
        Output layer   — linear / identity (needed for unbounded regression)

    Weight initialisation:
        Glorot-uniform for the first hidden layer.
        He-normal for subsequent layers (more suitable after tanh).

    Training method:
        Mini-Batch SGD: gradients averaged over BATCH_SIZE samples per update.
        Momentum term prevents oscillations and speeds convergence.
        Step-based learning-rate decay avoids overshooting near the minimum.
        Gradient clipping prevents the exploding-gradient problem.
    """

    def __init__(self, dims, lr, momentum):
        self.dims     = dims
        self.lr       = lr
        self.momentum = momentum
        self.n_layers = len(dims) - 1    # number of weight matrices

        # --- weight initialisation ---
        self.W, self.b = [], []
        for i in range(self.n_layers):
            fan_in  = dims[i]
            fan_out = dims[i + 1]
            if i == 0:
                # Glorot uniform: good for tanh in first layer
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            else:
                # He normal: works well for subsequent tanh layers
                W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.W.append(W)
            self.b.append(np.zeros((1, fan_out)))

        # --- velocity buffers (momentum) ---
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ── forward pass ──────────────────────────────────────────────────────────
    def _forward(self, x_row):
        """
        Forward pass for a single training sample.
        x_row  : shape (1, n_features)
        Returns : pre-activations `zs`  and all activations `acts`
                  (acts[0] = input, acts[-1] = network output)
        """
        zs, acts = [], [x_row]

        for i in range(self.n_layers):
            z = acts[-1] @ self.W[i] + self.b[i]
            zs.append(z)
            # hidden layers → tanh;  output layer → linear (identity)
            a = np.tanh(z) if i < self.n_layers - 1 else z
            acts.append(a)

        return zs, acts

    # ── backward pass + weight update (mini-batch) ────────────────────────────
    def _backward(self, zs, acts, y_batch):
        """
        Backpropagation for a mini-batch of samples.
        Averages gradients across all samples in the batch before updating.
        Uses old weight values (saved before update) for delta propagation
        to avoid using stale weights in the chain rule.
        """
        batch_n = len(y_batch)
        preds   = acts[-1].flatten()                          # shape (batch_n,)

        # MSE gradient at output: dL/d_pred = 2/n * (pred - y)
        delta = ((2.0 / batch_n) * (preds - y_batch)).reshape(-1, 1)  # (batch_n, 1)

        for i in range(self.n_layers - 1, -1, -1):
            # gradient w.r.t. weights: sum over batch (already /n from delta)
            grad_W = acts[i].T @ delta                        # (fan_in, fan_out)
            grad_b = delta.sum(axis=0, keepdims=True)         # (1, fan_out)

            # clip to prevent exploding gradients
            grad_W = np.clip(grad_W, -GRAD_CLIP_VAL, GRAD_CLIP_VAL)
            grad_b = np.clip(grad_b, -GRAD_CLIP_VAL, GRAD_CLIP_VAL)

            # save weights BEFORE update — used for correct delta propagation
            old_W = self.W[i].copy()

            # momentum SGD update
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grad_W
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grad_b
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

            # propagate delta to the previous layer using the PRE-update weights
            if i > 0:
                delta = (delta @ old_W.T) * (1.0 - np.tanh(zs[i - 1]) ** 2)

    # ── training loop ─────────────────────────────────────────────────────────
    def train(self, X, y):
        """
        Mini-batch SGD training loop.
        Each epoch: shuffle all samples → split into batches of BATCH_SIZE
                    → forward + backward for each batch.
        LR decays by LR_DECAY_MULT every LR_DECAY_EVERY epochs.
        Stops early if normalised MSE < EARLY_STOP_EPS.
        Returns list of per-epoch MSE values (on normalised target).
        """
        n = X.shape[0]
        loss_log = []

        for epoch in range(1, MAX_EPOCHS + 1):

            # step-based LR decay
            if epoch % LR_DECAY_EVERY == 0:
                self.lr *= LR_DECAY_MULT

            # shuffle all training samples at the start of each epoch
            order  = np.random.permutation(n)
            X_shuf = X[order]
            y_shuf = y[order]

            epoch_sq_err = 0.0

            # iterate over mini-batches
            for start in range(0, n, BATCH_SIZE):
                X_batch = X_shuf[start: start + BATCH_SIZE]   # (batch_n, n_feat)
                y_batch = y_shuf[start: start + BATCH_SIZE]   # (batch_n,)

                zs, acts = self._forward(X_batch)
                self._backward(zs, acts, y_batch)

                preds = acts[-1].flatten()
                epoch_sq_err += float(np.sum((preds - y_batch) ** 2))

            mse = epoch_sq_err / n
            loss_log.append(mse)

            if epoch % 100 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>5d}/{MAX_EPOCHS}  │  MSE(norm): {mse:.7f}"
                      f"  │  LR: {self.lr:.6f}")

            if mse < EARLY_STOP_EPS:
                print(f"  Early stop at epoch {epoch} (MSE < {EARLY_STOP_EPS})")
                break

        return loss_log

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, X):
        """Run forward pass for every row in X and return predictions array."""
        preds = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            _, acts = self._forward(X[i: i + 1, :])
            preds[i] = float(acts[-1].flat[0])
        return preds


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGRESSION METRICS (computed from scratch using numpy)
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """
    Compute MAE, MSE, RMSE, and R² in the original (un-normalised) scale.
    All formulas implemented with pure numpy — no sklearn.
    """
    err      = y_pred - y_true
    n        = len(y_true)

    mae      = float(np.sum(np.abs(err)) / n)
    mse      = float(np.sum(err ** 2) / n)
    rmse     = float(np.sqrt(mse))

    ss_res   = float(np.sum(err ** 2))
    ss_tot   = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORT WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_report(model, train_m, test_m, loss_log,
                 y_tr, y_tr_pred, y_te, y_te_pred,
                 scale, report_path='report.txt'):
    """
    Write a detailed structured report file containing:
      • Dataset summary and normalization details
      • Algorithm explanation (MLP, backprop, SGD)
      • Full ANN parameters
      • Training progress log (key epochs)
      • Metric formulas with computed values
      • Sample predictions (first 10 train + first 10 test)
      • Final weights and biases of every layer
    """
    lines = []

    def ln(s=''):
        s = str(s) if not isinstance(s, str) else s
        lines.append(s)
        print(s)

    SEP = "=" * 70

    ln(SEP)
    ln("  COMP4350 — Midterm Exam Project — Part A Report")
    ln("  MLP Neural Network with Backpropagation (implemented from scratch)")
    ln(SEP)
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: DATASET SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 1: DATASET SUMMARY")
    ln("━" * 70)
    ln()
    ln("  Task         : House price regression (predict Price in TRY)")
    ln("  Source files : midtermProject-part1-train.xlsx")
    ln("                 midtermProject-part1-test.xlsx")
    ln()
    ln("  Input features (3 total):")
    ln("    1. Neighborhood         — categorical integer (1, 2, 3, 4)")
    ln("    2. Age (Years)          — building age in years (0 – 33)")
    ln("    3. Net Square Meters    — usable living area in m² (50 – 245)")
    ln()
    ln("  Output (target):")
    ln("    Price (TRY)             — house price in Turkish Lira")
    ln()
    ln(f"  Training set size        : {len(y_tr)} records")
    ln(f"  Test set size            : {len(y_te)} records")
    ln(f"  Price range (train)      : {y_tr.min():>14,.0f} TRY  (min)")
    ln(f"                             {y_tr.max():>14,.0f} TRY  (max)")
    ln(f"  Price mean  (train)      : {y_tr.mean():>14,.0f} TRY")
    ln(f"  Price std   (train)      : {y_tr.std():>14,.0f} TRY")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: NORMALISATION
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 2: NORMALISATION")
    ln("━" * 70)
    ln()
    ln("  Method: Min-Max Scaling — maps all values to the [0, 1] range.")
    ln("  Formula applied to each column independently:")
    ln()
    ln("           x_norm = (x - x_min) / (x_max - x_min)")
    ln()
    ln("  IMPORTANT: normalization statistics (min, max) are computed")
    ln("  ONLY on the training set to prevent data leakage from the test set.")
    ln("  The same training statistics are then applied to the test set.")
    ln()
    ln("  Scaling parameters (computed from training set):")
    feat_names = ['Neighborhood', 'Age (Years)', 'Net Sq.Meters (m2)']
    for i, name in enumerate(feat_names):
        lo = float(scale['feat_min'][i])
        hi = float(scale['feat_min'][i] + scale['feat_range'][i])
        ln(f"    {name:<24} min={lo:>7.2f}   max={hi:>7.2f}")
    ln(f"    {'Price (TRY)':<24} min={scale['y_min']:>14,.0f}   "
       f"max={scale['y_min'] + scale['y_range']:>14,.0f}")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: ALGORITHM DESCRIPTION
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 3: ALGORITHM DESCRIPTION — MLP with Backpropagation")
    ln("━" * 70)
    ln()
    ln("  A Multi-Layer Perceptron (MLP) is a fully connected feedforward")
    ln("  neural network. Information flows in one direction: input → hidden")
    ln("  layers → output. Each neuron computes a weighted sum of its inputs,")
    ln("  adds a bias, and passes the result through an activation function.")
    ln()
    ln("  FORWARD PASS:")
    ln("  ─────────────")
    ln("  For each layer l (l = 1 … L):")
    ln()
    ln("    z^(l) = a^(l-1) · W^(l) + b^(l)     (linear combination)")
    ln("    a^(l) = f( z^(l) )                   (activation function)")
    ln()
    ln("  where:")
    ln("    a^(0) = x   (input vector, normalised)")
    ln("    W^(l) = weight matrix for layer l")
    ln("    b^(l) = bias vector for layer l")
    ln("    f(·)  = tanh for hidden layers, identity for output layer")
    ln()
    ln("  tanh activation formula:")
    ln("    tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))")
    ln("    tanh'(z) = 1 - tanh(z)²   ← used in backpropagation")
    ln()
    ln("  Loss function: Mean Squared Error (MSE) on normalised targets")
    ln("    L = (1/n) · Σ (ŷ_i - y_i)²")
    ln()
    ln("  BACKPROPAGATION:")
    ln("  ─────────────────")
    ln("  Starting from the output layer, the error gradient is propagated")
    ln("  backwards through the network using the chain rule:")
    ln()
    ln("  Output layer delta:    δ^(L) = 2 · (ŷ - y)   [MSE derivative]")
    ln("  Hidden layer delta:    δ^(l) = (δ^(l+1) · W^(l+1)ᵀ) ⊙ tanh'(z^(l))")
    ln()
    ln("  Gradients:")
    ln("    ∂L/∂W^(l) = a^(l-1)ᵀ · δ^(l)")
    ln("    ∂L/∂b^(l) = δ^(l)")
    ln()
    ln("  SGD WITH MOMENTUM UPDATE RULE:")
    ln("  ────────────────────────────────")
    ln("  Velocity (momentum) buffers carry a fraction of the previous")
    ln("  update direction to smooth oscillations and accelerate learning:")
    ln()
    ln("    v_W^(l) ← α · v_W^(l) - η · ∂L/∂W^(l)")
    ln("    W^(l)   ← W^(l) + v_W^(l)")
    ln()
    ln("  where:  η = learning rate,  α = momentum coefficient")
    ln()
    ln("  GRADIENT CLIPPING:")
    ln("  ───────────────────")
    ln("  If any gradient value exceeds ±4.0, it is clipped to ±4.0.")
    ln("  This prevents the exploding gradient problem that can occur")
    ln("  in deep networks or when the loss surface is steep.")
    ln()
    ln("  LEARNING RATE DECAY:")
    ln("  ─────────────────────")
    ln("  Every 200 epochs the learning rate is multiplied by 0.92.")
    ln("  This allows large steps early in training (fast convergence)")
    ln("  and smaller steps later (fine-tuning near the minimum).")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: ANN PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 4: ANN PARAMETERS")
    ln("━" * 70)
    ln()
    ln("  ┌─────────────────────────────────────────────────────────────┐")
    ln(f"  │  Architecture (layer sizes) : {' → '.join(str(d) for d in LAYER_DIMS):<30}│")
    ln(f"  │  Total trainable parameters : "
       f"{sum(w.size + b.size for w, b in zip(model.W, model.b)):<30}│")
    ln(f"  │  Hidden layer activation    : {'tanh':<30}│")
    ln(f"  │  Output layer activation    : {'linear / identity':<30}│")
    ln(f"  │  Initial learning rate (η)  : {INIT_LR:<30}│")
    ln(f"  │  Momentum coefficient (α)   : {MOMENTUM_COEF:<30}│")
    ln(f"  │  Mini-batch size (SGD)      : {BATCH_SIZE:<30}│")
    ln(f"  │  LR decay factor            : {LR_DECAY_MULT} every {LR_DECAY_EVERY} epochs{'':<16}│")
    ln(f"  │  Gradient clipping          : ±{GRAD_CLIP_VAL:<29}│")
    ln(f"  │  Max training epochs        : {MAX_EPOCHS:<30}│")
    ln(f"  │  Early stop threshold       : MSE(norm) < {EARLY_STOP_EPS:<19}│")
    ln(f"  │  Weight init (Layer 1)      : {'Glorot uniform':<30}│")
    ln(f"  │  Weight init (Layers 2+)    : {'He normal':<30}│")
    ln(f"  │  Training method            : {'Mini-Batch SGD (batch_size=16)':<30}│")
    ln(f"  │  Sample shuffle per epoch   : {'Yes (random permutation)':<30}│")
    ln(f"  │  Random seed                : {RANDOM_SEED:<30}│")
    ln("  └─────────────────────────────────────────────────────────────┘")
    ln()
    ln("  Parameter count breakdown:")
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        layer_type = "Output" if i == model.n_layers - 1 else f"Hidden {i + 1}"
        ln(f"    Layer {i+1} [{layer_type}] : "
           f"{W.shape[0]}×{W.shape[1]} weights + {b.size} biases "
           f"= {W.size + b.size} params")
    total = sum(w.size + b.size for w, b in zip(model.W, model.b))
    ln(f"    {'─' * 48}")
    ln(f"    TOTAL PARAMETERS : {total}")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: TRAINING PROGRESS
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 5: TRAINING PROGRESS (MSE on normalised target)")
    ln("━" * 70)
    ln()
    ln("  Epoch   | MSE (normalised) | Change from prev")
    ln("  --------|------------------|------------------")
    checkpoints = [1, 100, 200, 300, 400, 500, 600,
                   700, 800, 900, 1000, 1100, 1200]
    prev_mse = None
    for ep in checkpoints:
        idx = ep - 1
        if idx >= len(loss_log):
            idx = len(loss_log) - 1
        mse_val = loss_log[idx]
        if prev_mse is None:
            change_str = "  —  (initial)"
        else:
            delta = mse_val - prev_mse
            change_str = f"  {delta:+.7f}"
        ln(f"  {ep:>6d}  |  {mse_val:.7f}       |{change_str}")
        prev_mse = mse_val
    ln()
    ln(f"  Total epochs trained : {len(loss_log)}")
    ln(f"  Initial MSE(norm)    : {loss_log[0]:.7f}")
    ln(f"  Final   MSE(norm)    : {loss_log[-1]:.7f}")
    ln(f"  MSE reduction        : {loss_log[0] - loss_log[-1]:.7f}  "
       f"({100*(1 - loss_log[-1]/loss_log[0]):.1f}%)")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: PERFORMANCE METRICS
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 6: PERFORMANCE METRICS")
    ln("━" * 70)
    ln()
    ln("  Metric formulas (applied in original TRY scale):")
    ln()
    ln("    MAE  = (1/n) · Σ |ŷ_i - y_i|")
    ln("           Average absolute prediction error.")
    ln()
    ln("    MSE  = (1/n) · Σ (ŷ_i - y_i)²")
    ln("           Penalises large errors more heavily than MAE.")
    ln()
    ln("    RMSE = √MSE")
    ln("           Same unit as the target (TRY); easier to interpret.")
    ln()
    ln("    R²   = 1 - SS_res / SS_tot")
    ln("           SS_res = Σ(ŷ_i - y_i)²   (residual sum of squares)")
    ln("           SS_tot = Σ(y_i - ȳ)²     (total sum of squares)")
    ln("           R² = 1 → perfect fit | R² = 0 → predicts the mean")
    ln()
    ln("  ┌────────────────────────────────────────────┐")
    ln("  │ Train Results (501 samples)                │")
    ln("  ├────────────────────────────────────────────┤")
    ln(f"  │  MAE  : {train_m['MAE']:>18,.2f}  TRY        │")
    ln(f"  │  MSE  : {train_m['MSE']:>18,.2f}  TRY²       │")
    ln(f"  │  RMSE : {train_m['RMSE']:>18,.2f}  TRY        │")
    ln(f"  │  R²   : {train_m['R2']:>22.6f}             │")
    ln("  └────────────────────────────────────────────┘")
    ln()
    ln("  ┌────────────────────────────────────────────┐")
    ln("  │ Test Results — Macro Average (126 samples) │")
    ln("  ├────────────────────────────────────────────┤")
    ln(f"  │  MAE  : {test_m['MAE']:>18,.2f}  TRY        │")
    ln(f"  │  MSE  : {test_m['MSE']:>18,.2f}  TRY²       │")
    ln(f"  │  RMSE : {test_m['RMSE']:>18,.2f}  TRY        │")
    ln(f"  │  R²   : {test_m['R2']:>22.6f}             │")
    ln("  └────────────────────────────────────────────┘")
    ln()
    ln("  Interpretation:")
    ln(f"    On average, the model's house price predictions are off by")
    ln(f"    approximately {test_m['MAE']:,.0f} TRY on the test set.")
    ln(f"    The RMSE of {test_m['RMSE']:,.0f} TRY reflects larger individual errors.")
    ln(f"    R² = {test_m['R2']:.4f} means the model explains about "
       f"{test_m['R2']*100:.1f}% of the")
    ln(f"    variance in house prices on the test set.")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: SAMPLE PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 7: SAMPLE PREDICTIONS (first 10 of each set)")
    ln("━" * 70)
    ln()
    ln("  Training set predictions:")
    ln(f"  {'#':>4}  {'Actual (TRY)':>18}  {'Predicted (TRY)':>18}  {'Error (TRY)':>16}")
    ln("  " + "─" * 62)
    for i in range(min(10, len(y_tr))):
        err = y_tr_pred[i] - y_tr[i]
        ln(f"  {i+1:>4}  {y_tr[i]:>18,.0f}  {y_tr_pred[i]:>18,.0f}  {err:>+16,.0f}")
    ln()
    ln("  Test set predictions:")
    ln(f"  {'#':>4}  {'Actual (TRY)':>18}  {'Predicted (TRY)':>18}  {'Error (TRY)':>16}")
    ln("  " + "─" * 62)
    for i in range(min(10, len(y_te))):
        err = y_te_pred[i] - y_te[i]
        ln(f"  {i+1:>4}  {y_te[i]:>18,.0f}  {y_te_pred[i]:>18,.0f}  {err:>+16,.0f}")
    ln()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8: FINAL WEIGHTS AND BIASES
    # ══════════════════════════════════════════════════════════════════════════
    ln("━" * 70)
    ln("  SECTION 8: FINAL WEIGHTS AND BIASES (values after training)")
    ln("━" * 70)
    ln()
    ln("  These are the learned parameters of the network at the end of")
    ln("  training. All values are in normalised space (inputs and targets")
    ln("  were scaled to [0,1] before training).")
    ln()
    for i, (Wm, b) in enumerate(zip(model.W, model.b)):
        layer_type = "Output" if i == model.n_layers - 1 else f"Hidden {i + 1}"
        ln(f"  Layer {i + 1}  [{layer_type}]")
        ln(f"  Input  dimension : {Wm.shape[0]}")
        ln(f"  Output dimension : {Wm.shape[1]}")
        ln(f"  Weight matrix W  ({Wm.shape[0]} rows × {Wm.shape[1]} cols):")
        for row_idx, row in enumerate(Wm):
            vals = "  ".join(f"{v:+.7f}" for v in row)
            ln(f"    W[{row_idx}] = [ {vals} ]")
        bias_vals = "  ".join(f"{v:+.7f}" for v in b.flatten())
        ln(f"  Bias vector b  = [ {bias_vals} ]")
        ln()

    ln(SEP)
    ln("  End of Report")
    ln(SEP)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TRAIN_PATH = 'midtermProject-part1-train.xlsx'
    TEST_PATH  = 'midtermProject-part1-test.xlsx'

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading datasets...")
    X_tr, y_tr, X_te, y_te = load_datasets(TRAIN_PATH, TEST_PATH)
    print(f"  Train : {X_tr.shape[0]} samples  |  Test : {X_te.shape[0]} samples")
    print(f"  Price range (train): {y_tr.min():,.0f} – {y_tr.max():,.0f} TRY")

    # ── Normalize ─────────────────────────────────────────────────────────────
    print("\nApplying min-max normalisation (fit on train set only)...")
    X_tr_n, y_tr_n, X_te_n, y_te_n, scale = minmax_normalize(X_tr, y_tr, X_te, y_te)

    # ── Build & train MLP ─────────────────────────────────────────────────────
    print(f"\nBuilding MLP  ({' → '.join(str(d) for d in LAYER_DIMS)})")
    print(f"Training with SGD  (LR={INIT_LR}, momentum={MOMENTUM_COEF}, "
          f"max {MAX_EPOCHS} epochs)\n")
    model = MLP(LAYER_DIMS, lr=INIT_LR, momentum=MOMENTUM_COEF)
    loss_log = model.train(X_tr_n, y_tr_n)

    # ── Predict and inverse-transform ─────────────────────────────────────────
    print("\nGenerating predictions on train and test sets...")
    y_tr_pred_n = model.predict(X_tr_n)
    y_te_pred_n = model.predict(X_te_n)

    y_tr_pred = inverse_y(y_tr_pred_n, scale)    # back to TRY
    y_te_pred = inverse_y(y_te_pred_n, scale)

    # ── Compute metrics ───────────────────────────────────────────────────────
    train_m = regression_metrics(y_tr, y_tr_pred)
    test_m  = regression_metrics(y_te, y_te_pred)

    # ── Write report ──────────────────────────────────────────────────────────
    print("\nWriting report...\n")
    write_report(model, train_m, test_m, loss_log,
                 y_tr, y_tr_pred, y_te, y_te_pred, scale)

    print("\nDone.  Deliverable: report.txt")
