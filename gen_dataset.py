"""
Generate the MVDR dataset (same grid as the user's Phase 2) with:
  X_deg (M,2)     : theta_s, theta_i in degrees
  Y     (M,2N)    : split-order [re part, im part] real targets
  W     (M,N)     : complex MVDR weights (float64)
  AB    (M,2)     : (alpha, beta) with w = alpha*a_s + beta*a_i   [F1 basis]
  GT    (M,2)     : (g, tau)    with w = g*u + tau*a_i            [null-aligned]
  JMIN  (M,)      : J_min = 1/(a_s^H R^-1 a_s)
Plus a fixed train/val/test split used by every ablation variant.

Pairs with |theta_s - theta_i| < MIN_SEP_DEG are excluded: as the separation
shrinks, N^2 - |rho|^2 -> 0 and the labels (beta, tau) blow up (at 1 deg
mid-array |tau|/DELTA ~ 8, at 1 deg endfire ~ 30 -> required net outputs of
160-600 vs O(0.3) typical), and u itself loses float32 precision.

The module is intentionally import-safe: generation happens only when this file
is executed as a script, not when another module imports its constants.
"""
from pathlib import Path

import numpy as np

import physics as ph

N = 8
MIN_SEP_DEG = 3.0
ANGLES = np.arange(-60, 61, 1)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def build_dataset():
    """Return the arrays and deterministic split used by the ablation study."""
    rng = np.random.default_rng(42)
    X, W, AB, GT, JMIN = [], [], [], [], []

    for ts in ANGLES:
        for ti in ANGLES:
            if abs(ti - ts) < MIN_SEP_DEG:
                continue
            w = ph.mvdr_np(ts, ti, N)
            a_s = ph.steering_np(ts, N)
            a_i = ph.steering_np(ti, N)
            u = ph.exact_null_np(ts, ti, N)

            # (alpha, beta): lstsq on G = [a_s, a_i].
            G = np.stack([a_s, a_i], axis=1)
            coef, *_ = np.linalg.lstsq(G, w, rcond=None)

            # (g, tau): w = g*u + tau*a_i.  Because u^H a_i = 0,
            # tau* = (w^H a_i)/N and g* = (w^H u)/||u||^2.
            tau = np.conj((w.conj() @ a_i) / N)
            g = np.conj((w.conj() @ u) / (u.conj() @ u))

            X.append([ts, ti])
            W.append(w)
            AB.append(coef)
            GT.append([g, tau])
            JMIN.append(ph.j_min_np(ts, ti, N))

    X = np.array(X)
    W = np.array(W)
    AB = np.array(AB)
    GT = np.array(GT)
    JMIN = np.array(JMIN)

    # User-format real target matrix.
    Y = np.concatenate([W.real, W.imag], axis=1).astype(np.float32)

    # Fixed split so every ablation variant sees the same examples.
    M = len(X)
    idx = rng.permutation(M)
    n_train = int(0.8 * M)
    n_val = int(0.1 * M)
    split = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    return X, Y, W, AB, GT, JMIN, split


def main():
    X, Y, W, AB, GT, JMIN, split = build_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "dataset_N8.npz",
        X_deg=X,
        Y=Y,
        W=W,
        AB=AB,
        GT=GT,
        JMIN=JMIN,
        idx_train=split["train"],
        idx_val=split["val"],
        idx_test=split["test"],
    )

    # Sanity checks: both decompositions must reconstruct W.
    A_s = np.stack([ph.steering_np(t, N) for t in X[:, 0]])
    A_i = np.stack([ph.steering_np(t, N) for t in X[:, 1]])
    U = np.stack([ph.exact_null_np(ts, ti, N) for ts, ti in X])
    err_ab = np.max(np.abs(W - (AB[:, :1] * A_s + AB[:, 1:2] * A_i)))
    err_gt = np.max(np.abs(W - (GT[:, :1] * U + GT[:, 1:2] * A_i)))

    taus = GT[:, 1]
    nulls = 20 * np.log10(np.abs(taus) * N + 1e-30)
    n_train = len(split["train"])
    n_val = len(split["val"])
    print(f"samples: {len(X)}   (train {n_train} / val {n_val} / test {len(split['test'])})")
    print(f"max |w - (alpha a_s + beta a_i)|      = {err_ab:.2e}")
    print(f"max |w - (g u + tau a_i)|             = {err_gt:.2e}")
    print(f"min angle separation: {np.abs(X[:, 0] - X[:, 1]).min():.0f} deg  (MIN_SEP_DEG={MIN_SEP_DEG})")
    print(f"|tau| range: {np.abs(taus).min():.2e} .. {np.abs(taus).max():.2e}")
    print(f"implied MVDR null depth (20log10|tau|N): {nulls.min():.1f} .. {nulls.max():.1f} dB")
    print(f"JMIN range: {JMIN.min():.3f} .. {JMIN.max():.3f}")
    print(f"|g| range: {np.abs(GT[:, 0]).min():.4f} .. {np.abs(GT[:, 0]).max():.4f}")
    print(f"saved {DATA_DIR / 'dataset_N8.npz'}")


if __name__ == "__main__":
    main()
