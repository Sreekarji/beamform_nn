"""
recommended_solution.py — self-contained MVDR beamforming NN with null-depth
preservation (PyTorch).  Drop-in replacement for the user's Phase-3 pipeline.

Why this works (details + ablation in report.md):
  * MVDR weights live in span{a_s, a_i}; in the basis {u, a_i} with u the
    unit-gain EXACT-null vector, w = g*u + tau*a_i with g = 1 - tau*rho
    (rho = a_s^H a_i).  Then a_s^H w = 1 exactly and |w^H a_i| = |tau|*N
    exactly: the network's 2 real outputs are (the scaled) null coordinate.
  * Plain MSE on w in these coordinates reaches ~-90 dB median nulls and
    0.000 dB SINR gap vs MVDR on the standard dataset (baseline: -22.7 dB /
    15.3 dB).  Optional physics loss + dB-hinge for generalization.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'

# ─── scenario ────────────────────────────────────────────────────────────────
N = 8
SIGMA_N, SIGMA_S, SIGMA_I = 1.0, 100.0, 1000.0     # SNR 20 dB, INR 30 dB
DELTA = SIGMA_N / (SIGMA_N + SIGMA_I * N)          # natural scale of tau
TAU_SCALE = 0.05 * DELTA                           # tau = TAU_SCALE * net_output
MIN_SEP_DEG = 3.0                                  # exclude ill-conditioned close pairs

# ─── physics (numpy labels) ─────────────────────────────────────────────────
def steering_np(theta_deg):
    n = np.arange(N)
    return np.exp(1j * n * np.pi * np.sin(np.deg2rad(theta_deg)))

def mvdr_np(theta_s, theta_i):
    a_s, a_i = steering_np(theta_s), steering_np(theta_i)
    G = np.stack([a_s, a_i], axis=1)
    M = SIGMA_N * np.diag([1/SIGMA_S, 1/SIGMA_I]) + G.conj().T @ G
    v = a_s - G @ np.linalg.solve(M, G.conj().T @ a_s)
    return v / (a_s.conj() @ v)

def make_dataset():
    # labels computed in float64; training runs in complex64 (float32 null
    # floor ~ -150 dB is far below the ~-96 dB MVDR targets we need to match)
    rows = [(ts, ti, mvdr_np(ts, ti))
            for ts in np.arange(-60, 61, 1) for ti in np.arange(-60, 61, 1)
            if abs(ts - ti) >= MIN_SEP_DEG]
    X = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float32)
    W = torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.complex64)
    return X, W

# ─── physics (torch, differentiable) ────────────────────────────────────────
def steering_t(theta_deg):
    n = torch.arange(N, dtype=theta_deg.dtype, device=theta_deg.device)
    phase = torch.sin(theta_deg * (np.pi / 180)).unsqueeze(-1) * (n * np.pi)
    return torch.complex(torch.cos(phase), torch.sin(phase))

class BeamformerNet(nn.Module):
    """2 -> 128 -> 256 -> 256 -> 128 -> 2, physics layer on top.
    Input:  (B,2) [theta_s, theta_i] in degrees
    Output: (B,N) complex weights, always satisfying w^H a_s = 1 and
            |w^H a_i| = |tau|*N (tau = out_scale * net output)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2))

    def forward(self, ang_deg):
        ts, ti = ang_deg[:, 0], ang_deg[:, 1]
        out = self.net(torch.stack([ts / 60.0, ti / 60.0], dim=-1))
        a_s, a_i = steering_t(ts), steering_t(ti)
        rho = (a_s.conj() * a_i).sum(-1)                 # a_s^H a_i
        tau = TAU_SCALE * torch.complex(out[:, 0], out[:, 1])
        g = 1.0 - tau * rho                              # exact: a_s^H w = 1 (⇔ w^H a_s = 1)
        gamma = (a_i.conj() * a_s).sum(-1, keepdim=True) / N
        w0 = a_s - gamma * a_i                           # exact-null (before scale)
        u = w0 / (a_s.conj() * w0).sum(-1, keepdim=True)  # u^H a_i = 0, u^H a_s = 1
        return g.unsqueeze(-1) * u + tau.unsqueeze(-1) * a_i

# ─── losses ──────────────────────────────────────────────────────────────────
def loss_mse(w, w_star):
    d = torch.cat([(w - w_star).real, (w - w_star).imag], dim=-1)
    return (d ** 2).mean()

def loss_physics(w, a_s, a_i, j_min):
    """log J(w)/J_min with J the MVDR objective (optional, for generalization).
    PRECONDITION: w must satisfy a_s^H w = 1 (all charts here do); J_min is the
    minimum over the distortionless set only, so unconstrained w can give
    negative values with vanishing gradients."""
    J = (SIGMA_S * (w.conj() * a_s).sum(-1).abs() ** 2
       + SIGMA_I * (w.conj() * a_i).sum(-1).abs() ** 2
       + SIGMA_N * (w.conj() * w).real.sum(-1))
    return torch.log(J / j_min + 1e-12).mean()

def loss_null_hinge(w, a_s, a_i, target_db):
    p = ((w.conj() * a_i).sum(-1).abs() ** 2 /
         ((w.conj() * a_s).sum(-1).abs() ** 2 + 1e-30))
    return torch.relu(10 * torch.log10(p + 1e-30) - target_db).pow(2).mean()

# ─── train / eval ────────────────────────────────────────────────────────────
def train(model, X, W, epochs=200, batch=256, lr=1e-3):
    g = torch.Generator().manual_seed(42)
    n = len(X); idx = torch.randperm(n, generator=g)
    tr, va, te = idx[:int(.8*n)], idx[int(.8*n):int(.9*n)], idx[int(.9*n):]
    tr_dl = DataLoader(TensorDataset(X[tr], W[tr]), batch_size=batch, shuffle=True)
    va_dl = DataLoader(TensorDataset(X[va], W[va]), batch_size=batch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best = (float('inf'), None)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, wb in tr_dl:
            opt.zero_grad()
            loss = loss_mse(model(xb), wb)          # <-- plain MSE is enough
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_mse(model(xb), wb).item() for xb, wb in va_dl])
        sched.step(vl)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
        if ep % 40 == 0:
            print(f"epoch {ep:3d} | val {vl:.6f}")
    if best[1] is None:
        raise RuntimeError("validation loss was never finite; refusing to evaluate an untrained model")
    model.load_state_dict(best[1])
    return model, te

@torch.no_grad()
def evaluate(model, X, W):
    model.eval()
    ts, ti = X[:, 0].to(torch.float64), X[:, 1].to(torch.float64)
    a_s, a_i = steering_t(ts), steering_t(ti)
    w, w_star = model(X).to(torch.complex128), W.to(torch.complex128)
    resp = lambda v, a: (v.conj() * a).sum(-1)
    nd = 10 * torch.log10(resp(w, a_i).abs()**2 / (resp(w, a_s).abs()**2 + 1e-30) + 1e-30)
    nd_star = 10 * torch.log10(resp(w_star, a_i).abs()**2 / (resp(w_star, a_s).abs()**2 + 1e-30) + 1e-30)
    sinr = lambda v: 10 * torch.log10(
        SIGMA_S * resp(v, a_s).abs()**2 /
        (SIGMA_I * resp(v, a_i).abs()**2 + SIGMA_N * (v.conj()*v).real.sum(-1) + 1e-30))
    gap = sinr(w_star) - sinr(w)
    print(f"null depth NN   median {nd.median():7.1f} dB | MVDR median {nd_star.median():7.1f} dB")
    print(f"SINR gap        median {gap.median():7.3f} dB | p90 {gap.quantile(0.9):7.3f} dB")

if __name__ == '__main__':
    torch.manual_seed(42); np.random.seed(42)
    X, W = make_dataset()
    model = BeamformerNet()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    model, te = train(model, X, W)
    print(f"evaluating on held-out test split ({len(te)} samples)")
    evaluate(model, X[te], W[te])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = DATA_DIR / 'model_nullaligned_recommended.pth'
    torch.save(model.state_dict(), checkpoint)
    print(f"saved checkpoint -> {checkpoint}")
