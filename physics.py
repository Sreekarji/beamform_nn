"""
Differentiable ULA physics in PyTorch + NumPy MVDR reference.

Conventions (match the user's Phase-1 code):
  a_n(theta) = exp(j * n * pi * sin(theta))      (d = lambda/2, k*d = pi)
  MVDR:  w = R^-1 a_s / (a_s^H R^-1 a_s),  R = ss*a_s a_s^H + si*a_i a_i^H + sn*I
  Distortionless constraint:  w^H a_s = 1  <=>  a_s^H w = 1   (a_s^H R^-1 a_s is real)

Key analytic facts used by the physics-informed model:
  F1. w_mvdr lies EXACTLY in span{a_s, a_i}            (2 complex coefficients, not N)
  F2. w_mvdr = eta * (a_s - mu * a_i),  mu = si*(a_i^H a_s)/(sn + si*N)
  F3. 'Exact-null' basis vector u = (a_s - gamma*a_i)/(a_s^H(a_s - gamma*a_i)),
      gamma = (a_i^H a_s)/N,  satisfies u^H a_i = 0 and u^H a_s = 1 EXACTLY.
      => w = g*u + tau*a_i has  |w^H a_i| = |tau|*N : the coordinate tau IS the null.
"""
import numpy as np
import torch
import torch.nn as nn

# Scenario powers (fixed in this project; kept module-level for the loss helpers)
SIGMA_N = 1.0
SIGMA_S = 100.0   # SNR 20 dB
SIGMA_I = 1000.0  # INR 30 dB

# ────────────────────────────── NumPy reference ──────────────────────────────

def steering_np(theta_deg, N):
    n = np.arange(N)
    return np.exp(1j * n * np.pi * np.sin(np.deg2rad(theta_deg)))


def mvdr_np(theta_s, theta_i, N, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N):
    """MVDR weights via the 2x2 Woodbury form (float64, exact for this problem)."""
    a_s = steering_np(theta_s, N)
    a_i = steering_np(theta_i, N)
    G = np.stack([a_s, a_i], axis=1)                  # N x 2
    Gamma = G.conj().T @ G                            # [[N, rho], [rho*, N]]
    M = sn * np.diag([1.0 / ss, 1.0 / si]) + Gamma    # 2x2
    u = np.linalg.solve(M, G.conj().T @ a_s)
    v = a_s - G @ u
    return v / (a_s.conj() @ v)


def j_min_np(theta_s, theta_i, N, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N):
    """Optimal MVDR cost J_min = w^H R w = 1 / (a_s^H R^-1 a_s)."""
    a_s = steering_np(theta_s, N)
    a_i = steering_np(theta_i, N)
    G = np.stack([a_s, a_i], axis=1)
    Gamma = G.conj().T @ G
    M = sn * np.diag([1.0 / ss, 1.0 / si]) + Gamma
    u = np.linalg.solve(M, G.conj().T @ a_s)
    v = a_s - G @ u                       # = sn * R^-1 a_s
    return float(sn / (a_s.conj() @ v).real)   # = 1/(a_s^H R^-1 a_s)


def exact_null_np(theta_s, theta_i, N):
    """Unit-gain, exact-null beamformer u (F3). Null is analytically zero."""
    a_s = steering_np(theta_s, N)
    a_i = steering_np(theta_i, N)
    gamma = (a_i.conj() @ a_s) / N
    w0 = a_s - gamma * a_i
    return w0 / (a_s.conj() @ w0)


# ────────────────────────────── PyTorch physics ──────────────────────────────

def steering_t(theta_deg, N, dtype=None):
    """(B,) degrees -> (B, N) complex steering vectors.
    dtype: element-index precision; defaults to theta_deg.dtype (never mix)."""
    if dtype is None:
        dtype = theta_deg.dtype
    n = torch.arange(N, dtype=dtype, device=theta_deg.device)
    phase = torch.sin(theta_deg * (np.pi / 180.0)).unsqueeze(-1) * (n * np.pi)
    return torch.complex(torch.cos(phase), torch.sin(phase))


def response(w, a):
    """w^H a for batches: w, a are (B, N) complex -> (B,) complex."""
    return (w.conj() * a).sum(-1)


def mvdr_cost(w, a_s, a_i, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N):
    """J(w) = ss|w^H a_s|^2 + si|w^H a_i|^2 + sn||w||^2   (the MVDR objective)."""
    rs = response(w, a_s).abs() ** 2
    ri = response(w, a_i).abs() ** 2
    nrm = (w.conj() * w).real.sum(-1)
    return ss * rs + si * ri + sn * nrm


def null_depth_db(w, a_s, a_i, eps=1e-30):
    """Scale-invariant null depth 10 log10( |w^H a_i|^2 / |w^H a_s|^2 )."""
    p = response(w, a_i).abs() ** 2 / (response(w, a_s).abs() ** 2 + eps)
    return 10.0 * torch.log10(p + eps)


def sinr_db_t(w, a_s, a_i, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N, eps=1e-30):
    sig = ss * response(w, a_s).abs() ** 2
    itf = si * response(w, a_i).abs() ** 2
    noi = sn * (w.conj() * w).real.sum(-1)
    return 10.0 * torch.log10(sig / (itf + noi + eps))


class DistortionlessProjection(nn.Module):
    """Hard-enforce a_s^H w = 1 by RADIAL (multiplicative) normalization:
         w = w_raw / (a_s^H w_raw)
    vs. the additive form w += a_s(1 - a_s^H w_raw)/N.  The additive form injects
    an a_s-component whose response at theta_i is first-order in the gain error,
    which destroys deep nulls; the multiplicative form preserves
    |w^H a_i| / |w^H a_s| exactly.  |den| is clamped away from 0 for stability
    at init (then a tiny additive fix makes the constraint exact again)."""

    def __init__(self, min_den=0.2):
        super().__init__()
        self.min_den = min_den

    def forward(self, w_raw, a_s):
        c = (a_s.conj() * w_raw).sum(-1, keepdim=True)          # a_s^H w_raw
        mag = c.abs()
        # when |c| < min_den, use a DETACHED phase-preserving denominator so the
        # backward pass never divides by |c| -> stable at initialization
        c_clamped = (c / mag.detach().clamp(min=1e-8) * self.min_den).detach()
        c_safe = torch.where(mag < self.min_den, c_clamped, c)
        w = w_raw / c_safe
        # exact constraint even in the clamped regime (correction ~0 in normal use)
        g = (a_s.conj() * w).sum(-1, keepdim=True)
        return w + a_s * (1.0 - g) / a_s.shape[-1]


def synth_subspace(coef, a_s, a_i):
    """Physics-informed output basis (F1): w = alpha*a_s + beta*a_i.
    coef: (B, 2) complex -> w: (B, N) complex."""
    return coef[:, :1] * a_s + coef[:, 1:2] * a_i


def exact_null_basis(a_s, a_i):
    """Unit-gain exact-null vector u (F3) from steering vectors. (B,N) -> (B,N)."""
    N = a_s.shape[-1]
    gamma = (a_i.conj() * a_s).sum(-1, keepdim=True) / N    # a_i^H a_s / N
    w0 = a_s - gamma * a_i
    den = (a_s.conj() * w0).sum(-1, keepdim=True)           # real, > 0
    return w0 / den


def synth_null_aligned(coef, a_s, a_i):
    """Null-aligned basis (F3): w = g*u + tau*a_i, u exact-null unit-gain.
    |w^H a_i| = |tau|*N exactly -> the coordinate tau directly IS the null level.
    coef: (B, 2) complex (g, tau) -> w: (B, N) complex."""
    u = exact_null_basis(a_s, a_i)
    return coef[:, :1] * u + coef[:, 1:2] * a_i


def to_complex(y):
    """(B, 2N) real, split ordering [re0..re_{N-1}, im0..im_{N-1}]
    -> (B, N) complex.  (This matches to_real_split / gen_dataset Y.)"""
    N = y.shape[-1] // 2
    return torch.complex(y[:, :N], y[:, N:])


def to_real_split(w):
    """(B, N) complex -> (B, 2N) real, [re part, im part] concatenated."""
    return torch.cat([w.real, w.imag], dim=-1)


# ────────────────────────────── Loss functions ───────────────────────────────

def loss_weight_mse(w, w_star):
    """Plain MSE in the weight domain (the user's current loss), scaled to the
    same convention as real-valued MSE over 2N reals."""
    diff = torch.cat([(w - w_star).real, (w - w_star).imag], dim=-1)
    return (diff ** 2).mean()


def loss_excess_cost(w, a_s, a_i, j_min, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N):
    """Physics loss: L = log( J(w) / J_min )  >= 0, = 0 iff w = w_mvdr.
    J(w) = ss|w^H a_s|^2 + si|w^H a_i|^2 + sn||w||^2 is the MVDR objective, so
    this single term trades off null depth vs. white-noise gain EXACTLY the way
    MVDR does. j_min = 1/(a_s^H R^-1 a_s) precomputed per sample.

    PRECONDITION: w must satisfy the distortionless constraint a_s^H w = 1
    (e.g. via a synthesis chart or DistortionlessProjection). J_min is the
    minimum of J over that constraint set only; for unconstrained w (e.g. w=0)
    J < J_min makes the log negative with vanishing gradients."""
    J = mvdr_cost(w, a_s, a_i, ss, si, sn)
    return torch.log(J / j_min + 1e-12).mean()


def loss_null_hinge(w, a_s, a_i, target_db=-60.0, eps=1e-30):
    """dB-domain hinge on the interferer response:
       L = mean( ReLU( null_dB - target_db )^2 ).
    Only penalizes nulls that are too shallow; once deep enough the term is 0.
    Gradients flow through log -> well-conditioned even at -80 dB."""
    nd = null_depth_db(w, a_s, a_i, eps)
    return torch.relu(nd - target_db).pow(2).mean()


def loss_gain(w, a_s):
    """Soft backup for the distortionless constraint (only needed if no
    projection layer is used): |a_s^H w - 1|^2."""
    return ((response(w, a_s) - 1.0).abs() ** 2).mean()


def loss_sinr(w, a_s, a_i, ss=SIGMA_S, si=SIGMA_I, sn=SIGMA_N):
    """Negative output SINR (dB). Scale-invariant -> combine with the
    projection layer or a gain loss."""
    return -sinr_db_t(w, a_s, a_i, ss, si, sn).mean()
