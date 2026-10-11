"""
Verify the analytic structure of the MVDR solution for the 2-source ULA problem.

Key questions:
  Q1: What is the TRUE MVDR null depth (vs. np.linalg.inv numerical artifact)?
  Q2: Does w_mvdr lie exactly in span{a_s, a_i}?  (physics-informed output basis)
  Q3: How deep is the null of the 'exact-null projector' beamformer?
  Q4: How sensitive is null depth to relative weight perturbations?
  Q5: What SINR gap does a -24 dB null actually cost at INR=30 dB?
"""
import numpy as np

# ─── Parameters (identical to user's Phase 1) ─────────────────
f, c = 2.4e9, 3e8
lam = c / f
d = lam / 2
k = 2 * np.pi / lam            # k*d = pi

SNR_dB, INR_dB = 20, 30
sigma_n = 1.0
sigma_s = sigma_n * 10 ** (SNR_dB / 10)
sigma_i = sigma_n * 10 ** (INR_dB / 10)
N = 8


def steering(theta_deg, N):
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * n * k * d * np.sin(theta))


def mvdr_inv(theta_s, theta_i, N):
    """User's Phase-1 implementation (explicit inverse)."""
    a_s = steering(theta_s, N).reshape(-1, 1)
    a_i = steering(theta_i, N).reshape(-1, 1)
    R = (sigma_s * a_s @ a_s.conj().T
         + sigma_i * a_i @ a_i.conj().T
         + sigma_n * np.eye(N))
    R_inv = np.linalg.inv(R)
    num = R_inv @ a_s
    den = (a_s.conj().T @ R_inv @ a_s)[0, 0]
    return (num / den).flatten()


def mvdr_woodbury(theta_s, theta_i, N):
    """
    Closed-form MVDR via Woodbury. R = G P G^H + sn I with G=[a_s,a_i].
    R^-1 = (1/sn)[I - G M^-1 G^H],  M = sn P^-1 + G^H G   (2x2)
    """
    a_s = steering(theta_s, N)
    a_i = steering(theta_i, N)
    G = np.stack([a_s, a_i], axis=1)                 # N x 2
    Gamma = G.conj().T @ G                            # [[N, rho],[rho*, N]]
    P_inv = np.diag([1 / sigma_s, 1 / sigma_i])
    M = sigma_n * P_inv + Gamma                       # 2x2
    rhs = G.conj().T @ a_s                            # [N, rho*]
    u = np.linalg.solve(M, rhs)                       # 2x1
    v = a_s - G @ u                                   # = sn * R^-1 a_s
    w = v / (a_s.conj() @ v)                          # distortionless
    return w, u


def mvdr_qinv(theta_s, theta_i, N):
    """MVDR via Sherman-Morrison on Q = si a_i a_i^H + sn I (noise+interference only).
       w ∝ Q^-1 a_s:  Q^-1 = (1/sn)(I - a_i a_i^H si/(sn + si N))."""
    a_s = steering(theta_s, N)
    a_i = steering(theta_i, N)
    mu = sigma_i * (a_i.conj() @ a_s) / (sigma_n + sigma_i * N)   # scalar
    v = a_s - mu * a_i
    w = v / (a_s.conj() @ v)
    return w, mu


def null_depth_db(w, theta_i, norm_to_gain=True, theta_s=None):
    a_i = steering(theta_i, N)
    p = abs(w.conj() @ a_i) ** 2
    if norm_to_gain:
        a_s = steering(theta_s, N)
        p /= abs(w.conj() @ a_s) ** 2
    return 10 * np.log10(max(p, 1e-300))


def sinr_db(w, theta_s, theta_i):
    a_s = steering(theta_s, N)
    a_i = steering(theta_i, N)
    sig = abs(w.conj() @ a_s) ** 2 * sigma_s
    itf = abs(w.conj() @ a_i) ** 2 * sigma_i
    noi = sigma_n * np.vdot(w, w).real
    return 10 * np.log10(sig / (itf + noi))


def sinr_in_db():
    return 10 * np.log10(sigma_s / (sigma_i + sigma_n))


print("=" * 78)
print(f"Q1/Q2: MVDR structure at (θs=30°, θi=-40°), N={N}, INR={INR_dB} dB")
print("=" * 78)
th_s, th_i = 30.0, -40.0

w_inv = mvdr_inv(th_s, th_i, N)
w_wb, u = mvdr_woodbury(th_s, th_i, N)
w_q, mu = mvdr_qinv(th_s, th_i, N)

print(f"  ||w_inv - w_woodbury|| / ||w||  = {np.linalg.norm(w_inv - w_wb)/np.linalg.norm(w_wb):.3e}")
print(f"  ||w_inv - w_sherman||   / ||w||  = {np.linalg.norm(w_inv - w_q)/np.linalg.norm(w_wb):.3e}")
print(f"  distortionless check |w^H a_s - 1| (inv): {abs(w_inv.conj() @ steering(th_s,N) - 1):.3e}")

# subspace check: residual of w_inv outside span{a_s,a_i}
a_s = steering(th_s, N); a_i = steering(th_i, N)
G = np.stack([a_s, a_i], axis=1)
coef, *_ = np.linalg.lstsq(G, w_inv, rcond=None)
resid = np.linalg.norm(w_inv - G @ coef) / np.linalg.norm(w_inv)
print(f"  residual of w_mvdr outside span{{a_s,a_i}}: {resid:.3e}   (Q2 answer)")
print(f"  coefficients (alpha, beta) = {coef}")

nd = lambda w, norm=True: null_depth_db(w, th_i, norm_to_gain=norm, theta_s=th_s)
print(f"\n  Null depth  |w^H a_i|^2 / |w^H a_s|^2 :")
print(f"    np.linalg.inv version : {nd(w_inv):8.2f} dB")
print(f"    Woodbury    version   : {nd(w_wb):8.2f} dB")
print(f"    Sherman-M.  version   : {nd(w_q):8.2f} dB")

# exact-null projector: w0 = a_s - a_i (a_i^H a_s)/N,  then scale so w^H a_s = 1
gamma = (a_i.conj() @ a_s) / N
w0 = a_s - gamma * a_i
w_null = w0 / (a_s.conj() @ w0)     # a_s^H w0 is real positive
print(f"    exact-null projector  : {nd(w_null):8.2f} dB   (Q3 answer)")
print(f"    ||w_mvdr||^2={np.vdot(w_inv,w_inv).real:.5f}   ||w_null||^2={np.vdot(w_null,w_null).real:.5f}")
print(f"    ||w_mvdr - w_null||/||w_mvdr|| = {np.linalg.norm(w_inv-w_null)/np.linalg.norm(w_inv):.3e}")
print(f"    SINR: mvdr={sinr_db(w_inv,th_s,th_i):.3f} dB | exact-null={sinr_db(w_null,th_s,th_i):.3f} dB "
      f"| input={sinr_in_db():.3f} dB")

# how far is MVDR from exact null? theory: mu = gamma*(1 - delta), delta = sn/(sn+si*N)
delta = sigma_n / (sigma_n + sigma_i * N)
print(f"\n  theory: mu/gamma = {mu/gamma:.8f}  vs  (1-delta)={1-delta:.8f}   delta={delta:.3e}")
print(f"  theory null power ~ |eta*gamma*delta*N|^2 ... see report")

print("\n" + "=" * 78)
print("Q1b: null depth vs INR (is -93 dB stable? where does it come from?)")
print("=" * 78)
for inr in [0, 10, 20, 30, 40]:
    s_i = 10 ** (inr / 10)
    a_s_ = steering(th_s, N).reshape(-1, 1); a_i_ = steering(th_i, N).reshape(-1, 1)
    R = sigma_s * a_s_ @ a_s_.conj().T + s_i * a_i_ @ a_i_.conj().T + sigma_n * np.eye(N)
    wi = np.linalg.solve(R, a_s_); w = (wi / (a_s_.conj().T @ wi)[0, 0]).flatten()
    # analytic
    mu_ = s_i * (a_i_.conj().T @ a_s_)[0, 0] / (sigma_n + s_i * N)
    va = a_s_.flatten() - mu_ * a_i_.flatten()
    wa = va / (a_s_.flatten().conj() @ va)
    print(f"  INR={inr:3d} dB | null(inv)={null_depth_db(w.flatten(), th_i, True, th_s):8.2f} dB"
          f" | null(analytic)={null_depth_db(wa, th_i, True, th_s):8.2f} dB"
          f" | SINR={sinr_db(w.flatten(), th_s, th_i):7.3f} dB")

print("\n" + "=" * 78)
print("Q4: sensitivity of null depth to relative weight perturbations (float64)")
print("=" * 78)
rng = np.random.default_rng(0)
w_star = w_inv
ref_gain = abs(w_star.conj() @ a_s) ** 2
print(f"  {'rel.err':>10} {'null (dB)':>10} {'SINR(dB)':>10} {'pointing?':>10}")
for eps in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
    nulls, sinrs = [], []
    for _ in range(200):
        dirv = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        dirv /= np.linalg.norm(dirv)
        w_p = w_star + eps * np.linalg.norm(w_star) * dirv
        p_i = abs(w_p.conj() @ a_i) ** 2 / abs(w_p.conj() @ a_s) ** 2
        nulls.append(10 * np.log10(max(p_i, 1e-300)))
        sinrs.append(sinr_db(w_p, th_s, th_i))
    print(f"  {eps:10.0e} {np.mean(nulls):10.2f} {np.mean(sinrs):10.3f}")

print("\n" + "=" * 78)
print("Q5: SINR cost of a 'good-looking' -24 dB null (INR=30 dB)")
print("=" * 78)
for nd_target in [-20, -24, -30, -40, -50, -60, -80, -93]:
    # construct w with exact gain 1 at θs and the given null depth: blend projector with tilt
    amp = 10 ** (nd_target / 20)
    # w = w_null + t*a_i  chosen so |w^H a_i| = amp
    t = amp / N           # |(t a_i)^H a_i| = |t| N
    w_t = w_null + t * a_i
    w_t = w_t / (a_s.conj() @ w_t)
    s = sinr_db(w_t, th_s, th_i)
    gap = sinr_db(w_inv, th_s, th_i) - s
    print(f"  null={nd_target:5d} dB -> SINR={s:7.3f} dB, gap vs MVDR={gap:6.3f} dB")

print("\nDone.")
