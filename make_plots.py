"""
Generate report figures from ablation results + saved test-set predictions.
Run after train_ablation.py.  Figures:
  plots/fig1_sensitivity.png        - null depth vs weight error + model operating points
  plots/fig2_ablation_bars.png      - median null depth / SINR gap per variant
  plots/fig3_patterns.png           - MVDR vs baseline vs recommended beampatterns
  plots/fig4_null_cdf.png           - test-set null-depth CDF
  plots/fig5_mvdr_null_vs_inr.png   - why the null target must depend on INR
  plots/fig6_sinr_cost.png          - SINR gap as a function of null depth
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physics as ph

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N = 8
SNR, INR = 20, 30
SIG_S, SIG_I, SIG_N = 100.0, 1000.0, 1.0
with open(DATA_DIR / 'ablation_results.json') as f:
    res = json.load(f)
TH = np.linspace(-90, 90, 1800)
A = np.stack([ph.steering_np(t, N) for t in TH])          # (1800, N)


def pattern_db(w):
    p = np.abs(np.conj(w) @ A.T) ** 2                    # |w^H a(theta)|^2
    p = np.maximum(p, 1e-16)
    return 10 * np.log10(p / p.max())

def pattern_db_abs(w):
    """|w^H a|^2 in dB, normalized by |w^H a_s| (reported separately)."""
    p = np.abs(np.conj(w) @ A.T) ** 2
    return 10 * np.log10(np.maximum(p, 1e-20))


def plot_ecdf(ax, values, **kwargs):
    """Matplotlib-version-independent empirical CDF helper."""
    values = np.sort(np.asarray(values))
    if values.size == 0:
        return
    y = np.arange(1, values.size + 1) / values.size
    ax.step(values, y, where='post', **kwargs)


# ─── Fig 1: sensitivity curve + operating points ─────────────────────────────
rng = np.random.default_rng(0)
ts, ti = 30.0, -40.0
w_star = ph.mvdr_np(ts, ti, N)
a_s = ph.steering_np(ts, N); a_i = ph.steering_np(ti, N)
eps_list = np.logspace(-6, -1, 25)
nulls, sinrs = [], []
for eps in eps_list:
    nd, sr = [], []
    for _ in range(300):
        d = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        d /= np.linalg.norm(d)
        w = w_star + eps * np.linalg.norm(w_star) * d
        p = abs(w.conj() @ a_i) ** 2 / abs(w.conj() @ a_s) ** 2
        nd.append(10 * np.log10(max(p, 1e-300)))
        sig = abs(w.conj() @ a_s) ** 2 * SIG_S
        itf = abs(w.conj() @ a_i) ** 2 * SIG_I
        noi = SIG_N * np.vdot(w, w).real
        sr.append(10 * np.log10(sig / (itf + noi)))
    nulls.append(np.median(nd)); sinrs.append(np.median(sr))

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.loglog(eps_list, 10 ** (np.array(nulls) / 10), 'k-', lw=2, label='perturbed MVDR')
for nm, mk, col in [('baseline', 'o', 'tab:red'), ('raw16_norm', 's', 'tab:orange'),
                    ('subspace_mse', '^', 'tab:purple'), ('nullaligned_mse', 'D', 'tab:green'),
                    ('nullaligned_full', '*', 'tab:blue')]:
    if nm in res and 'rel_w_err_median' in res[nm]:
        e, nd = res[nm]['rel_w_err_median'], res[nm]['null_median']
        ax1.loglog([e], [10 ** (nd / 10)], mk, color=col, ms=11, label=nm)
ax1.set_xlabel('relative weight error  ||δw|| / ||w||', fontsize=11)
ax1.set_ylabel('|w$^H$ a$_i$|$^2$ / |w$^H$ a$_s$|$^2$', fontsize=11)
ax1.set_title('Null depth is a 1000× accuracy problem\n'
              '(N=8, θs=30°, θi=−40°; dots = measured test-set medians)', fontsize=12)
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(fontsize=9)
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig1_sensitivity.png', dpi=150); plt.close()

# ─── Fig 2: ablation bars ───────────────────────────────────────────────────
names = [k for k in res if 'pointing_median' in res[k]]
mvdr_nd = res['mvdr_ground_truth']['null_median']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nd_vals = [res[n]['null_median'] for n in names]
sg_vals = [res[n]['sinr_gap_median'] for n in names]
colors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:blue'][:len(names)]
axes[0].bar(range(len(names)), nd_vals, color=colors)
axes[0].axhline(mvdr_nd, color='k', ls='--', lw=1.5, label=f'MVDR ({mvdr_nd:.0f} dB)')
axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=25, ha='right')
axes[0].set_ylabel('median null depth (dB)'); axes[0].set_title('Null depth (higher = deeper null)')
axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
axes[1].bar(range(len(names)), sg_vals, color=colors)
axes[1].axhline(0, color='k', ls='--', lw=1.5)
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=25, ha='right')
axes[1].set_ylabel('median SINR gap vs MVDR (dB)'); axes[1].set_title('SINR gap (lower is better)')
axes[1].grid(True, alpha=0.3, axis='y')
plt.suptitle('Ablation on the ULA MVDR dataset (test set, N=8)', fontsize=13)
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig2_ablation_bars.png', dpi=150); plt.close()

# ─── Fig 3: pattern overlays for 3 test cases ───────────────────────────────
d = np.load(DATA_DIR / 'dataset_N8.npz')
X, W = d['X_deg'], d['W']
test_idx = d['idx_test']
cases = [(30, -40), (-25, 45), (5, -55)]
variants_show = ['baseline', 'nullaligned_mse', 'nullaligned_full']
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (cs, ci) in zip(axes, cases):
    j = test_idx[np.argmin(np.abs(X[test_idx, 0] - cs) + np.abs(X[test_idx, 1] - ci))]
    w_t = W[j]
    ax.plot(TH, pattern_db(w_t), 'k-', lw=2, label='MVDR')
    for vn, ls, col in zip(variants_show, ['--', '-.', ':'],
                           ['tab:red', 'tab:green', 'tab:blue']):
        if vn in res:
            w_p = np.load(DATA_DIR / f'pred_{vn}.npz')['w_pred'][np.where(test_idx == j)[0][0]]
            ax.plot(TH, pattern_db(w_p), ls, color=col, lw=1.6, label=vn)
    ax.axvline(cs, color='green', ls=':', lw=1); ax.axvline(ci, color='orange', ls=':', lw=1)
    ax.set_ylim(-80, 3); ax.set_title(f'θs={cs}°, θi={ci}°'); ax.set_xlabel('angle (deg)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
axes[0].set_ylabel('normalized pattern (dB)')
plt.suptitle('Beampattern overlays (test-set samples nearest to requested geometry)', fontsize=13)
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig3_patterns.png', dpi=150); plt.close()

# ─── Fig 4: null-depth CDF ──────────────────────────────────────────────────
def nulls_of(w_pred, idx):
    out = []
    for k, j in enumerate(idx):
        ts_, ti_ = X[j]
        a_s_ = ph.steering_np(ts_, N); a_i_ = ph.steering_np(ti_, N)
        w = w_pred[k]
        out.append(10 * np.log10(max(abs(w.conj() @ a_i_) ** 2 / abs(w.conj() @ a_s_) ** 2, 1e-300)))
    return np.array(out)

fig, ax = plt.subplots(figsize=(9, 5.5))
w_t = W[test_idx]
plot_ecdf(ax, nulls_of(w_t, test_idx), color='k', lw=2, label='MVDR ground truth')
for vn, col in [('baseline', 'tab:red'), ('raw16_norm', 'tab:orange'),
                ('subspace_mse', 'tab:purple'), ('nullaligned_mse', 'tab:green'),
                ('nullaligned_full', 'tab:blue')]:
    if vn in res:
        w_p = np.load(DATA_DIR / f'pred_{vn}.npz')['w_pred']
        plot_ecdf(ax, nulls_of(w_p, test_idx), color=col, lw=1.6, label=vn)
ax.set_xlabel('null depth  10log10(|w$^H$a$_i$|$^2$/|w$^H$a$_s$|$^2$)  [dB]')
ax.set_ylabel('fraction of test set ≤ x')
ax.set_title('Null-depth CDF over the test set (deeper = better)')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig4_null_cdf.png', dpi=150); plt.close()

# ─── Fig 5: MVDR null vs INR (theory) ──────────────────────────────────────
inr_db = np.arange(0, 45, 2.5)
nd_t, nd_u = [], []
for inr in inr_db:
    si = 10 ** (inr / 10)
    a_s_ = ph.steering_np(ts, N); a_i_ = ph.steering_np(ti, N)
    R = SIG_S * np.outer(a_s_, a_s_.conj()) + si * np.outer(a_i_, a_i_.conj()) + SIG_N * np.eye(N)
    v = np.linalg.solve(R, a_s_)
    w = v / (a_s_.conj() @ v)
    p = abs(w.conj() @ a_i_) ** 2 / abs(w.conj() @ a_s_) ** 2
    nd_t.append(10 * np.log10(max(p, 1e-300)))
    w_u = ph.exact_null_np(ts, ti, N)
    pu = abs(w_u.conj() @ a_i_) ** 2 / abs(w_u.conj() @ a_s_) ** 2
    nd_u.append(10 * np.log10(max(pu, 1e-300)))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(inr_db, nd_t, 'r-o', lw=2, label='MVDR (noise-gain tradeoff)')
ax.plot(inr_db, nd_u, 'b--', lw=2, label='exact-null beamformer')
ax.set_xlabel('INR (dB)'); ax.set_ylabel('null depth (dB)')
ax.set_title('MVDR null depth tracks INR (~2 dB per dB)\na fixed −90 dB null target is wrong at low INR', fontsize=12)
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig5_mvdr_null_vs_inr.png', dpi=150); plt.close()

# ─── Fig 6: SINR cost of shallow nulls ─────────────────────────────────────
nds = np.arange(-90, -10, 2.5)
gaps = []
w_null = ph.exact_null_np(ts, ti, N)
for nd in nds:
    amp = 10 ** (nd / 20)
    w = w_null + (amp / N) * a_i
    w = w / (a_s.conj() @ w)
    sig = abs(w.conj() @ a_s) ** 2 * SIG_S
    itf = abs(w.conj() @ a_i) ** 2 * SIG_I
    noi = SIG_N * np.vdot(w, w).real
    s = 10 * np.log10(sig / (itf + noi))
    w_m = w_star
    sigm = abs(w_m.conj() @ a_s) ** 2 * SIG_S
    itfm = abs(w_m.conj() @ a_i) ** 2 * SIG_I
    noim = SIG_N * np.vdot(w_m, w_m).real
    sm = 10 * np.log10(sigm / (itfm + noim))
    gaps.append(sm - s)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nds, gaps, 'b-', lw=2)
ax.set_xlabel('null depth (dB)'); ax.set_ylabel('SINR gap vs MVDR (dB)')
ax.set_title('Every dB of missing null costs SINR (INR=30 dB)\n'
             '≤0.5 dB SINR gap requires null ≈ −50 dB or deeper', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(PLOTS_DIR / 'fig6_sinr_cost.png', dpi=150); plt.close()

print('saved plots/fig1..fig6')
