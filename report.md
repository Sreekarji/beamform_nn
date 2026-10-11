# Preserving MVDR Null Depth in NN-Predicted Beamforming Weights

**Problem:** MLP 2→128→256→256→128→16 with MSE on raw complex weights gets beam
pointing right (median 0.23°) but nulls collapse from −93 dB (MVDR) to ~−24 dB.

**TL;DR — the fix is the output representation, not the loss or the network.**

1. **Re-parameterize the output in null-aligned coordinates.** MVDR weights lie
   exactly in `span{a_s, a_i}`; in the basis `{u, a_i}` where `u` is the
   unit-gain *exact-null* vector, the single complex output `tau` **is** the null
   amplitude: `|w^H a_i| = |tau|·N` by construction. You are asking the net to
   regress a smooth O(0.01) number instead of cancelling 16 O(0.1) numbers to
   1e-5.
2. **Enforce `w^H a_s = 1` by construction** (it's a 1-line substitution
   `g = 1 − tau·rho`), not with a penalty.
3. **Keep weight-MSE** — with the new coordinates plain MSE is already enough:
   in our ablation the median null goes from **−27.2 dB (your setup) to −92.9 dB**
   and the SINR gap collapses from **11.6 dB to 0.000 dB**; with the full physics
   loss the median null reaches **−96.4 dB vs MVDR's −96.2 dB — preservation to
   0.1 dB**. A physics loss `log J(w)/J_min` and/or an annealed dB-hinge on the
   null (§4) are optional polish and matter mainly when powers vary or `R` is
   estimated.
4. Evaluate with **normalized null depth** `|w^H a_i|²/|w^H a_s|²` and **SINR
   gap** — a −24 dB null at INR 30 dB costs **15 dB of SINR**; you only need
   ≈ −50 dB to be within 0.3 dB of MVDR.
5. **Exclude `|θs−θi| < 3°` pairs** (ill-conditioned labels; §6) — the fix is
   now implemented in the code (`MIN_SEP_DEG = 3.0`).

All results below are from `beamform_nn/` (PyTorch), trained on the exact
dataset grid from your Phase 2 (`θs, θi ∈ {−60..60}`, 14,042 samples after
excluding `|θs−θi| < 3°` — see §6), with
your exact architecture/optimizer recipe where the ablation allows.

---

## 1. Diagnosis: why weight-space MSE cannot reach MVDR nulls

### 1.1 The null is a cancellation, pointing is not

Let `w = w* + δw` (`w*` = MVDR). Then

```
w^H a_i = w*^H a_i + δw^H a_i = 0 + δw^H a_i,     |δw^H a_i| ≤ ||δw||·√N
```

Null depth is first-order in the *absolute* weight error, while pointing error
is only first-order in `||δw||` relative to the mainlobe slope (~13° beamwidth).
That is exactly why you see 0.23° pointing with 24 dB-degraded nulls: a
`||δw||/||w|| ≈ 10⁻²` regression error barely moves the beam peak but raises
the interferer response from 10⁻⁴·⁵ to 10⁻²·⁴ in power.

Measured sensitivity (Gaussian perturbations of the true `w*` at
θs=30°, θi=−40°, N=8 — `verify_physics.py`, fig. 1):

| rel. weight error | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 5e-2 |
|---|---|---|---|---|---|
| null depth (dB) | −95.9 | −90.6 | −71.0 | −51.1 | −37.1 |

Your baseline MLP reaches a test RMSE of ~1e-2 per weight component (val MSE
≈ 1e-4) — squarely in the −20…−30 dB null regime (our exact reproduction of
your setup measures −22.7 dB with close pairs included, −27.2 dB without). To hit −80 dB you need
absolute weight accuracy ~3e-5, i.e. **a 100–1000× smaller regression error
than any generic MLP+MSE run converges to** (float32 Adam simply does not put
1e-5 of signal into a 16-D output when the loss is dominated by the O(0.1)
components). This is not a capacity or learning-rate problem; the loss and
coordinates are misaligned with the metric.

### 1.2 The SINR cost of a "pretty good" null

From the MVDR objective itself (INR = 30 dB, SNR = 20 dB, N = 8):

| null depth | −24 dB | −40 dB | −50 dB | −60 dB | −80 dB |
|---|---|---|---|---|---|
| SINR gap vs MVDR | **15.0 dB** | 2.5 dB | 0.33 dB | 0.03 dB | 0.00 dB |

**Metric recommendation:** track `SINR_gap = SINR_mvdr − SINR_nn` as the
primary KPI and normalized null depth as a secondary one. "Null preservation"
of −50 dB or better is where SINR saturates; chasing −93 dB with MSE is
fighting 4 orders of magnitude for 0.0 dB of measurable benefit.

### 1.3 Two evaluation bugs worth fixing

* You report `10log10|w^H a_i|²` without normalizing by gain. For MVDR
  (`w^H a_s = 1`) that's fine; for NN weights it is not scale-invariant. Use
  `10log10(|w^H a_i|²/|w^H a_s|²)` (or project the NN output first).
* `np.linalg.inv` is not the issue here (we verified the −93…−96 dB null is the
  true analytic MVDR value), but for generalization work prefer
  `np.linalg.solve`/Woodbury and float64 for labels.

### 1.4 The MVDR null is INR-dependent by design

The analytic MVDR null depth for one interferer is (derived in §2):

```
|w^H a_i|² / |w^H a_s|² ≈ (δ·N·|γ|·|η|)²,   δ = σn/(σn + σi·N)
```

which gives **2 dB of null depth per dB of INR** (measured: −37 dB @ INR 0,
−56 @ 10, −76 @ 20, **−96 @ 30**, −116 @ 40 — fig. 5). At low INR, MVDR
deliberately trades null depth for white-noise gain. So: do not hard-code a
−90 dB target unless your dataset is fixed at INR 30 dB; prefer the physics
loss of §4.2, which reproduces this tradeoff automatically.

---

## 2. The physics to bake in (all verified numerically, `verify_physics.py`)

With `R = σs a_s a_s^H + σi a_i a_i^H + σn I` and `w = R⁻¹a_s/(a_s^H R⁻¹ a_s)`:

**F1 — the solution lives in a 2-D complex subspace.**
`w_mvdr ∈ span{a_s, a_i}` exactly (residual 2e-13 in float64 across the whole
angle grid). The16 real outputs of your MLP are fitting a 4-real-dimensional
manifold; the other 12 dimensions are pure error channels that leak into the
null.

**F2 — closed form.** With `ρ = a_s^H a_i`, `γ = ρ*/N`:

```
w_mvdr = η·(a_s − μ·a_i),   μ = σi·ρ* /(σn + σi·N) = γ·(1 − δ),   δ = σn/(σn+σi·N)
```

(the Sherman–Morrison form). This is a good sanity check and a teacher, but the
recommendations below do not depend on it — they survive when `R` is estimated
from snapshots and no closed form exists.

**F3 — exact-null basis.** Define

```
u = (a_s − γ·a_i) / (a_s^H (a_s − γ·a_i))     →   u^H a_i = 0,  u^H a_s = 1  (exactly)
```

Every distortionless beamformer with a null at θi is `α·u`; MVDR is
`g·u + τ·a_i` with `τ = η·γ·δ` (tiny: |τ| ~ 1e-5 at INR 30 dB, and **exactly 0**
whenever `ρ ≈ 0`). The exact-null beamformer `u` itself has identical SINR to
MVDR at INR 30 dB and a deeper-than-needed null (fig. 5).

**Consequence for the network:** write

```
w = g·u + τ·a_i,   g = 1 − τ·ρ        (⇒ a_s^H w = 1 exactly)
⇒  w^H a_i = conj(τ)·N  and  |w^H a_i|²/|w^H a_s|² = |τ|²·N² / |1|²
```

The single complex output `tau` *is* the null amplitude, and the constraint is
identically satisfied. `|τ|` is naturally scaled by `δ = σn/(σn+σi·N)` so the
regression target is O(0.01–30) across the dataset — ordinary regression
accuracy (1e-3) now maps to null floors around −120 dB instead of −24 dB.

Compare the three charts for how the null is represented:

| chart | outputs | null condition | needs |
|---|---|---|---|
| raw `w` (16 reals) | 16 | 16 reals cancel to 1e-5 | impossible with MSE |
| `{a_s, a_i}`: `w = ((1−βρ)/N)·a_s + β·a_i` | 2 | one complex cancellation | hard |
| `{u, a_i}`: `w = (1−τρ)·u + τ·a_i` | 2 | **none — `τ` is the null** | easy |

---

## 3. Recommended model (PyTorch)

The complete implementation is `beamform_nn/models.py` + `physics.py`. Core
pieces:

```python
# steering vectors from angles — part of the model, not learned
def steering_t(theta_deg, N):                       # (B,) deg -> (B,N) complex
    n = torch.arange(N, dtype=theta_deg.dtype, device=theta_deg.device)
    phase = torch.sin(theta_deg * (np.pi/180)).unsqueeze(-1) * (n * np.pi)
    return torch.complex(torch.cos(phase), torch.sin(phase))

DELTA = sigma_n / (sigma_n + sigma_i * N)           # natural scale of tau (~1.25e-4)

class BeamNet(nn.Module):
    """2 -> 128 -> 256 -> 256 -> 128 -> 2  (+ physics layer)."""
    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2))                      # (tau_re, tau_im)
        self.out_scale = 0.05 * DELTA

    def forward(self, ang_deg):                     # (B,2) degrees
        ts, ti = ang_deg[:, 0], ang_deg[:, 1]
        out = self.net(torch.stack([ts/60.0, ti/60.0], dim=-1))
        a_s = steering_t(ts, self.N)
        a_i = steering_t(ti, self.N)
        rho = (a_s.conj() * a_i).sum(-1)            # a_s^H a_i
        tau = self.out_scale * torch.complex(out[:, 0], out[:, 1])
        g   = 1.0 - tau * rho                       # exact distortionless constraint
        u   = exact_null_basis(a_s, a_i)            # u^H a_i = 0, u^H a_s = 1
        return g.unsqueeze(-1) * u + tau.unsqueeze(-1) * a_i

def exact_null_basis(a_s, a_i):
    N = a_s.shape[-1]
    gamma = (a_i.conj() * a_s).sum(-1, keepdim=True) / N     # a_i^H a_s / N
    w0 = a_s - gamma * a_i
    den = (a_s.conj() * w0).sum(-1, keepdim=True)            # real > 0
    return w0 / den
```

Why this works:

* **12 of 16 output dimensions removed.** The network only fits `tau` (2
  reals), a smooth function of `(θs, θi)` (and of `sinθs − sinθi` in u-space).
* **The null cannot drift.** Whatever the net outputs,
  `|w^H a_i| = |tau|·N` up to float32 rounding — there is no cancellation of
  independent errors to accumulate.
* **No divisions by network activations** anywhere; the only division is by the
  analytic `den` (bounded: `N² − |ρ|² ≥ N² − N² = 0` — avoid training pairs with
  `θs ≈ θi`, see §6).
* **Distortionless gain exactly**, so pointing is anchored and null depth is
  automatically normalized.

For reference, the intermediate chart (`subspace`: predict `β`, set
`α = (1−βρ)/N`) is implemented too — it isolates how much of the gain is the
subspace constraint vs. the null-aligned basis (§5).

### Optional features

The map `(θs,θi) → w` is smoothest in **u = sinθ space** (the array factor is a
Dirichlet kernel in `Δu = sinθs − sinθi`). Adding `[sinθs, sinθi, Δu]` to the
inputs (`feats='full'` in `models.py`) is a cheap conditioning win, especially
near endfire. Not required for the ablation (kept `ang2` everywhere for
fairness).

---

## 4. Losses

### 4.1 Weight MSE — keep it (in the new coordinates)

With the null-aligned chart, plain MSE on `w` delivers **−92.9 dB median nulls
and a 0.000 dB SINR gap** (vs −27.2 dB / 11.6 dB for the baseline with the same
loss and architecture). The reason: in these coordinates the MSE optimum has
`τ = τ*`, and the *typical* `|τ*|` is 1e-5 — output noise of 1e-3 in
`τ/δ` units contributes `|δw^H a_i| ≈ 1e-3·δ·N ≈ 1e-6`, i.e. a −120 dB floor.
The loss never had the wrong optimum; it had the wrong coordinates. The
measured weight-space error drops from 7.4e-2 to 2.1e-5 relative — the 3–4
orders of magnitude §1.1 said were needed.

### 4.2 Physics loss — the MVDR objective (recommended when you can build `R`)

```python
def mvdr_cost(w, a_s, a_i, ss, si, sn):
    return (ss * (w.conj()*a_s).sum(-1).abs()**2
          + si * (w.conj()*a_i).sum(-1).abs()**2
          + sn * (w.conj()*w).real.sum(-1))

loss = torch.log(mvdr_cost(w, a_s, a_i, ss, si, sn) / j_min).mean()   # j_min = 1/(a_s^H R^-1 a_s)
```

* `J(w) = σs|w^H a_s|² + σi|w^H a_i|² + σn||w||²` is *the* MVDR objective, so
  this loss trades off null depth against white-noise gain exactly as MVDR does
  (and automatically re-targets the null when INR changes — §1.4).
* `log(·)` compresses the 1e-8…1e2 dynamic range; `J/J_min − 1 = 0` iff
  `w = w_mvdr`. `j_min` is available analytically (Woodbury 2×2) or via
  `torch.linalg.solve` on the small `R` (differentiable, float64).
* Unlike a pattern-domain loss it costs O(N) per sample, no angle grid.

### 4.3 Annealed dB-hinge on the null (optional, for last-mile depth)

```python
nd = 10*torch.log10( |w^H a_i|^2 / |w^H a_s|^2 + 1e-30)      # scale-invariant
loss_hinge = torch.relu(nd - target_db).pow(2).mean()
# target_db anneals: -25 dB at start -> -60 dB by epoch 80
```

* Hinge, not squared-error: once the null is deep enough you stop paying for
  the last 20 dB (which is where MSE-on-dB becomes ill-behaved).
* Annealing matters: pushing for −60 dB from epoch 0 starves the mainlobe/noise
  terms early in training.

### 4.4 Full recommended objective

```
L = log(J(w)/J_min)  +  0.01·MSE(w, w*)  +  0.02·hinge_dB(annealed)
```

(`loss='full'` in `train_ablation.py`.) The MSE term is a mild anchor; the
physics term dominates. Per the ablation, for this fixed-power problem
**plain MSE in the null-aligned chart is already sufficient** (−92.9 dB /
0.000 dB gap) — adopt the composite objective when you vary powers, estimate
`R` from snapshots, or want exact MVDR-null matching (median preservation
0.1 dB) rather than uniformly deep nulls (§9).

### 4.5 Pattern-domain loss (your option 1) — when to use it

`L = mean_θ (dB(|w^H a(θ)|²) − dB(|w*^H a(θ)|²))²` over a θ-grid is the right
tool when you must control *sidelobe shape* or have many interferers and no
trustworthy `R`. For this problem it is strictly worse than §4.2: 100× more
compute per step, and the dB-MSE over deep-sidelobe regions creates artificial
gradient mass at −80 dB where nothing matters. If you use it, mask the loss to
`[θi±2°]` ∪ mainlobe ∪ sidelobes above −40 dB.

---

## 5. Ablation (test set, 1405 samples; all variants share data split, hidden
sizes `(128,256,256,128)`, ReLU, Adam 1e-3, ReduceLROnPlateau, 200 epochs —
your exact recipe)

```
variant                point°  null_med  null_worst  preserv   sinr_gap  sinr_p90   rel_err
-----------------------------------------------------------------------------------------
mvdr_ground_truth        --      -96.2 dB   -76.6 dB     --      0.000 dB     --        --
exact_null_analytic      --     -300.0 dB  -299.9 dB     --      0.000 dB     --        --
baseline (your setup)   0.176   -27.2 dB   -21.8 dB    70.8 dB  11.574 dB  16.854 dB  7.36e-02
raw16_norm              0.121   -33.4 dB   -23.1 dB    65.2 dB   6.252 dB  16.013 dB  3.33e-02
subspace_mse            0.122   -26.7 dB   -22.1 dB    70.7 dB  12.547 dB  16.962 dB  4.62e-02
subspace_cost           0.169   -47.9 dB   -39.7 dB    47.8 dB   0.429 dB   2.562 dB  3.97e-03
nullaligned_mse         0.171   -92.9 dB   -92.1 dB     3.2 dB   0.000 dB   0.000 dB  2.11e-05
nullaligned_full        0.171   -96.4 dB   -76.4 dB     0.1 dB   0.000 dB   0.000 dB  1.32e-05
```

(`preserv` = median of `null_nn − null_mvdr`, 0 dB = perfect; `sinr_p90` = 90th
percentile of the per-sample SINR gap; `rel_err` = median `||w−w*||/||w*||`.
Dataset: 14,042 samples (your Phase-2 grid minus `|θs−θi|<3°`); all variants:
your hidden sizes `(128,256,256,128)`, ReLU, inputs `θs/60, θi/60`, Adam 1e-3 +
ReduceLROnPlateau, 200 epochs, batch 256, identical split. With close pairs
*included*, baseline measures −22.7 dB / 15.3 dB — your reported −24 dB regime
either way.)

What each row isolates:

* **baseline → raw16_norm:** a hard distortionless constraint alone gives a
  real but modest improvement (−27.2 → −33.4 dB, SINR gap 11.6 → 6.3 dB) —
  and it only trains cleanly once close-angle pairs are excluded (with them,
  the radial normalization's denominator is ill-conditioned and it used to
  stall at val 1e-2). Necessary, ~60 dB short of sufficient.
* **baseline → subspace_mse:** just changing to the `span{a_s,a_i}` chart
  halves the weight error and improves pointing (0.176→0.122°) but the null
  stays ~−27 dB — because the null condition in this chart is still a
  cancellation (`β ≈ −ρ̄/(N²−|ρ|²)`).
* **subspace_mse → subspace_cost:** replacing MSE with the physics loss on the
  same chart cuts the SINR gap 12.5→0.4 dB and deepens the null to −48 dB.
  The loss matters — but the chart conditioning caps the null.
* **subspace_mse → nullaligned_mse:** same MSE loss, same architecture, only
  the output chart changes: **null −26.7 → −92.9 dB, SINR gap 12.5 → 0.000 dB**.
  This is the whole story: put the output in coordinates where the null is a
  *coordinate*, not a cancellation.
* **nullaligned_mse → nullaligned_full:** the physics loss + annealed hinge
  pins the median to MVDR almost exactly (**preservation 0.1 dB**) by letting
  the shallowest MVDR nulls stay shallow (its `null_worst` follows MVDR's
  −76.4 dB p90), while pure MSE uniformly overshoots to τ≈0 for those samples
  (uniform −92 dB). Pick `mse` for maximum guaranteed depth, `full` for exact
  MVDR matching. Both give 0.000 dB SINR gap.

**Honest caveat (worth understanding):** for this *exact* problem — fixed
SNR/INR, known steering vectors, single interferer — MVDR has a closed form
(§2, F2), and the null-aligned chart hands almost all of it to the physics
layer: the network's entire remaining job is the 2-real function `τ(θs,θi)`.
The baseline network was learning 16 outputs where physics already knew 14 of
the answer. That is exactly why the reparameterization is dramatic here — and
it is also why, if you only ever need this one fixed scenario, an analytic
layer (`exact_null_analytic` row, SINR gap 0.000, null −300 dB) beats every NN.
The NN earns its keep when powers vary, `R` is estimated from snapshots,
interferers are unknown, or the array is perturbed — and in all of those cases
the *same* chart + physics loss still applies with the network predicting the
parts physics cannot (§9).

**Figures** (`beamform_nn/plots/`):

* `fig1_sensitivity.png` — null depth vs. weight error, with each variant's
  measured operating point. The baseline sits exactly on the perturbation
  curve: it is not "broken", it is exactly as accurate as its loss demands.
* `fig2_ablation_bars.png` — median null depth and SINR gap per variant.
* `fig3_patterns.png` — MVDR vs. baseline vs. recommended beampatterns.
* `fig4_null_cdf.png` — null-depth CDFs over the test set.
* `fig5_mvdr_null_vs_inr.png` — the INR-dependent MVDR null (why fixed targets
  are wrong).
* `fig6_sinr_cost.png` — SINR gap vs. null depth.

---

## 6. Data & normalization notes

* **Labels in float64** via the 2×2 Woodbury solve (identical to
  `np.linalg.inv` at 1e-13, faster, and it exposes `(α,β)`/`τ` for free).
* **Close-angle pairs (`|θs − θi| < 3°`) are excluded by default**
  (`MIN_SEP_DEG = 3.0` in `gen_dataset.py` / `recommended_solution.py`;
  removes 478 of 14,520 pairs). They are doubly ill-conditioned: as
  `N² − |ρ|² → 0` the labels blow up — `|τ|/δ` is 7.9 at 1° mid-array and ≈ 30
  at 1° endfire, i.e. required net outputs of 160–600 vs O(0.3) typical — and
  `u` itself loses float32 precision (its denominator is `(N²−|ρ|²)/N = 0.125`
  at 1°). Weight-space MSE won't diverge (the `g·u` term dominates), but those
  samples get poor predicted nulls and dominate the loss landscape for no
  benefit. If you must cover them, predict `tau_db = 20log10|τ|` with a bounded
  head instead of linear `τ`.
* **Per-sample normalization of raw `w` targets is pointless in the new chart**
  (`g` is O(1), `τ/δ` is well-scaled); if you keep raw-16 outputs, z-score each
  of the 2N columns and re-scale at inference.
* **Input:** `θ/60` is fine; `[θs/60, θi/60, sinθs, sinθi, Δu]` is better
  (u-space conditioning, endfire behavior).

---

## 7. What about the alternatives you listed?

* **Complex-valued NNs (CVNNs).** A full complex-valued MLP
  (Trabelsi-style complex linear layers) mainly helps optimization of phase
  structure inside the network. It does not fix the null problem, because the
  null sensitivity lives in the *output parameterization*, not the hidden
  layers. The synthesis layer above already does the complex-valued part that
  matters. CVNN bodies are a reasonable optional upgrade for larger problems;
  they are not the fix here.
* **Predicting coefficients in another domain.** That is precisely what §3 is:
  the network predicts `τ` (null-coordinate) instead of `w`. Two other viable
  domains: (a) **root/zero placement** — the array polynomial
  `B(z) = Σ w_n* z^n` has a zero at `z = e^{jπ sinθi}` for an exact null; predict
  zero locations + gain and synthesize `w` from polynomial coefficients. Exact
  nulls by construction, but the map to MVDR's finite null needs an extra tilt
  parameter (= `τ`). (b) **beampattern samples → w** via least squares: useful
  in pattern-synthesis literature, strictly more work than (a) for no gain here.
* **Physics-informed NN / unrolled solvers.** If you later move to
  snapshot-based `R` (no closed form), the scalable version of this recipe is:
  input = (angles or `R` itself), network predicts the *small* stuff (e.g. a
  diagonal-loading factor or the 2×2 `M` matrix of the Woodbury form), and a
  differentiable `torch.linalg.solve` layer produces `w`. Deep-unfolded
  WMMSE/matrix-inversion networks are the published version of this idea (see
  §8). For your current problem the closed-form physics layer is the unrolled
  solver already — 2 layers, no learning needed for the linear algebra.
* **Just make the MLP deeper/wider.** No. The baseline is already at its loss's
  accuracy floor (val MSE plateaus at ~3e-4 by epoch 120).

---

## 8. Published work (mapping to your question 5)

* **Direct AoA→weights NNs (your current paradigm), trained with MSE on
  weights** — e.g. A-DNNABF (Transformer/DARTS AoA→MVDR weights,
  [link](https://jit.ndhu.edu.tw/article/download/3215/3241)); LSTM/RNN
  null-steering beamformers; CNN-MVDR for radar jamming
  ([null-space beamforming + DL, 2025](https://www.researchgate.net/publication/392677390)).
  These papers mostly report weight-MSE/pointing error and coarse nulls — the
  deep-null preservation problem you hit is real and under-discussed there.
* **Speech enhancement literature — the warning is old.** The CHiME-2016 study
  ([link](https://www.chimechallenge.org/workshops/chime2016/papers/CHiME_2016_paper_17.pdf))
  found networks that *directly predict beamforming weights* underperform
  mask-estimated MVDR, and suggested explicit gain modeling + MVDR supervision.
  ADL-MVDR ([arXiv:2012.13442](https://arxiv.org/pdf/2012.13442)) replaces the
  matrix inverse and eigenvector step with GRUs predicting frame-level weights
  from *estimated covariances*, and explicitly notes that NNs directly learning
  weights "are not successful since noise information is not explicitly
  considered" — i.e., bake the physics in.
* **Deep beamforming networks** (Xiong et al., ICASSP 2016,
  [link](https://home.ttic.edu/~llu/pdf/icassp2016_deep_beamforming.pdf)):
  predict frequency-domain beamformer weights from GCC features (4112 real
  outputs!) — same output-parameterization pain at larger scale.
* **Model-driven / deep-unfolding** — unfolded WMMSE for precoding
  (Hu et al., IEEE TWC 2021; summarized in
  [Guo & Yang](https://arxiv.org/pdf/2212.00860)), GNN-unfolded solvers,
  two-stage CNN robust adaptive beamforming
  ([MDPI Electronics 2023](https://www.mdpi.com/2079-9292/12/12/2751)):
  learn the algorithm's small parameters, solve the linear algebra
  analytically. Our `τ`-chart is the minimal instance of this philosophy.
* **CVNN beamforming** — complex-valued nets mapping covariance→weights have
  been benchmarked recently
  ([ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2666827025001252));
  gains are modest vs. structured output layers.
* **Classical exact-null steering** (the `u` basis) is textbook null-steering:
  `w ∝ (I − a_i a_i^H/||a_i||²)a_s` with the distortionless fix — see any
  robust adaptive beamforming treatment (e.g. Van Trees). Our contribution is
  just to put the NN's *outputs* in that basis.

---

## 9. Generalizing beyond the toy problem

| extension | what changes |
|---|---|
| **K interferers** | `w ∈ span{a_s, a_i1..a_iK}` (F1 generalizes). Build `u` by projecting `a_s` onto the orthogonal complement of `[a_i1..a_iK]` (QR/`torch.linalg.lstsq` on the K steering vectors — differentiable). Predict K complex tilt coordinates `τ_k`. Constraint `a_s^H w = 1` still a 1-line substitution. |
| **INR/SNR variation** | feed powers (or their logs) as inputs; use the physics loss (§4.2) so the null target tracks INR automatically. `δ`-scaling generalizes to `σn/(σn+σi·N)`. |
| **Snapshot-based `R`** | input `R` (e.g. vec of upper triangle) instead of angles; keep the `u`/projection structure with the *estimated* steering vector; predict only the small correction (`τ`, or diagonal loading). This is the ADL-MVDR/deep-unfolding regime. |
| **Wideband** | per-bin `τ(f)` prediction with the same chart; smoothness across `f` is easy to regularize. |
| **Close angles / grating** | exclude or handle `N² − |ρ|² → 0` (§6); predict `tau_db` with a bounded output head there. |

---

## 11. Implementation notes (code audit, verified and fixed)

An independent audit of the code found the following; all items were verified
against the source and fixed in this package:

| item | status | fix |
|---|---|---|
| `physics.steering_t` had `dtype=torch.float32` default, ignoring `theta_deg.dtype` | real (minor) — silent 8.5e-7 phase-precision loss when called with float64, fragile on older torch | `dtype=None` → defaults to `theta_deg.dtype` |
| `recommended_solution.py` evaluated on the full dataset (incl. train) | real (methodology) — numbers were not inflated (−89.2 full-set vs −89.5 test-set) but reporting was inconsistent | `train()` returns the test split; `evaluate` runs on it only |
| close-angle pairs ill-conditioned (`\|τ\|/δ` up to ~30, `u` denominator → 0.125) | real | `MIN_SEP_DEG = 3.0` exclusion implemented in both dataset generators (removes 478 pairs) |
| `loss_excess_cost`/`loss_physics` valid only for distortionless `w` (else `log` can go negative) | real — all call sites already enforced the constraint | precondition documented in both docstrings |
| `gen_dataset.py` verified only the `(α,β)` reconstruction | real | both decompositions now checked (`err_ab`, `err_gt` ≈ 1.6e-14) |
| `OUT_SCALE` factored differently across files | real (clarity) | unified `TAU_SCALE = 0.05·DELTA`, `BETA_SCALE = 0.05` |
| comment `# exact: w^H a_s = 1` vs code computing `a_s^H w` | nit | comment now states both (they are complex conjugates of each other = 1) |
| `W` trained in complex64 | documented | comment in `make_dataset`: float32 floor ≈ −150 dB ≫ any −96 dB target |

Physics claims (F1/F2/F3, τ scaling, 2 dB/dB INR slope, SINR-cost table,
exact-null SINR equivalence) were independently re-verified: all correct as
written. One nuance worth internalizing: at Xavier init the null-aligned model
starts at ≈ −106 dB null (τ_init ≈ 0.1·TAU_SCALE) — *already deeper than the
MVDR target*. Training is not "learning to null"; it is learning to place τ at
MVDR's finite, INR-dependent value.

## 10. Files

```
beamform_nn/
├── recommended_solution.py  # ★ START HERE — self-contained PyTorch solution
├── physics.py            # differentiable steering / basis / losses + numpy MVDR
├── models.py             # BeamNet: raw16 | subspace | nullaligned  (ablation)
├── gen_dataset.py        # Phase-2-compatible dataset + (α,β), (g,τ), J_min labels
├── verify_physics.py     # F1/F2/F3 checks + sensitivity + SINR-cost tables
├── train_ablation.py     # all 6 variants, identical recipe, metrics + JSON
├── make_plots.py         # fig1..fig6
├── report.md             # this document
├── README.md             # run instructions
├── data/                 # dataset_N8.npz, ablation_results.json, models, preds
└── plots/                # fig1..fig6
```

Quick start: `python3 recommended_solution.py` (trains in ~80 s on CPU and
prints the two KPIs on the held-out test split). Expected output:

```
evaluating on held-out test split (1405 samples)
null depth NN   median   -93.6 dB | MVDR median   -96.1 dB
SINR gap        median    0.000 dB | p90    0.000 dB
```

**One-line takeaway:** stop asking a 16-output regressor to cancel 12 spurious
degrees of freedom to 1e-5; hand it the two coordinates physics says exist,
make the null one of them, and ordinary regression accuracy becomes more than
enough.
