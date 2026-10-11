# Physics-Informed Neural Beamforming for MVDR Null Preservation

A reproducible research codebase for learning narrowband minimum-variance distortionless response (MVDR) beamformer weights for an eight-element uniform linear array (ULA), while preserving the deep interference null that ordinary weight regression tends to destroy.

The project is deliberately small enough to audit end to end and is intended as a university project submission, portfolio example, and starting point for full-wave Ansys HFSS validation.

## 1. Problem statement

Consider an `N = 8` ULA at `f0 = 2.4 GHz` with element spacing `d = lambda/2`, one desired source at `theta_s`, and one interferer at `theta_i`. With the array convention used throughout this repository,

```text
a_n(theta) = exp(j n pi sin(theta)),  n = 0, ..., N - 1
```

The covariance model is

```text
R = sigma_s a_s a_s^H + sigma_i a_i a_i^H + sigma_n I,

w_MVDR = R^-1 a_s / (a_s^H R^-1 a_s),
```

with SNR = 20 dB, INR = 30 dB, and `sigma_n = 1`. The distortionless constraint is `w^H a_s = 1`.

The learning task is to map `(theta_s, theta_i)` to complex beamforming weights. A conventional MLP can point the main beam correctly while missing the interferer by tens of decibels: a null is produced by cancellation of many weight components, so small component-wise errors create a large relative error at the null.

## 2. Approach: MVDR plus physics-informed learning

The repository compares a raw-weight MLP with two physics-informed output charts.

1. **Reference physics.** `physics.py` computes float64 NumPy MVDR labels with a 2-by-2 Woodbury solve rather than explicitly inverting the full covariance matrix.
2. **Subspace chart.** For this two-source covariance, MVDR lies exactly in `span{a_s, a_i}`. The network predicts one complex coefficient rather than 16 real weight components.
3. **Null-aligned chart.** Define the unit-gain exact-null vector

   ```text
   gamma = (a_i^H a_s) / N
   u = (a_s - gamma a_i) / (a_s^H (a_s - gamma a_i))
   ```

   and synthesize

   ```text
   w = g u + tau a_i,
   g = 1 - tau rho,
   rho = a_s^H a_i.
   ```

   This enforces `a_s^H w = 1` analytically and gives `|w^H a_i| = |tau| N`. The network therefore predicts the physically meaningful null coordinate `tau`, rather than learning a fragile cancellation among 16 outputs.

The recommended model is a `2 -> 128 -> 256 -> 256 -> 128 -> 2` ReLU MLP. Inputs are `[theta_s / 60, theta_i / 60]`; the final two real outputs are interpreted as the real and imaginary parts of the scaled complex `tau`. The synthesis layer supplies the steering vectors and null-aligned basis.

## 3. Key results

The completed run reports the following held-out evaluation results:

| Variant | Median normalized null depth | Median SINR gap to MVDR |
|---|---:|---:|
| `baseline` | **−26.8 dB** | **11.999 dB** |
| `nullaligned_mse` | **−92.9 dB** | **0.000 dB** |
| `nullaligned_full` | **−96.4 dB** | **0.000 dB** |
| `recommended_solution` | **−93.6 dB** | **0.000 dB** |

Null depth is reported as

```text
10 log10( |w^H a_i|^2 / |w^H a_s|^2 ),
```

and the SINR gap is `SINR_MVDR - SINR_NN`; lower SINR gap is better.

The tracked artifact in this checkout uses the stricter `|theta_s - theta_i| >= 3 deg` dataset filter. Its saved `data/ablation_results.json` gives the following reproducible reference points:

| Variant | Median null depth | Median SINR gap |
|---|---:|---:|
| MVDR ground truth | −96.2 dB | 0.000 dB |
| `baseline` | −26.8 dB | 11.999 dB |
| `nullaligned_mse` | −92.9 dB | 0.000 dB to three decimals |
| `nullaligned_full` | −96.4 dB | 0.000 dB to three decimals |

The table above records the completed ablation run in `data/ablation_results.json`. The standalone `recommended_solution.py` run is recorded separately above because its result is not an entry in that JSON file.

In this run, `raw16_norm` worsened both key metrics relative to `baseline`: its median null was −21.5 dB versus −26.8 dB, and its median SINR gap was 17.556 dB versus 11.999 dB. Hard distortionless projection alone was therefore not an improvement in this experiment.

## 4. Repository contents

```text
beamform_nn/
├── physics.py              # NumPy MVDR and differentiable PyTorch physics/losses
├── models.py               # raw16, subspace, and nullaligned BeamNet charts
├── gen_dataset.py          # deterministic angle grid, labels, and data split
├── train_ablation.py       # six variants, training, evaluation, and saved metrics
├── recommended_solution.py # self-contained null-aligned training example
├── verify_physics.py       # Woodbury/Sherman–Morrison and sensitivity checks
├── make_plots.py           # regenerates six analysis figures
├── hfss_overlay.py         # imports HFSS exports and makes EM/analytical overlays
├── requirements.txt        # Python dependencies
├── data/                   # tracked dataset, checkpoints, predictions, and metrics
└── plots/                  # tracked figures; HFSS outputs can go in plots/hfss/
```

## 5. Requirements and installation

- Python 3.9 or newer
- NumPy
- PyTorch (the CPU build is sufficient)
- Matplotlib

Recommended installation:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

No GPU, MATLAB, HFSS, SciPy, or external dataset is required for the analytical experiments. HFSS is required only for the optional physical-validation phase.

## 6. Reproducing the analytical results

The completed results are already recorded in the repository artifacts. The following commands are optional reproducibility steps; the scripts resolve data and output paths relative to the repository rather than the process working directory.

### Fast path

```bash
python recommended_solution.py
```

This trains the recommended model for 200 epochs on CPU, evaluates only the held-out test split, and writes `data/model_nullaligned_recommended.pth`.

### Complete ablation and figures

```bash
python verify_physics.py
python gen_dataset.py
python train_ablation.py
python make_plots.py
```

Expected dataset size is 14,042 examples: an integer grid from −60° through +60° with pairs closer than 3° removed. The full ablation trains six variants for 200 epochs each and writes checkpoints, test predictions, and `data/ablation_results.json`. To train selected variants:

```bash
python train_ablation.py baseline nullaligned_mse nullaligned_full
```

The checked-in artifacts allow `make_plots.py` to regenerate the figures without retraining. Generated outputs are intentionally not treated as new source data.

## 7. Figures

| Figure | Interpretation |
|---|---|
| [Sensitivity](plots/fig1_sensitivity.png) | Null depth as a function of relative weight error. |
| [Ablation](plots/fig2_ablation_bars.png) | Null-depth and SINR-gap comparison across output charts and losses. |
| [Beampatterns](plots/fig3_patterns.png) | MVDR, raw MLP, and null-aligned overlays for three geometries. |
| [Null CDF](plots/fig4_null_cdf.png) | Test-set distribution of normalized null depth. |
| [Null versus INR](plots/fig5_mvdr_null_vs_inr.png) | Why a fixed −90 dB null target is not physically appropriate at every INR. |
| [SINR cost](plots/fig6_sinr_cost.png) | Operational cost of a shallow interference null. |

## 8. HFSS physical-validation phase

The purpose of the next phase is not to claim that a point-source ULA is already a full-wave model. It is to quantify agreement and then explain the residual differences caused by element patterns, mutual coupling, port normalization, finite ground/materials, and the HFSS angular convention.

### Recommended HFSS geometry

Use a parameterized free-space ULA as the first controlled experiment:

- `N = 8` identical center-fed half-wave wire dipoles, parallel to `z`.
- Array axis along `x`, element index increasing with `x`.
- Center-to-center spacing `d = lambda/2`.
- At 2.4 GHz, the code's rounded constants give `lambda = 125 mm` and `d = 62.5 mm`; using the exact speed of light gives approximately `124.91 mm` and `62.46 mm`. Use one convention consistently in both HFSS and Python.
- Give each element its own 50-ohm lumped/terminal port. Preserve a documented element/port order from 0 to 7.
- Enclose the array in an air region with a radiation boundary or PML at least `lambda/4` from the radiating structure; `lambda/2` is a safer starting margin.

A half-wave dipole is intentionally more realistic than the isotropic steering-vector model. A disagreement in the element factor is expected; the first comparison should focus on beam/null angle, relative null depth, and normalized pattern shape.

### Exact HFSS workflow

1. Create a Driven Modal or Driven Terminal HFSS design and set `f0 = 2.4 GHz` plus variables for `lambda`, `d`, dipole length, radius, and array center.
2. Draw and duplicate the eight elements at `x_n = (n - 3.5)d`. Create one independent port at each feed gap. Confirm the port numbering in the design tree and export the S-matrix once as a sanity check.
3. Add the air box and radiation/PML boundary. Add an adaptive solution setup at 2.4 GHz, refine until the far-field/main-beam and S-parameter changes are acceptably small, and add a frequency sweep only if a frequency study is desired.
4. Add **Radiation → Insert Far Field Setup → Infinite Sphere**. For a signed Python-angle cut, export the x-z plane: use `Phi = 0°` for the +x half-plane and `Phi = 180°` for the −x half-plane. Use a one-degree angular step initially. Record whether HFSS `Theta` is spherical polar angle or a signed report variable.
5. Generate a far-field report with **Results → Create Far Fields Report**. For direct case comparisons, use the same quantity for every case, preferably `dB(RealizedGainTotal)` or `dB(GainTotal)`. Normalize all curves in Python; do not compare arbitrary absolute dB offsets.
6. Compute the five Python MVDR weight vectors below. Because the Python quantity is `w^H a(theta)`, set the HFSS transmit/source voltage vector to a common-scale version of `conj(w)` for the matching receive-array convention. If a different HFSS phasor or port convention is used, determine the required conjugation/sign with a one-element or broadside calibration before running all cases.
7. In **Edit Sources**, set each of the eight source magnitudes and phases for a case. A common scale, such as `conj(w) / max(abs(w))`, does not change a normalized pattern. Solve/reuse the linear frequency-domain solution as supported by the AEDT version, and export one CSV/TAB/TXT report per case.
8. For a stronger validation, solve each port as an embedded element with unit excitation and export complex `Etheta`/`Ephi` versus angle. Coherently combine those complex fields in Python with the same source vector. Do not try to combine per-port dB magnitudes; phase information has already been discarded.
9. Place the direct, combined HFSS files in `hfss_exports/case_01.csv` through `case_05.csv` and run:

   ```bash
   # If the exported angle column is already the signed Python angle:
   python hfss_overlay.py --input-dir hfss_exports --output-dir plots/hfss

   # For standard spherical Theta with both Phi=0 and Phi=180 x-z cuts:
   python hfss_overlay.py --input-dir hfss_exports --output-dir plots/hfss \
       --convention hfss_spherical --phi both
   ```

   For embedded complex fields exported as `case_01_port_01.csv` through `case_05_port_08.csv`, use `--mode embedded` instead; the default combines `Re/Im(Etheta)` with the conjugated MVDR source vector before converting to dB. The script accepts common HFSS CSV/TAB/TXT headers such as `Theta`, `Phi`, `dB(GainTotal)`, `dB(RealizedGainTotal)`, and `dB(DirectivityTotal)`. It writes one overlay PNG per case plus `metrics.csv`. Use `--help` for angular-convention, field-column, and checkpoint options.

### What HFSS exports

There is not one universal radiation-pattern extension across all AEDT/HFSS report workflows. The practical report export is usually comma-separated CSV; tab-delimited or plain text exports are also common. The first lines may contain metadata, followed by columns similar to:

```text
Theta [deg], Phi [deg], Frequency [GHz], dB(GainTotal)
```

The exact field name depends on whether the report uses gain, realized gain, directivity, or a component such as `Etheta`. A dB gain export is sufficient for a **single already-combined source case**. It is not sufficient for coherent embedded-element recombination: that workflow needs complex field components (real/imaginary columns or magnitude/phase columns) for every port.

`hfss_overlay.py` supports `.csv`, `.tab`, `.txt`, and `.dat` report files, metadata/comment lines, common dB/linear pattern columns, and complex `Etheta`/`Ephi` columns for embedded-element files. If an export uses a custom header, pass the column names explicitly.

### Five recommended test cases

All cases are inside the current Python grid and satisfy the implemented `>= 3°` separation rule.

| Case | `(theta_s, theta_i)` | Coverage |
|---|---:|---|
| 01 | `(0°, 30°)` | Broadside desired source and off-broadside interferer. |
| 02 | `(30°, −40°)` | Existing `verify_physics.py` reference; asymmetric, well separated. |
| 03 | `(−25°, 45°)` | Sign-asymmetric steering and a wide angular separation. |
| 04 | `(5°, −55°)` | Near-broadside desired source with a near-endfire interferer. |
| 05 | `(58°, 55°)` | Three-degree near-endfire stress case at the dataset boundary. |

Case 05 is intentionally difficult rather than a claim of ideal conditioning. If HFSS cannot resolve it at the selected mesh/report floor, retain it as a diagnostic and add a less-conditioned repeat such as `(45°, 0°)`.

## 9. Known limitations and HFSS risks

- The analytical steering vector assumes identical uncoupled point elements. HFSS includes element patterns, mutual coupling, finite ports, conductor/dielectric losses, and boundary effects.
- The Python model uses angles and fixed SNR/INR; it does not ingest measured embedded steering vectors or a snapshot-estimated covariance. The physically rigorous next step is to build `R` from HFSS embedded patterns and then solve MVDR with those patterns.
- dB-only HFSS exports lose phase. They can validate a directly excited case, but cannot reproduce a coherent sum of eight embedded patterns.
- The sign of the steering phase, HFSS time convention, port orientation, array axis, and `w` versus `conj(w)` source mapping must be calibrated. A mirrored pattern can otherwise look like a model failure.
- Very deep nulls can be below the HFSS adaptive-mesh, interpolation, or report numerical floor. Record the floor and never interpret a clipped −80 dB trace as a measured −120 dB null.
- The random angle split contains nearby points in train and test. It is appropriate for interpolation, not a strong out-of-distribution generalization claim.
- Near-coincident source/interferer angles make the null-aligned basis ill-conditioned. The current generator excludes separations below 3°; the HFSS stress case at exactly 3° should be reported separately.
- The headline metrics and the checked-in artifact metrics above refer to the completed run; the standalone recommended result is listed separately because it is not stored in the ablation JSON.

## 10. Citations

1. J. Capon, “High-resolution frequency-wavenumber spectrum analysis,” *Proceedings of the IEEE*, 57(8), 1408–1418, 1969. DOI: [10.1109/PROC.1969.7278](https://doi.org/10.1109/PROC.1969.7278).
2. H. L. Van Trees, *Optimum Array Processing*, Wiley, 2002. The MVDR and constrained beamforming formulation follows the standard treatment.
3. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013. The low-rank solve in `physics.py` follows the standard Woodbury matrix identity.
4. C. Trabelsi et al., “Deep Complex Networks,” ICLR 2018, [arXiv:1705.09792](https://arxiv.org/abs/1705.09792). This project uses real hidden layers with a structured complex synthesis layer rather than a full CVNN.
5. Ansys, *HFSS Help: Radiation and Far-Field Reports*, [Ansys Electronics Desktop documentation](https://ansyshelp.ansys.com/). The exact report labels and export dialogs can vary by AEDT release.

## Takeaway

A generic regressor is asked to cancel many order-one complex contributions to produce a tiny interference response. The null-aligned physics layer removes that unnecessary numerical burden: the network predicts the one complex coordinate that controls the null, while MVDR structure enforces the desired-signal constraint exactly.
