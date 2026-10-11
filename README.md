# MVDR Beamforming NN — Null-Depth Preservation

Reproduces everything in `report.md`: physics verification, the 6-variant
ablation, and the standalone recommended solution.

## Requirements

- Python 3.9+
- `numpy`, `torch` (CPU build is fine — nothing here needs a GPU), `matplotlib`

```bash
pip install numpy torch matplotlib
```

## Quick start (recommended solution only, ~80 s on CPU)

```bash
python3 recommended_solution.py
```

Trains the recommended model (null-aligned output chart, 2 outputs instead of
16) and prints the two KPIs on the held-out test split. Expected output:

```
evaluating on held-out test split (1405 samples)
null depth NN   median   -93.6 dB | MVDR median   -96.1 dB
SINR gap        median    0.000 dB | p90    0.000 dB
```

(`recommended_solution.py` is self-contained — copy it into your own pipeline.)

## Full reproduction (~10 min on CPU)

```bash
python3 verify_physics.py      # F1/F2/F3 physics checks, sensitivity + SINR-cost tables
python3 gen_dataset.py         # dataset (same grid as your Phase 2) -> data/dataset_N8.npz
python3 train_ablation.py      # 6 variants x 200 epochs -> data/ablation_results.json
python3 make_plots.py          # figs 1-6 -> plots/
```

`train_ablation.py` can also run single variants: `python3 train_ablation.py baseline nullaligned_mse`.

Pretrained outputs from our run are already in `data/` (`ablation_results.json`,
`pred_*.npz`, `model_*.pth`) and figures in `plots/`, so `make_plots.py` alone
will regenerate the figures without retraining.

## What to look at

| file | what |
|---|---|
| `report.md` | full analysis, math, ablation table, literature |
| `recommended_solution.py` | the fix in one file |
| `plots/fig1_sensitivity.png` | why MSE fails: null depth vs weight error |
| `plots/fig2_ablation_bars.png` | null depth / SINR gap per variant |
| `plots/fig3_patterns.png` | MVDR vs NN beampatterns |
| `verify_physics.py` | where the analytic claims come from |

## Notes

- Labels/targets are float64; training runs on CPU in minutes.
- `train_ablation.py` uses one fixed train/val/test split for all variants.
- The dataset excludes `|θs − θi| < 3°` (`MIN_SEP_DEG` in the generators);
  see report §6 for why.
