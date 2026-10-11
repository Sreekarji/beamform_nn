"""
Train + evaluate all ablation variants. Usage:
    python3 train_ablation.py [variant ...]     # default: all variants

Variants isolate one design decision at a time:
  baseline           raw 16-D w, MSE, no constraint          (user's setup)
  raw16_norm         + hard distortionless projection
  subspace_mse       + w = alpha*a_s + beta*a_i output basis
  nullaligned_mse    + w = g*u + tau*a_i output basis (tau IS the null)
  subspace_cost      subspace + physics loss log J(w)/J_min
  nullaligned_full   nullaligned + projection + cost + annealed dB-null hinge
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import physics as ph
from models import BeamNet

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'

# ─── config (kept identical to the user's Phase-3 recipe where applicable) ───
N = 8
EPOCHS = 200
BATCH = 256
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cpu')

VARIANTS = {
    'baseline':          dict(mode='raw16',       hidden=(128, 256, 256, 128), act='relu',
                              project=False, feats='ang2', loss='mse'),
    'raw16_norm':        dict(mode='raw16',       hidden=(128, 256, 256, 128), act='relu',
                              project=True,  feats='ang2', loss='mse'),
    'subspace_mse':      dict(mode='subspace',    hidden=(128, 256, 256, 128), act='relu',
                              project=False, feats='ang2', loss='mse'),
    'subspace_cost':     dict(mode='subspace',    hidden=(128, 256, 256, 128), act='relu',
                              project=False, feats='ang2', loss='cost'),
    'nullaligned_mse':   dict(mode='nullaligned', hidden=(128, 256, 256, 128), act='relu',
                              project=False, feats='ang2', loss='mse'),
    'nullaligned_full':  dict(mode='nullaligned', hidden=(128, 256, 256, 128), act='relu',
                              project=False, feats='ang2', loss='full'),
}

# ─── data ────────────────────────────────────────────────────────────────────
d = np.load(DATA_DIR / 'dataset_N8.npz')
X_deg = torch.tensor(d['X_deg'], dtype=torch.float32)
W_star = torch.tensor(d['W'], dtype=torch.complex64)
JMIN = torch.tensor(d['JMIN'], dtype=torch.float32)
idx = {k: torch.tensor(d[f'idx_{k}'], dtype=torch.long)
       for k in ('train', 'val', 'test')}

def loader(k, shuffle):
    ds = TensorDataset(X_deg[idx[k]], W_star[idx[k]], JMIN[idx[k]])
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_dl, val_dl, test_dl = loader('train', True), loader('val', False), loader('test', False)
print(f"data: train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])}")

# ─── loss helpers ────────────────────────────────────────────────────────────
def hinge_target(epoch):
    """Anneal the null-depth target: -25 dB -> -60 dB between epochs 30 and 80."""
    if epoch < 30:
        return -25.0
    return -25.0 - 35.0 * min(1.0, (epoch - 30) / 50.0)

def compute_loss(kind, w, a_s, a_i, w_star, jmin, epoch):
    if kind == 'mse':
        return ph.loss_weight_mse(w, w_star)
    if kind == 'cost':
        return ph.loss_excess_cost(w, a_s, a_i, jmin)
    if kind == 'full':
        # physics cost (dominant) + mild weight anchor + annealed dB hinge on null
        L = ph.loss_excess_cost(w, a_s, a_i, jmin)
        L = L + 0.01 * ph.loss_weight_mse(w, w_star)
        if epoch >= 30:
            L = L + 0.02 * ph.loss_null_hinge(w, a_s, a_i, hinge_target(epoch))
        return L
    raise ValueError(kind)

# ─── evaluation ──────────────────────────────────────────────────────────────
def steering_grid():
    g = torch.linspace(-90, 90, 1800)
    return ph.steering_t(g, N), g

A_GRID, TH_GRID = steering_grid()

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    ang, wps, wts, jms = [], [], [], []
    for xb, wb, jb in dl:
        ang.append(xb); wts.append(wb); jms.append(jb)
        wps.append(model(xb))
    ang = torch.cat(ang); wps = torch.cat(wps); wts = torch.cat(wts); jms = torch.cat(jms)

    ts, ti = ang[:, 0].to(torch.float64), ang[:, 1].to(torch.float64)
    a_s = ph.steering_t(ts, N); a_i = ph.steering_t(ti, N)   # complex128
    wp, wt = wps.to(torch.complex128), wts.to(torch.complex128)

    # pointing error via 1800-pt grid
    scores = (wps.conj() @ A_GRID.T).abs() ** 2          # (B, 1800)
    peak = TH_GRID[scores.argmax(dim=1)]
    pointing = (peak - ang[:, 0]).abs()

    nd_p = ph.null_depth_db(wp, a_s, a_i)
    nd_t = ph.null_depth_db(wt, a_s, a_i)
    sinr_p = ph.sinr_db_t(wp, a_s, a_i)
    sinr_t = ph.sinr_db_t(wt, a_s, a_i)
    rel_err = ((wp - wt).abs().pow(2).sum(-1).sqrt() / wt.abs().pow(2).sum(-1).sqrt())

    return dict(
        pointing_median=float(pointing.median()),
        pointing_max=float(pointing.max()),
        null_median=float(nd_p.median()),
        null_worst=float(nd_p.quantile(0.90)),        # shallow 10th-percentile nulls
        null_mvdr_median=float(nd_t.median()),
        null_preservation=float((nd_p - nd_t).median()),
        sinr_gap_median=float((sinr_t - sinr_p).median()),
        sinr_gap_worst=float((sinr_t - sinr_p).quantile(0.90)),
        rel_w_err_median=float(rel_err.median()),
        w_pred=wps.numpy(),
    )

# ─── training ────────────────────────────────────────────────────────────────
def train_variant(name, cfg):
    torch.manual_seed(SEED)
    model = BeamNet(N=N, mode=cfg['mode'], hidden=cfg['hidden'], act=cfg['act'],
                    project=cfg['project'], feats=cfg['feats']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    best_val, best_state, t0 = float('inf'), None, time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, wb, jb in train_dl:
            a_s = ph.steering_t(xb[:, 0], N)
            a_i = ph.steering_t(xb[:, 1], N)
            opt.zero_grad()
            w = model(xb)
            loss = compute_loss(cfg['loss'], w, a_s, a_i, wb, jb, epoch)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = []
            for xb, wb, jb in val_dl:
                a_s = ph.steering_t(xb[:, 0], N)
                a_i = ph.steering_t(xb[:, 1], N)
                vl.append(compute_loss(cfg['loss'], model(xb), a_s, a_i, wb, jb, epoch).item())
        val_loss = float(np.mean(vl))
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 40 == 0 or epoch == 1:
            print(f"  [{name}] epoch {epoch:3d}/{EPOCHS} | train {np.mean(losses):.5f} "
                  f"| val {val_loss:.5f} | {time.time()-t0:.0f}s")

    if best_state is None:
        raise RuntimeError(f"{name}: validation loss was never finite")
    model.load_state_dict(best_state)
    metrics = evaluate(model, test_dl)
    metrics['best_val_loss'] = best_val
    metrics['train_seconds'] = time.time() - t0
    w_pred = metrics.pop('w_pred')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / f'pred_{name}.npz', w_pred=w_pred)
    torch.save(model.state_dict(), DATA_DIR / f'model_{name}.pth')
    return metrics

# ─── analytic references ─────────────────────────────────────────────────────
@torch.no_grad()
def analytic_refs():
    out = {}
    dl = test_dl
    ang, wts = [], []
    for xb, wb, jb in dl:
        ang.append(xb); wts.append(wb)
    ang = torch.cat(ang); wts = torch.cat(wts)
    ts, ti = ang[:, 0].to(torch.float64), ang[:, 1].to(torch.float64)
    a_s = ph.steering_t(ts, N); a_i = ph.steering_t(ti, N)   # complex128
    for nm, w in [('mvdr_ground_truth', wts.to(torch.complex128)),
                  ('exact_null_analytic', ph.exact_null_basis(a_s, a_i).to(torch.complex128))]:
        nd = ph.null_depth_db(w, a_s, a_i)
        sinr = ph.sinr_db_t(w, a_s, a_i)
        out[nm] = dict(null_median=float(nd.median()),
                       null_worst=float(nd.quantile(0.90)),
                       sinr_gap_median=0.0 if nm == 'mvdr_ground_truth' else
                       float((ph.sinr_db_t(wts.to(torch.complex128), a_s, a_i) - sinr).median()))
    return out

def summarize(all_res):
    hdr = f"{'variant':<20} {'point':>7} {'null_med':>9} {'null_worst':>11} {'preserv':>8} {'sinr_gap':>9} {'sinr_p90':>9} {'rel_err':>9}"
    print("\n" + "=" * len(hdr))
    print("TEST-SET RESULTS (median unless noted; null/preservation in dB; sinr_gap = MVDR - NN)")
    print(hdr)
    print("-" * len(hdr))
    for nm, r in all_res.items():
        if 'pointing_median' not in r:
            print(f"{nm:<20} {'':>7} {r['null_median']:>9.1f} {r['null_worst']:>11.1f} {'':>8} "
                  f"{r['sinr_gap_median']:>9.3f} {'':>9} {'':>9}")
        else:
            print(f"{nm:<20} {r['pointing_median']:>7.3f} {r['null_median']:>9.1f} "
                  f"{r['null_worst']:>11.1f} {r['null_preservation']:>8.1f} "
                  f"{r['sinr_gap_median']:>9.3f} {r['sinr_gap_worst']:>9.3f} {r['rel_w_err_median']:>9.2e}")
    print("=" * len(hdr))

if __name__ == '__main__':
    names = sys.argv[1:] or list(VARIANTS)
    all_res = analytic_refs()
    for nm in names:
        print(f"\n>>> training {nm}  {VARIANTS[nm]}")
        all_res[nm] = train_variant(nm, VARIANTS[nm])
    summarize(all_res)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / 'ablation_results.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"\nresults -> {DATA_DIR / 'ablation_results.json'}")
