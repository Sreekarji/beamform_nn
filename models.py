"""
BeamNet: physics-informed beamforming-weight predictor (PyTorch).

Output representations (all end-to-end differentiable, all enforce the MVDR
distortionless constraint a_s^H w = 1 EXACTLY BY CONSTRUCTION except raw16):

  raw16       - network emits 2N real numbers = w directly
                (the user's current approach; `project=True` adds a hard
                gain-normalization layer)
  subspace    - network emits 1 complex number beta (2 reals):
                  w = alpha*a_s + beta*a_i,  alpha = (1 - beta*rho)/N,  rho = a_s^H a_i
                Every distortionless w in span{a_s,a_i} has this form, and the
                MVDR solution lives there exactly (physics fact F1). The null
                requires a cancellation between the two terms -> moderately
                sensitive.
  nullaligned - network emits 1 complex number tau (2 reals):
                  w = g*u + tau*a_i,  g = 1 - tau*rho,
                  u = unit-gain EXACT-null vector (F3: u^H a_i = 0, u^H a_s = 1)
                Then |w^H a_i| = |tau|*N EXACTLY: the single output tau IS the
                null level. tau is scaled by DELTA = sn/(sn+si*N) (its natural
                MVDR magnitude) so the regression target is O(0.01-30).

Every output is distortionless with NO matrix inversion, NO division by a
network activation, and (for nullaligned) a null that cannot drift away.
"""
import numpy as np
import torch
import torch.nn as nn
import physics as ph

# The project uses N=8, but keep the scale definition parameterized so that
# constructing BeamNet(N != 8) does not silently use the wrong INR-dependent
# tau scale.  TAU_SCALE remains the N=8 convenience constant used by the
# pre-trained checkpoints.
DELTA = ph.SIGMA_N / (ph.SIGMA_N + ph.SIGMA_I * 8)
BETA_SCALE = 0.05                                    # typical |beta| ~ 0.016
TAU_SCALE = 0.05 * DELTA                             # tau = TAU_SCALE * net_output


def _act(name):
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[name]()


class BeamNet(nn.Module):
    def __init__(self, N=8, mode='raw16', hidden=(128, 256, 256, 128),
                 act='relu', project=False, feats='ang2', tau_scale=None):
        super().__init__()
        if mode not in ('raw16', 'subspace', 'nullaligned'):
            raise ValueError(f"unknown output mode: {mode}")
        if feats not in ('ang2', 'u5'):
            raise ValueError(f"unknown feature set: {feats}")
        self.N, self.mode, self.project = N, mode, project
        self.feats = feats
        self.tau_scale = (0.05 * ph.SIGMA_N /
                          (ph.SIGMA_N + ph.SIGMA_I * N)
                          if tau_scale is None else tau_scale)

        in_dim = 2 if feats == 'ang2' else 5
        out_dim = 2 * N if mode == 'raw16' else 2
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), _act(act)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = ph.DistortionlessProjection()   # radial gain normalization

    def _features(self, ts, ti):
        if self.feats == 'ang2':
            return torch.stack([ts / 60.0, ti / 60.0], dim=-1)
        s_s = torch.sin(ts * np.pi / 180.0)
        s_i = torch.sin(ti * np.pi / 180.0)
        return torch.stack([ts / 60.0, ti / 60.0, s_s, s_i, s_s - s_i], dim=-1)

    def forward(self, ang_deg):
        ts, ti = ang_deg[:, 0], ang_deg[:, 1]
        out = self.net(self._features(ts, ti))
        a_s = ph.steering_t(ts, self.N)
        a_i = ph.steering_t(ti, self.N)

        if self.mode == 'raw16':
            w = ph.to_complex(out)
            if self.project:
                w = self.norm(w, a_s)
            return w

        rho = (a_s.conj() * a_i).sum(-1)                  # a_s^H a_i
        if self.mode == 'subspace':
            beta = BETA_SCALE * torch.complex(out[:, 0], out[:, 1])
            alpha = (1.0 - beta * rho) / self.N           # exact: a_s^H w = 1
            return ph.synth_subspace(torch.stack([alpha, beta], dim=-1), a_s, a_i)
        else:  # nullaligned
            tau = self.tau_scale * torch.complex(out[:, 0], out[:, 1])
            g = 1.0 - tau * rho                           # exact: a_s^H w = 1
            return ph.synth_null_aligned(torch.stack([g, tau], dim=-1), a_s, a_i)
