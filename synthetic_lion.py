#!/usr/bin/env python3
"""
synthetic_lion.py — Synthetic LION: DDQN vs. DAI-P vs. Break-Even
on a 2-D Gaussian CTI curriculum-learning task.

Self-contained: no smartville imports.  Run with:
    python synthetic_lion.py [--episodes 300] [--seeds 5] [--no-plot]

Multi-GPU / parallel execution
-------------------------------
    python synthetic_lion.py --gpus 0,1,2,3          # 4 GPUs, 1 worker each
    python synthetic_lion.py --gpus auto              # all GPUs with ≥10 GB free
    python synthetic_lion.py --gpus 0 --workers-per-gpu 4   # 4 workers on GPU 0
    python synthetic_lion.py --gpus auto --min-free-gpu-gb 20

Each (seed × agent) pair runs as an independent subprocess on the assigned GPU.
Pretraining happens in the main process (fast) and the state dict is serialised
to each worker via pickle.

Weights & Biases logging
-------------------------
    echo "WANDB_API_KEY=<your_key>" > .env
    python synthetic_lion.py --wandb-project synthetic-lion [--wandb-entity <team>]

One W&B run is created per (seed × agent) under a shared group tag so the UI
can aggregate statistics across seeds.

Why DAI-P should beat DDQN here
---------------------------------
Unknown class A (malicious) overlaps in 2-D with Known Benign 0, so the
inference module initially classifies A-type flows as benign with HIGH
confidence (low anomaly score).  The agent therefore accepts them and bleeds
budget.  Once label-A is purchased, the inference module immediately adds A's
prototype to its known-class roster: A-type flows flip from "high-confidence
benign" to "high-confidence malicious A", producing the LARGEST possible
one-step change in the proprioceptive state
(known-class confidence, anomaly score, n_labels_bought, budget).
Because this transition is maximally surprising, the DAI-P transition model
carries a high residual prediction error—high perceptive-epistemic gain—for
the buy-label-A action, driving the agent to execute it early.

DDQN explores with Boltzmann sampling over Q-values and relies on
discovering the reward signal of label-A through trial and error.  It can
eventually converge, but it loses critical budget in the early episodes.

The Break-Even agent buys the label with the HIGHEST anomaly score first
(the natural heuristic).  Unknown A has the LOWEST anomaly score (it looks
benign), so the Break-Even agent consistently de-prioritises it—buying B or
C first—and suffers the same budget haemorrhage as naive DDQN

The epistemic gain is NOT foreknowledge — it is learned across episodes.
The first time the agent happens to buy label A (via Boltzmann exploration),
the transition network fails to predict the ensuing proprioceptive flip and
incurs a large MSE.  This error augments the EFE target for buy-label-A,
raising its intrinsic value.  Subsequent episodes make the agent more likely
to repeat the purchase, generating more confirmatory replay data.  The signal
is immediate (one-step MSE) rather than delayed (downstream reward), which is
why DAI-P learns the priority of label-A faster than DDQN.

Architecture overview
---------------------
    DataGenerator      7 2-D Gaussians (3 known + 4 unknown)
    InferenceModule    prototypical classifier + soft anomaly/cluster head
                       updates online via EMA; label buying adds a prototype
    SyntheticLIONEnv   budget episode (block=0 / accept=1 / buy-label=2)
    TwoStreamNet       shared backbone for Q-net / EFE-net  (5 extero + 9 proprio)
    TransitionNet      predicts next proprioceptive state (9-D) from (state, action)
    DDQNAgent          Double-DQN with Boltzmann sampling (no epistemic gain)
    DAIPAgent          DAI-P: reward + epistemic gain from transition MSE
    BreakEvenAgent     rule-based: buy highest-anomaly-score unlabelled cluster when affordable
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import multiprocessing
import os
import pathlib
import random
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Global hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Gaussian data
    input_dim    = 2,
    hidden_dim   = 24,       # encoder output / prototype dimension
    n_known      = 3,
    n_unknown    = 4,
    proto_temp   = 3.0,      # softmax temperature for prototypical assignment

    # Episode
    # Budget is tight: total label cost = 5+6+4+4 = 19 > init_budget of 15.
    # The agent can afford at most 2-3 labels. It MUST choose which ones.
    # Break-Even always buys the highest-anomaly-score label first (= B),
    # spending 6 of 15 up front; then can only afford C (4) and is stuck with
    # 5 left—not enough for A (5 exact) if any A-flow has already drained budget.
    # DAI-P's epistemic gain pushes it to buy A first despite A's low anomaly score.
    max_steps    = 150,
    init_budget  = 15.0,
    min_budget   = 0.0,
    max_budget   = 40.0,

    # Label prices  (A, B, C, D)
    label_prices = [5.0, 6.0, 4.0, 4.0],

    # Rewards: (uninformed, informed) per action per class type
    # known-class flows always use informed rewards
    r_accept_benign_informed    =  2.0,
    r_block_benign_informed     = -2.5,
    r_accept_malicious_informed = -5.0,
    r_block_malicious_informed  =  2.5,

    # unknown flows before label is bought
    # A: overlapping-malicious, B: separate-malicious, C: separate-benign, D: near-malicious-benign
    r_accept_A_uninformed =  -7.0,   # agent is fooled: thinks it's benign, accepts
    r_block_A_uninformed  =   0.5,   # accidental good block
    r_accept_B_uninformed =  -3.5,   # anomaly visible, but accepting possible
    r_block_B_uninformed  =   1.5,   # anomaly detected and blocked
    r_accept_C_uninformed =   0.5,   # benign but looks anomalous → OK to accept
    r_block_C_uninformed  =  -1.5,   # false positive on benign
    r_accept_D_uninformed =   0.5,   # benign, near malicious → sometimes accepted
    r_block_D_uninformed  =  -1.0,   # false positive

    # penalty for buying an invalid label
    buy_invalid_penalty  = -0.3,

    # RL hyper-params
    gamma        = 0.95,
    lr           = 4e-4,
    batch_size   = 64,
    memory_size  = 6000,
    target_update = 20,       # steps between target-net hard updates
    temperature  = 2.0,       # Boltzmann softmax temperature
    min_memory_to_train = 128,

    # DAI-P
    epistemic_weight = 0.5,   # weight on perceptive-epistemic gain in EFE target

    # Inference module
    anomaly_threshold = 1.8,  # hidden-space L2 distance → anomaly if >
    proto_ema         = 0.08, # EMA coefficient for unknown prototype update

    # Pretraining
    pretrain_epochs      = 120,
    pretrain_n_per_class = 200,
    pretrain_lr          = 1e-3,
)

# ──────────────────────────────────────────────────────────────────────────────
# 0b.  .env loader, GPU helpers, W&B wrapper
# ──────────────────────────────────────────────────────────────────────────────

def load_dotenv(path: str = '.env') -> None:
    """Minimal .env parser — no python-dotenv dependency required."""
    p = pathlib.Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _available_gpus(min_free_gb: float = 10.0) -> List[str]:
    """Return cuda:N ids for GPUs with at least min_free_gb of free VRAM."""
    if not torch.cuda.is_available():
        return ['cpu']
    found = []
    for i in range(torch.cuda.device_count()):
        try:
            free_b, _ = torch.cuda.mem_get_info(i)
            if free_b / 1e9 >= min_free_gb:
                found.append(f'cuda:{i}')
        except Exception:
            pass
    return found if found else ['cpu']


def _parse_gpus(gpus_arg: str, min_free_gb: float) -> List[str]:
    if gpus_arg == 'auto':
        found = _available_gpus(min_free_gb)
        print(f'  [auto-GPU] free GPUs (≥{min_free_gb:.0f} GB): {found}')
        return found
    if gpus_arg.lower() == 'cpu':
        return ['cpu']
    out = []
    for g in gpus_arg.split(','):
        g = g.strip()
        out.append(f'cuda:{g}' if not g.startswith('cuda') else g)
    return out


class _WandbLogger:
    """
    Optional wandb wrapper.  Creates one run per (seed × agent) under a
    shared group tag so the W&B UI can aggregate across seeds automatically.
    Falls back to a no-op when wandb is absent or WANDB_API_KEY is missing.
    """

    def __init__(self, project: Optional[str], entity: Optional[str],
                 group: str, agent_name: str, seed: int, cfg: dict):
        self._run = None
        if project is None:
            return
        api_key = os.environ.get('WANDB_API_KEY', '')
        if not api_key:
            print('[wandb] WANDB_API_KEY not set — logging disabled.')
            return
        try:
            import wandb as _w
            self._w = _w
            self._run = _w.init(
                project=project, entity=entity or None,
                group=group, job_type=agent_name,
                name=f'{agent_name}_seed{seed}',
                config={**cfg, 'seed': seed, 'agent': agent_name},
                reinit=True,
            )
        except ImportError:
            print('[wandb] not installed — run: pip install wandb')

    def log(self, metrics: dict, step: int) -> None:
        if self._run is not None:
            self._w.log(metrics, step=step)

    def summary(self, metrics: dict) -> None:
        if self._run is not None:
            for k, v in metrics.items():
                self._run.summary[k] = v

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data generator
# ──────────────────────────────────────────────────────────────────────────────

class DataGenerator:
    """
    Seven 2-D Gaussians:

    Known (pretrained):
      K0 – benign,    centre (−2,  0),  σ=0.40
      K1 – benign,    centre ( 2,  0),  σ=0.40
      K2 – malicious, centre ( 0, −2),  σ=0.40

    Unknown (label must be bought):
      UA – malicious, centre (−1.5,  0.4), σ=0.35  ← overlaps K0; highest epistemic value
      UB – malicious, centre ( 0.0,  3.0), σ=0.45  ← clearly anomalous
      UC – benign,    centre ( 4.0,  1.0), σ=0.45  ← clearly anomalous, false-positive risk
      UD – benign,    centre ( 2.4, −1.0), σ=0.35  ← near K2, misclassified-malicious risk
    """
    N_KNOWN   = 3
    N_UNKNOWN = 4

    KNOWN_MEANS  = np.array([[-2.0,  0.0], [ 2.0,  0.0], [ 0.0, -2.0]])
    KNOWN_STDS   = [0.40, 0.40, 0.40]
    KNOWN_TYPES  = ['benign', 'benign', 'malicious']

    UNKNOWN_MEANS = np.array([[-1.5,  0.4], [ 0.0,  3.0], [ 4.0,  1.0], [ 2.4, -1.0]])
    UNKNOWN_STDS  = [0.35, 0.45, 0.45, 0.35]
    UNKNOWN_TYPES = ['malicious', 'malicious', 'benign', 'benign']
    UNKNOWN_NAMES = ['A(mal,overlap)', 'B(mal,sep)', 'C(ben,sep)', 'D(ben,nearmal)']

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def sample(self, class_id: Optional[int] = None):
        """Returns (x, class_id, is_known, unknown_id, class_type)."""
        n_total = self.N_KNOWN + self.N_UNKNOWN
        if class_id is None:
            class_id = self.rng.randint(0, n_total)

        if class_id < self.N_KNOWN:
            k = class_id
            x = self.rng.normal(self.KNOWN_MEANS[k], self.KNOWN_STDS[k])
            return x, class_id, True, None, self.KNOWN_TYPES[k]
        else:
            u = class_id - self.N_KNOWN
            x = self.rng.normal(self.UNKNOWN_MEANS[u], self.UNKNOWN_STDS[u])
            return x, class_id, False, u, self.UNKNOWN_TYPES[u]

    def sample_known_batch(self, n_per_class: int = 200):
        """Returns (X, y) for supervised pretraining on known classes."""
        Xs, ys = [], []
        for k in range(self.N_KNOWN):
            xs = self.rng.normal(self.KNOWN_MEANS[k], self.KNOWN_STDS[k],
                                 size=(n_per_class, 2))
            Xs.append(xs)
            ys.extend([k] * n_per_class)
        return np.vstack(Xs), np.array(ys, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Inference module (fully differentiable)
# ──────────────────────────────────────────────────────────────────────────────

class InferenceModule(nn.Module):
    """
    Prototypical classifier + differentiable soft-anomaly / cluster head.

    Known prototypes are supervised (pretrained).  When a CTI label is bought
    for unknown cluster uid, that cluster's EMA prototype is promoted to the
    known-prototype list so future flows are immediately classified correctly.

    Unknown cluster prototypes are EMA buffers (no backprop).
    The encoder is trained during pretraining via prototypical loss.

    Anomaly detection: a flow is anomalous if its L2 distance to the nearest
    known prototype exceeds `anomaly_threshold` in hidden space.
    Unknown clustering: differentiable soft-assignment via temperature softmax
    over distances to learnable unknown-cluster prototypes.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d_in  = cfg['input_dim']
        d_h   = cfg['hidden_dim']
        n_k   = cfg['n_known']
        n_u   = cfg['n_unknown']
        self.proto_temp        = cfg['proto_temp']
        self.anomaly_threshold = cfg['anomaly_threshold']
        self.n_known_initial   = n_k
        self.n_unknown         = n_u
        self.device            = torch.device(cfg.get('device', 'cpu'))

        # Encoder: 2-D → hidden
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_h * 2),
            nn.ReLU(),
            nn.Linear(d_h * 2, d_h),
        )

        # Known-class prototypes (supervised, grown when labels bought)
        self.known_protos = nn.Parameter(torch.randn(n_k, d_h) * 0.1)
        # Corresponding class types (extended when label bought)
        self.known_types_list: List[str] = []   # filled at env reset

        # Unknown-cluster EMA prototypes (not trained by backprop)
        self.register_buffer('unk_protos',  torch.randn(n_u, d_h) * 0.1)
        # Running stats for proprioceptive state
        self.register_buffer('unk_conf',    torch.zeros(n_u))
        self.register_buffer('unk_size',    torch.zeros(n_u))

        self.to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def known_classify(self, h: torch.Tensor):
        """Returns (pred_class, confidence, distances) for known prototypes."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        dists = torch.cdist(h, self.known_protos)       # (B, n_known)
        probs = F.softmax(-dists * self.proto_temp, dim=-1)
        pred  = probs.argmax(-1)
        conf  = probs.max(-1).values
        return pred, conf, dists

    def anomaly_detect(self, dists_to_known: torch.Tensor):
        """Returns (anomaly_score, is_anomaly_mask)."""
        min_dist   = dists_to_known.min(-1).values
        is_anomaly = min_dist > self.anomaly_threshold
        return min_dist, is_anomaly

    def unk_cluster(self, h: torch.Tensor):
        """Returns (cluster_id, cluster_conf, soft_weights) using current unk_protos."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        dists   = torch.cdist(h, self.unk_protos)
        weights = F.softmax(-dists * self.proto_temp, dim=-1)
        cid     = weights.argmax(-1)
        cconf   = weights.max(-1).values
        return cid, cconf, weights

    @torch.no_grad()
    def update_unk_proto(self, h: torch.Tensor, cid: int, ema: float):
        self.unk_protos[cid] = (1 - ema) * self.unk_protos[cid] + ema * h.squeeze()
        self.unk_size[cid]   = min(self.unk_size[cid] + 1.0, 200.0)

    @torch.no_grad()
    def update_unk_conf(self, cid: int, conf: float):
        self.unk_conf[cid] = 0.9 * self.unk_conf[cid] + 0.1 * conf

    def add_known_proto(self, proto: torch.Tensor, class_type: str):
        """Promote an unknown cluster to known when its label is bought."""
        proto = proto.detach().unsqueeze(0)
        self.known_protos = nn.Parameter(
            torch.cat([self.known_protos.detach(), proto], dim=0)
        )
        self.known_types_list.append(class_type)

    @torch.no_grad()
    def init_unk_protos_with_data(self, data_gen: DataGenerator, n_samples: int = 30):
        """
        Warm-start unknown cluster protos using the pretrained encoder applied to
        samples drawn near each true unknown cluster centre.
        This avoids cold-start misassignment of flows at episode start while
        keeping the prototype positions random enough to require online updating.
        """
        for u in range(self.n_unknown):
            xs = np.random.normal(data_gen.UNKNOWN_MEANS[u],
                                  data_gen.UNKNOWN_STDS[u] * 1.5,
                                  size=(n_samples, 2))
            h = self.encode(torch.tensor(xs, dtype=torch.float32, device=self.device))
            self.unk_protos[u] = h.mean(0)

    def reset_episode_state(self, original_n_known: int, data_gen: DataGenerator):
        """
        Roll back added prototypes + re-initialise unknown cluster protos.
        Called at the start of every episode.
        """
        with torch.no_grad():
            self.known_protos = nn.Parameter(
                self.known_protos.detach()[:original_n_known].clone()
            )
        self.known_types_list = list(data_gen.KNOWN_TYPES[:original_n_known])
        self.init_unk_protos_with_data(data_gen)
        self.unk_conf.zero_()
        self.unk_size.zero_()

    def forward(self, x: torch.Tensor) -> dict:
        h                        = self.encode(x)
        pred_k, conf_k, dists_k = self.known_classify(h)
        anorm_score, is_anom     = self.anomaly_detect(dists_k)
        cid, cconf, cweights     = self.unk_cluster(h)
        return dict(h=h, pred_k=pred_k, conf_k=conf_k, dists_k=dists_k,
                    anorm_score=anorm_score, is_anom=is_anom,
                    cid=cid, cconf=cconf, cweights=cweights)


def pretrain_inference(inf: InferenceModule, dg: DataGenerator, cfg: dict) -> float:
    """Supervised prototypical loss on known classes.  Returns final loss."""
    opt = optim.Adam(inf.parameters(), lr=cfg['pretrain_lr'])
    inf.train()
    X_np, y_np = dg.sample_known_batch(cfg['pretrain_n_per_class'])
    X = torch.tensor(X_np, dtype=torch.float32, device=inf.device)
    y = torch.tensor(y_np, dtype=torch.long,    device=inf.device)
    final_loss = float('nan')
    for _ in range(cfg['pretrain_epochs']):
        perm = torch.randperm(len(X), device=inf.device)
        X, y = X[perm], y[perm]
        h = inf.encode(X)
        _, _, dists = inf.known_classify(h)
        log_p = F.log_softmax(-dists * inf.proto_temp, dim=-1)
        loss  = F.nll_loss(log_p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    inf.eval()
    return final_loss


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Environment
# ──────────────────────────────────────────────────────────────────────────────

# State layout: [extero(5), proprio(9)] = 14-D total
#
#   extero  (5): known_pred_norm, known_conf, anorm_score_norm,
#                unk_proto_x_norm, unk_proto_y_norm
#
#   proprio (9): conf_unk_0, conf_unk_1, conf_unk_2, conf_unk_3,
#                n_labels_bought_frac, budget_frac, label_price_frac,
#                time_frac, known_conf_norm
#
# Four per-cluster confidences (EMA).  
# The transition network can cleanly learn that buying
# label-A drives conf_unk_0 to 0 while leaving the other three unchanged —
# the sharpest signal of the episode's epistemic structure.

STATE_DIM   = 14
PROPRIO_DIM = 9
ACTION_SIZE = 3   # 0=block, 1=accept, 2=buy-label


class SyntheticLIONEnv:
    """
    Budget episode environment.

    Each step: one flow is sampled, the inference module classifies/clusters
    it, and the agent picks block / accept / buy-label.

    When a CTI label is bought for unknown cluster uid:
      1. The cluster's EMA prototype is promoted to the known-proto list.
      2. unk_conf[uid] is zeroed (cluster is resolved; no longer uncertain).
      3. Future flows from that cluster are immediately classified correctly.

    This produces the LARGEST one-step proprioceptive state change for
    cluster A specifically — all four per-cluster confidences shift, but
    conf_unk_0 flips from a settled EMA value to 0 — giving the transition
    network its sharpest prediction failure and thus the highest epistemic
    gain for the buy-label-A action.
    """

    def __init__(self, data_gen: DataGenerator, inf_mod: InferenceModule, cfg: dict):
        self.dg     = data_gen
        self.inf    = inf_mod
        self.cfg    = cfg
        self.n_unk  = cfg['n_unknown']
        self.prices = cfg['label_prices']
        self.device = inf_mod.device

        # Store pretrained known-proto count for episode reset
        self._pretrained_n_known = inf_mod.known_protos.shape[0]

    def reset(self) -> torch.Tensor:
        self.budget        = self.cfg['init_budget']
        self.t             = 0
        self.labels_bought = [False] * self.n_unk
        self.inf.reset_episode_state(self._pretrained_n_known, self.dg)
        self._step_flow()
        return self._state()

    def _step_flow(self):
        """Sample next flow and run inference.  Caches result."""
        x, cid, is_k, uid, ctype = self.dg.sample()
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            r = self.inf(xt)
        self._flow = dict(
            x=x, class_id=cid, is_known=is_k, unknown_id=uid, class_type=ctype,
            h=r['h'],
            pred_k=r['pred_k'].item(), conf_k=r['conf_k'].item(),
            anorm_score=r['anorm_score'].item(), is_anom=r['is_anom'].item(),
            cid=r['cid'].item()   if uid is not None else -1,
            cconf=r['cconf'].item() if uid is not None else r['conf_k'].item(),
        )
        if uid is not None:
            self.inf.update_unk_proto(r['h'], r['cid'].item(), self.cfg['proto_ema'])
            self.inf.update_unk_conf(r['cid'].item(), r['cconf'].item())

    def _state(self) -> torch.Tensor:
        f   = self._flow
        cfg = self.cfg

        # Exteroceptive (5)
        known_pred_norm = f['pred_k'] / max(self.inf.known_protos.shape[0] - 1, 1)
        known_conf_norm = f['conf_k']
        anorm_norm      = min(f['anorm_score'] / 5.0, 1.0)
        if f['unknown_id'] is not None:
            up = self.inf.unk_protos[f['cid']].detach().cpu().numpy()
        else:
            up = self.inf.known_protos[f['pred_k']].detach().cpu().numpy()
        proto_x = float(up[0]) / 5.0
        proto_y = float(up[1]) / 5.0

        # Proprioceptive (9): 4 per-cluster EMA confidences + 5 scalars
        # Zeroed for clusters whose label has already been bought (resolved).
        conf_per_cluster = [
            0.0 if self.labels_bought[u]
            else float(self.inf.unk_conf[u].item())
            for u in range(self.n_unk)
        ]
        n_labels_frac = sum(self.labels_bought) / self.n_unk
        budget_frac   = self.budget / cfg['init_budget']
        uid           = f['unknown_id']
        price_frac    = (self.prices[uid] / cfg['init_budget']
                         if uid is not None and not self.labels_bought[uid] else 0.0)
        time_frac     = self.t / cfg['max_steps']

        extero  = [known_pred_norm, known_conf_norm, anorm_norm, proto_x, proto_y]
        proprio = conf_per_cluster + [n_labels_frac, budget_frac, price_frac,
                                       time_frac, known_conf_norm]
        return torch.tensor(extero + proprio, dtype=torch.float32, device=self.device)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        f   = self._flow
        cfg = self.cfg
        uid = f['unknown_id']
        reward = 0.0

        if action == 2:                         # ── buy label ──
            if f['is_known'] or uid is None or self.labels_bought[uid]:
                reward = cfg['buy_invalid_penalty']
            elif self.budget < self.prices[uid]:
                reward = cfg['buy_invalid_penalty']
            else:
                price  = self.prices[uid]
                reward = -price * 0.05          # tiny immediate cost signal
                self.budget -= price
                self.labels_bought[uid] = True
                self.inf.add_known_proto(self.inf.unk_protos[uid],
                                         self.dg.UNKNOWN_TYPES[uid])
                self.inf.unk_conf[uid] = 0.0   # cluster resolved; zero its proprio slot

        elif action == 0:                       # ── block ──
            if f['is_known']:
                ctype  = self.inf.known_types_list[f['pred_k']]
                reward = (cfg['r_block_malicious_informed'] if ctype == 'malicious'
                          else cfg['r_block_benign_informed'])
            elif uid is not None and self.labels_bought[uid]:
                ctype  = self.dg.UNKNOWN_TYPES[uid]
                reward = (cfg['r_block_malicious_informed'] if ctype == 'malicious'
                          else cfg['r_block_benign_informed'])
            else:
                reward = cfg.get(f'r_block_{self.dg.UNKNOWN_NAMES[uid][0]}_uninformed', 0.5)

        elif action == 1:                       # ── accept ──
            if f['is_known']:
                ctype  = self.inf.known_types_list[f['pred_k']]
                reward = (cfg['r_accept_benign_informed'] if ctype == 'benign'
                          else cfg['r_accept_malicious_informed'])
            elif uid is not None and self.labels_bought[uid]:
                ctype  = self.dg.UNKNOWN_TYPES[uid]
                reward = (cfg['r_accept_benign_informed'] if ctype == 'benign'
                          else cfg['r_accept_malicious_informed'])
            else:
                reward = cfg.get(f'r_accept_{self.dg.UNKNOWN_NAMES[uid][0]}_uninformed', -1.0)

        self.budget += reward
        self.t      += 1
        done = (self.budget <= cfg['min_budget']
                or self.budget >= cfg['max_budget']
                or self.t     >= cfg['max_steps'])
        win  = self.budget >= cfg['max_budget']
        self.budget = max(cfg['min_budget'], min(self.budget, cfg['max_budget'] + 5.0))
        if not done:
            self._step_flow()
        info = dict(budget=self.budget, win=win,
                    labels=list(self.labels_bought), n_labels=sum(self.labels_bought),
                    reward=reward)
        return self._state(), reward, done, info


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Neural modules (two-stream: extero ‖ proprio)
# ──────────────────────────────────────────────────────────────────────────────

class TwoStreamNet(nn.Module):
    """
    Shared backbone for DQN / EFE-net.
    Input: state (STATE_DIM=14) split into extero (5) and proprio (9).
    Output: action logits (ACTION_SIZE=3).
    """
    def __init__(self, out_dim: int = ACTION_SIZE, hidden: int = 48):
        super().__init__()
        extero_dim  = STATE_DIM - PROPRIO_DIM   # 5
        proprio_dim = PROPRIO_DIM               # 9

        self.ext_fc1 = nn.Linear(extero_dim,  hidden)
        self.ext_fc2 = nn.Linear(hidden,      hidden // 2)
        self.pro_fc1 = nn.Linear(proprio_dim, hidden)
        self.pro_fc2 = nn.Linear(hidden,      hidden // 2 * 2)
        self.out     = nn.Linear(hidden // 2 + hidden // 2 * 2 // 2, out_dim)
        self._h2 = hidden // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        e = x[:, :STATE_DIM - PROPRIO_DIM]
        p = x[:, STATE_DIM - PROPRIO_DIM:]
        e = F.relu(self.ext_fc1(e))
        e = F.relu(self.ext_fc2(e))
        p = F.relu(self.pro_fc1(p))
        p = F.relu(self.pro_fc2(p))
        p = p[:, :self._h2]
        return self.out(torch.cat([e, p], dim=-1))


class TransitionNet(nn.Module):
    """
    Predicts the next PROPRIOCEPTIVE state (9-D) given (state, action-one-hot).

    With four per-cluster confidence slots in proprio, the transition net can
    learn clean cluster-specific update rules:
      - buy-label-A  → conf_unk_0 → 0, others unchanged
      - accept A-flow → conf_unk_0 updates via EMA, others unchanged
    This sharpens the epistemic gain signal for the buy-label-A action.
    """
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, PROPRIO_DIM),
        )

    def forward(self, state: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action_oh], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: List = [None] * capacity
        self.pos = 0
        self.size = 0

    def push(self, *transition):
        self.buf[self.pos] = tuple(t.detach().clone() if isinstance(t, torch.Tensor)
                                   else t for t in transition)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = random.sample(range(self.size), n)
        return zip(*[self.buf[i] for i in idx])

    def __len__(self):
        return self.size


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DDQN agent
# ──────────────────────────────────────────────────────────────────────────────

class DDQNAgent:
    """Double-DQN with Boltzmann sampling.  No epistemic gain — pure reward."""

    def __init__(self, cfg: dict):
        self.gamma       = cfg['gamma']
        self.batch_size  = cfg['batch_size']
        self.temperature = cfg['temperature']
        self.target_upd  = cfg['target_update']
        self.min_mem     = cfg['min_memory_to_train']
        self.device      = torch.device(cfg.get('device', 'cpu'))

        self.net    = TwoStreamNet().to(self.device)
        self.target = TwoStreamNet().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.opt    = optim.Adam(self.net.parameters(), lr=cfg['lr'])
        self.buf    = ReplayBuffer(cfg['memory_size'])
        self._steps = 0

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q     = self.net(state.to(self.device)).squeeze()
            probs = F.softmax(self.temperature * q, dim=-1)
        return torch.multinomial(probs, 1).item()

    def push(self, s, a, r, s2, done):
        # Store on CPU; moved to device at training time to save GPU memory.
        self.buf.push(s.cpu(), torch.tensor(a),
                      torch.tensor(r, dtype=torch.float32),
                      s2.cpu(), torch.tensor(done, dtype=torch.bool))

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.buf) < self.min_mem:
            return None
        states, actions, rewards, next_states, dones = self.buf.sample(self.batch_size)

        S  = torch.stack(list(states)).to(self.device)
        A  = torch.stack(list(actions)).to(self.device)
        R  = torch.stack(list(rewards)).unsqueeze(1).to(self.device)
        S2 = torch.stack(list(next_states)).to(self.device)
        D  = torch.stack(list(dones)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            a_next       = self.net(S2).argmax(1, keepdim=True)
            q_next       = self.target(S2).gather(1, a_next)
            targets_full = self.net(S).detach()
            targets_full[range(self.batch_size), A] = (
                R.squeeze() + self.gamma * q_next.squeeze() * ~D.squeeze()
            )

        self.net.train()
        loss = F.mse_loss(self.net(S), targets_full)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_upd == 0:
            self.target.load_state_dict(self.net.state_dict())
        return {'loss': loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 7.  DAI-P agent
# ──────────────────────────────────────────────────────────────────────────────

class DAIPAgent:
    """
    DAI-P: the canonical value-based Active Inference agent.

    EFE target = reward  +  epistemic_weight × epistemic_gain
                         +  γ × EFE(s', a*)

    Epistemic gain = 0.5 × ||next_proprio − TransitionNet(s, a)||²

    The transition network is trained concurrently on supervised next-proprio
    prediction.  With four per-cluster confidence slots in the proprioceptive
    state, the transition network learns that buying label-A zeroes conf_unk_0
    while leaving the other slots intact — a clean, cluster-specific signal.
    This failure to predict the post-buy state produces high epistemic gain,
    driving the agent to prioritise label-A purchases before the reward signal
    alone would justify it.
    """

    def __init__(self, cfg: dict):
        self.gamma       = cfg['gamma']
        self.batch_size  = cfg['batch_size']
        self.temperature = cfg['temperature']
        self.target_upd  = cfg['target_update']
        self.min_mem     = cfg['min_memory_to_train']
        self.eps_w       = cfg['epistemic_weight']
        self.device      = torch.device(cfg.get('device', 'cpu'))

        self.efe_net    = TwoStreamNet().to(self.device)
        self.efe_target = TwoStreamNet().to(self.device)
        self.efe_target.load_state_dict(self.efe_net.state_dict())
        self.efe_target.eval()
        self.efe_opt    = optim.Adam(self.efe_net.parameters(), lr=cfg['lr'])

        self.trans_net  = TransitionNet().to(self.device)
        self.trans_opt  = optim.Adam(self.trans_net.parameters(), lr=cfg['lr'])

        self.buf    = ReplayBuffer(cfg['memory_size'])
        self._steps = 0
        self._eye   = torch.eye(ACTION_SIZE, device=self.device)

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            nefe  = self.efe_net(state.to(self.device)).squeeze()
            probs = F.softmax(self.temperature * nefe, dim=-1)
        return torch.multinomial(probs, 1).item()

    def push(self, s, a, r, s2, done):
        self.buf.push(s.cpu(), torch.tensor(a),
                      torch.tensor(r, dtype=torch.float32),
                      s2.cpu(), torch.tensor(done, dtype=torch.bool))

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.buf) < self.min_mem:
            return None
        states, actions, rewards, next_states, dones = self.buf.sample(self.batch_size)

        S   = torch.stack(list(states)).to(self.device)
        A   = torch.stack(list(actions)).to(self.device)
        R   = torch.stack(list(rewards)).unsqueeze(1).to(self.device)
        S2  = torch.stack(list(next_states)).to(self.device)
        D   = torch.stack(list(dones)).unsqueeze(1).to(self.device)
        AOH = self._eye[A]                         # one-hot actions  (B, 3)

        next_proprio = S2[:, STATE_DIM - PROPRIO_DIM:]

        # ── epistemic gain ────────────────────────────────────────────────────
        with torch.no_grad():
            pred_next = self.trans_net(S, AOH)
            epist     = 0.5 * ((next_proprio - pred_next) ** 2).sum(dim=1, keepdim=True)

        # ── EFE targets ───────────────────────────────────────────────────────
        with torch.no_grad():
            a_next      = self.efe_net(S2).argmax(1, keepdim=True)
            q_next      = self.efe_target(S2).gather(1, a_next)
            efe_targets = self.efe_net(S).detach().clone()
            efe_targets[range(self.batch_size), A] = (
                R.squeeze()
                + self.eps_w * epist.squeeze()
                + self.gamma * q_next.squeeze() * ~D.squeeze()
            )

        # ── train EFE network ─────────────────────────────────────────────────
        self.efe_net.train()
        efe_loss = F.mse_loss(self.efe_net(S), efe_targets)
        self.efe_opt.zero_grad()
        efe_loss.backward()
        self.efe_opt.step()

        # ── train transition network ──────────────────────────────────────────
        self.trans_net.train()
        trans_loss = F.mse_loss(self.trans_net(S, AOH), next_proprio)
        self.trans_opt.zero_grad()
        trans_loss.backward()
        self.trans_opt.step()

        self._steps += 1
        if self._steps % self.target_upd == 0:
            self.efe_target.load_state_dict(self.efe_net.state_dict())

        return {'loss': efe_loss.item(), 'trans_loss': trans_loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Break-Even rule-based agent
# ──────────────────────────────────────────────────────────────────────────────

class BreakEvenAgent:
    """
    Deterministic heuristic:
      – Accept flows that the inference module classifies as known-benign
        (or labelled-benign unknown).
      – Block flows classified as known-malicious / labelled-malicious unknown.
      – For unlabelled unknown clusters:
          • Buy the label if affordable and the cluster has the highest
            ANOMALY SCORE among available unlabelled clusters.
            (Rationale: high anomaly score → clearly anomalous → must be dangerous.)
          • Otherwise block if anomaly score > 0.5, else accept.

    This heuristic is deliberate: Unknown-A (the most dangerous cluster) has
    the LOWEST anomaly score (overlaps K0), so it is always
    de-prioritised.  B is bought first, then C or D.  A is reached last or
    not at all — the A-flows keep draining budget the entire time.
    """

    def act(self, state: torch.Tensor, env: SyntheticLIONEnv) -> int:
        f   = env._flow
        uid = f['unknown_id']

        if f['is_known'] or (uid is not None and env.labels_bought[uid]):
            inf_type = env.inf.known_types_list[f['pred_k']]
            return 0 if inf_type == 'malicious' else 1

        if uid is None:
            return 1

        avail = [i for i in range(env.n_unk) if not env.labels_bought[i]]
        if avail:
            anom_scores = {}
            for i in avail:
                up = env.inf.unk_protos[i].unsqueeze(0)
                with torch.no_grad():
                    anom_scores[i] = torch.cdist(up, env.inf.known_protos).min().item()
            best = max(anom_scores, key=anom_scores.__getitem__)
            if uid == best and env.budget >= env.prices[best] + 2.0:
                return 2

        return 0 if f['anorm_score'] > env.cfg['anomaly_threshold'] * 0.8 else 1

    def push(self, *args): pass
    def train_step(self) -> None: return None


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Episode runner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    total_reward:    float = 0.0
    n_steps:         int   = 0
    win:             bool  = False
    n_labels:        int   = 0
    labels:          List[bool]            = field(default_factory=lambda: [False] * 4)
    budget_final:    float = 0.0
    # label buy events: list of (step_idx, unknown_uid) tuples
    label_buy_steps: List[Tuple[int, int]] = field(default_factory=list)
    # training losses (averaged over all gradient steps in the episode)
    mean_loss:       float = 0.0
    mean_trans_loss: float = 0.0  # DAI-P only


def run_episode(agent, env: SyntheticLIONEnv, train: bool = True) -> EpisodeStats:
    state = env.reset()
    stats = EpisodeStats()
    losses, trans_losses = [], []

    while True:
        action = agent.act(state, env) if isinstance(agent, BreakEvenAgent) \
                 else agent.act(state)

        prev_labels = list(env.labels_bought)
        next_state, reward, done, info = env.step(action)
        stats.total_reward += reward

        # Record label-purchase events (step index inside episode)
        for uid, (before, after) in enumerate(zip(prev_labels, info['labels'])):
            if not before and after:
                stats.label_buy_steps.append((env.t, uid))

        if train:
            agent.push(state, action, reward, next_state, done)
            loss_info = agent.train_step()
            if loss_info:
                if 'loss'       in loss_info: losses.append(loss_info['loss'])
                if 'trans_loss' in loss_info: trans_losses.append(loss_info['trans_loss'])

        state = next_state
        if done:
            stats.n_steps         = env.t
            stats.win             = info['win']
            stats.n_labels        = info['n_labels']
            stats.labels          = list(info['labels'])
            stats.budget_final    = info['budget']
            stats.mean_loss       = float(np.mean(losses))       if losses       else 0.0
            stats.mean_trans_loss = float(np.mean(trans_losses)) if trans_losses else 0.0
            break

    return stats


def smooth(xs: List[float], w: int = 15) -> List[float]:
    out = []
    for i, x in enumerate(xs):
        lo = max(0, i - w)
        out.append(float(np.mean(xs[lo:i+1])))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 10. Agent factory
# ──────────────────────────────────────────────────────────────────────────────

AGENTS = ['DDQN', 'DAI-P', 'Break-Even']


def make_agent(name: str, cfg: dict):
    if name == 'DDQN':      return DDQNAgent(cfg)
    if name == 'DAI-P':     return DAIPAgent(cfg)
    return BreakEvenAgent()


# ──────────────────────────────────────────────────────────────────────────────
# 11. Per-(seed × agent) worker  — top-level so multiprocessing can pickle it
# ──────────────────────────────────────────────────────────────────────────────

def _run_one(task: dict) -> dict:
    """
    Runs one (seed, agent_name) pair in its own process.
    Receives the pretrained InferenceModule as a CPU state dict (picklable).
    """
    seed         = task['seed']
    agent_name   = task['agent_name']
    cfg          = task['cfg']            # already has 'device' set
    n_episodes   = task['n_episodes']
    log_interval = task['log_interval']
    verbose      = task['verbose']
    wandb_kw     = task['wandb_kw']
    env_file     = task['env_file']

    # Re-load .env in each subprocess (spawn doesn't inherit env mutations)
    load_dotenv(env_file)

    torch.manual_seed(seed + 1000 * AGENTS.index(agent_name))
    random.seed(seed + 1000 * AGENTS.index(agent_name))
    np.random.seed(seed + 1000 * AGENTS.index(agent_name))

    device = torch.device(cfg['device'])

    # Reconstruct inference module from serialised state dict
    dg  = DataGenerator(seed=seed)
    inf = InferenceModule(cfg)
    inf.load_state_dict({k: v.to(device) for k, v in task['inf_state_dict'].items()})
    inf.to(device)
    inf.eval()

    agent = make_agent(agent_name, cfg)
    env   = SyntheticLIONEnv(dg, inf, cfg)

    wlog = _WandbLogger(agent_name=agent_name, seed=seed, cfg=cfg, **wandb_kw)

    ep_rewards, ep_wins, ep_labels, ep_labelA, ep_budget = [], [], [], [], []

    for ep in range(n_episodes):
        stats = run_episode(agent, env, train=True)
        ep_rewards.append(stats.total_reward)
        ep_wins.append(int(stats.win))
        ep_labels.append(stats.n_labels)
        ep_labelA.append(int(stats.labels[0]))
        ep_budget.append(stats.budget_final)

        for step_idx, uid in stats.label_buy_steps:
            print(f'    [BUY] ep={ep:4d} seed={seed} {agent_name:10s} '
                  f'→ uid={uid} ({dg.UNKNOWN_NAMES[uid]}/{dg.UNKNOWN_TYPES[uid]}) '
                  f'step={step_idx:3d}  budget={stats.budget_final:.2f}', flush=True)

        if log_interval > 0 and (ep + 1) % log_interval == 0:
            w  = log_interval
            sl = slice(max(0, ep + 1 - w), ep + 1)
            line = (f'    [ep {ep+1:4d}] {agent_name:10s} seed={seed} '
                    f'rew={np.mean(ep_rewards[sl]):+7.2f}  '
                    f'win={np.mean(ep_wins[sl]):.2f}  '
                    f'lA={np.mean(ep_labelA[sl]):.2f}  '
                    f'loss={stats.mean_loss:.4f}')
            if agent_name == 'DAI-P':
                line += f'  tloss={stats.mean_trans_loss:.4f}'
            print(line, flush=True)

        wm = dict(reward=stats.total_reward, win=int(stats.win),
                  n_labels=stats.n_labels, label_A=int(stats.labels[0]),
                  budget_final=stats.budget_final, n_steps=stats.n_steps,
                  loss=stats.mean_loss)
        if agent_name == 'DAI-P':
            wm['trans_loss'] = stats.mean_trans_loss
        wlog.log(wm, step=ep)

    if verbose:
        late = slice(-50, None)
        print(f'  {agent_name:10s}  '
              f'mean_rew={np.mean(ep_rewards[late]):.2f}  '
              f'win_rate={np.mean(ep_wins[late]):.2f}  '
              f'label_A_rate={np.mean(ep_labelA[late]):.2f}  '
              f'mean_labels={np.mean(ep_labels[late]):.2f}', flush=True)

    for phase, sl in [('early', slice(0, 50)), ('late', slice(-50, None))]:
        wlog.summary({
            f'{phase}/mean_reward':  float(np.mean(ep_rewards[sl])),
            f'{phase}/win_rate':     float(np.mean(ep_wins[sl])),
            f'{phase}/label_A_rate': float(np.mean(ep_labelA[sl])),
        })
    wlog.finish()

    return dict(seed=seed, agent=agent_name,
                rewards=ep_rewards, wins=ep_wins,
                n_labels=ep_labels, label_A=ep_labelA, budget=ep_budget)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Experiment orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    n_episodes:    int,
    seeds:         List[int],
    cfg:           dict,
    gpus:          List[str],
    workers_per_gpu: int,
    verbose:       bool = True,
    log_interval:  int  = 50,
    wandb_project: Optional[str] = None,
    wandb_entity:  Optional[str] = None,
    wandb_group:   Optional[str] = None,
    env_file:      str  = '.env',
) -> dict:
    results = {name: {'rewards': [], 'wins': [], 'n_labels': [],
                      'label_A': [], 'budget': []}
               for name in AGENTS}

    if wandb_group is None:
        wandb_group = f'synthetic-lion-{int(time.time())}'

    # ── Pretraining: one per seed, in the main process ────────────────────────
    # Use the first GPU for pretraining (fast; state dict serialised for workers)
    pretrain_device = gpus[0]
    pretrain_cfg    = {**cfg, 'device': pretrain_device}

    pretrained: dict = {}
    for seed in seeds:
        dg  = DataGenerator(seed=seed)
        inf = InferenceModule(pretrain_cfg)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loss = pretrain_inference(inf, dg, pretrain_cfg)
        # Serialise to CPU so workers can load it regardless of their device
        pretrained[seed] = {k: v.cpu() for k, v in inf.state_dict().items()}
        print(f'  Seed {seed}: pretraining done (loss={loss:.4f})')

    # ── Build task list with round-robin GPU assignment ────────────────────────
    slot_cycle = itertools.cycle(gpus * workers_per_gpu)
    wandb_kw   = dict(project=wandb_project, entity=wandb_entity, group=wandb_group)

    tasks = []
    for seed in seeds:
        for agent_name in AGENTS:
            device_str = next(slot_cycle)
            tasks.append(dict(
                seed         = seed,
                agent_name   = agent_name,
                cfg          = {**cfg, 'device': device_str},
                n_episodes   = n_episodes,
                log_interval = log_interval,
                verbose      = verbose,
                wandb_kw     = wandb_kw,
                env_file     = env_file,
                inf_state_dict = pretrained[seed],
            ))

    n_workers = len(gpus) * workers_per_gpu
    print(f'\n  Dispatching {len(tasks)} jobs across {n_workers} worker(s) '
          f'on: {gpus}  (×{workers_per_gpu} per GPU)')
    for t in tasks:
        print(f'    seed={t["seed"]}  {t["agent_name"]:10s}  → {t["cfg"]["device"]}')
    print()

    if n_workers == 1:
        raw = [_run_one(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx
        ) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            raw = [f.result() for f in futures]   # preserves submission order

    for r in raw:
        name = r['agent']
        results[name]['rewards'].append(r['rewards'])
        results[name]['wins'].append(r['wins'])
        results[name]['n_labels'].append(r['n_labels'])
        results[name]['label_A'].append(r['label_A'])
        results[name]['budget'].append(r['budget'])

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 13. Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, n_episodes: int,
                 out_path: str = 'synthetic_lion_results.png') -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available – skipping plot.')
        return ''

    colours  = {'DDQN': '#2196F3', 'DAI-P': '#E91E63', 'Break-Even': '#4CAF50'}
    ls_map   = {'DDQN': '--',       'DAI-P': '-',       'Break-Even': ':'}
    smooth_w = max(1, n_episodes // 20)
    xs       = np.arange(n_episodes)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    n_seeds = len(next(iter(results.values()))['wins'])
    fig.suptitle(
        f'Synthetic LION — DDQN vs. DAI-P vs. Break-Even  ({n_seeds} seeds)',
        fontsize=13, fontweight='bold'
    )

    metrics = [
        ('rewards',  'Episode cumulative reward',  'Reward',   axes[0, 0]),
        ('wins',     'Win rate (rolling avg)',      'Win rate', axes[0, 1]),
        ('n_labels', 'Labels purchased / episode', '# labels', axes[0, 2]),
        ('label_A',  'Label-A purchased (frac.)',  'Label-A',  axes[1, 0]),
        ('budget',   'Final budget',               'Budget',   axes[1, 1]),
    ]

    for key, title, ylabel, ax in metrics:
        for name in AGENTS:
            arr = np.array(results[name][key], dtype=float)
            mu  = arr.mean(0)
            se  = arr.std(0) / max(1, math.sqrt(arr.shape[0]))
            smu = np.array(smooth(list(mu), smooth_w))
            sse = np.array(smooth(list(se), smooth_w))
            ax.plot(xs, smu, label=name, color=colours[name], lw=2.2, ls=ls_map[name])
            ax.fill_between(xs, smu - sse, smu + sse, alpha=0.18, color=colours[name])
        ax.axvline(50, color='grey', lw=0.8, ls='--', alpha=0.6)
        ax.text(52, ax.get_ylim()[0], 'ep 50', fontsize=7, color='grey')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[1, 2].axis('off')
    txt = (
        'Experimental design\n'
        '───────────────────\n'
        '7 Gaussian classes:\n'
        '  K0,K1 benign  K2 malicious (known)\n'
        '  A malicious overlapping K0  ← TRAP\n'
        '  B malicious separated\n'
        '  C benign  separated\n'
        '  D benign  near K2\n\n'
        'Budget 15 < total label cost 19:\n'
        '  must choose ≤ 3 labels wisely.\n\n'
        'Uninformed A-flow accepted → −7\n'
        'After label-A: blocked     → +2.5\n\n'
        'Break-Even: buys highest-anomaly\n'
        '  cluster first (= B, always).\n'
        '  A has LOW anomaly score → last.\n\n'
        'DAI-P epistemic gain (learned):\n'
        '  buy-label-A zeroes conf_unk_0\n'
        '  → largest proprio state change\n'
        '  → highest transition MSE\n'
        '  → highest intrinsic value\n'
        '  → faster early convergence.\n'
    )
    axes[1, 2].text(0.04, 0.97, txt, transform=axes[1, 2].transAxes,
                    fontsize=8.5, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'Plot saved → {out_path}')
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 14. Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Synthetic LION experiment')
    parser.add_argument('--episodes',          type=int,  default=300)
    parser.add_argument('--seeds',             type=int,  default=5)
    parser.add_argument('--no-plot',           action='store_true')
    parser.add_argument('--verbose',           action='store_true', default=True)
    parser.add_argument('--log-interval',      type=int,  default=50,
                        help='Print running stats every N episodes (0=off)')
    # GPU / parallelism
    parser.add_argument('--gpus',              type=str,  default='0',
                        help='"0", "0,1,2", "auto" (free GPUs), or "cpu"')
    parser.add_argument('--workers-per-gpu',   type=int,  default=1,
                        help='Parallel workers per GPU (default 1)')
    parser.add_argument('--min-free-gpu-gb',   type=float, default=10.0,
                        help='Min free VRAM (GB) for --gpus auto (default 10)')
    # Wandb
    parser.add_argument('--wandb-project',     type=str,  default='LION')
    parser.add_argument('--wandb-entity',      type=str,  default='jfcevallos')
    parser.add_argument('--wandb-group',       type=str,  default=None)
    parser.add_argument('--env-file',          type=str,  default='.env')
    parser.add_argument('--plot-path',         type=str,
                        default='synthetic_lion_results.png')
    args = parser.parse_args()

    load_dotenv(args.env_file)

    gpus = _parse_gpus(args.gpus, args.min_free_gpu_gb)

    print('=' * 65)
    print(' Synthetic LION experiment')
    print(f'  Episodes : {args.episodes}   Seeds : {args.seeds}')
    print(f'  GPUs     : {gpus}   Workers/GPU : {args.workers_per_gpu}')
    if args.wandb_project:
        print(f'  W&B      : project={args.wandb_project}  '
              f'entity={args.wandb_entity or "(default)"}')
    print('=' * 65)
    print()

    dg_preview = DataGenerator()
    print('Data layout:')
    for k in range(dg_preview.N_KNOWN):
        print(f'  Known  {k} ({dg_preview.KNOWN_TYPES[k]:8s}): μ={dg_preview.KNOWN_MEANS[k]}')
    for u in range(dg_preview.N_UNKNOWN):
        print(f'  Unknown {dg_preview.UNKNOWN_NAMES[u]:20s}: '
              f'μ={dg_preview.UNKNOWN_MEANS[u]}  price={CFG["label_prices"][u]}')
    print()

    t0      = time.time()
    seeds   = list(range(args.seeds))
    results = run_experiment(
        n_episodes      = args.episodes,
        seeds           = seeds,
        cfg             = CFG,
        gpus            = gpus,
        workers_per_gpu = args.workers_per_gpu,
        verbose         = args.verbose,
        log_interval    = args.log_interval,
        wandb_project   = args.wandb_project,
        wandb_entity    = args.wandb_entity,
        wandb_group     = args.wandb_group,
        env_file        = args.env_file,
    )
    elapsed = time.time() - t0

    # ── Summary tables ────────────────────────────────────────────────────────
    header = (f'{"Agent":12s}  {"MeanRew":>9}  {"WinRate":>8}  '
              f'{"LabelA":>8}  {"±σ WinRate":>11}')
    for label, slc in [('EARLY (ep 1-50)', slice(0, 50)),
                        ('LATE  (ep last 50)', slice(-50, None))]:
        print()
        print('─' * 65)
        print(f'  {label}')
        print('─' * 65)
        print(header)
        print('─' * 65)
        for name in AGENTS:
            rews    = np.array(results[name]['rewards'])[:, slc].mean()
            arr_w   = np.array(results[name]['wins'])[:, slc]
            wins    = arr_w.mean()
            wins_sd = arr_w.mean(axis=1).std()
            labelA  = np.array(results[name]['label_A'])[:, slc].mean()
            print(f'{name:12s}  {rews:>9.2f}  {wins:>8.3f}  '
                  f'{labelA:>8.3f}  {wins_sd:>11.3f}')
        print('─' * 65)

    print(f'\nTotal wall-clock time: {elapsed:.1f}s  '
          f'({elapsed / (args.seeds * len(AGENTS)):.1f}s per run avg)')

    if not args.no_plot:
        plot_results(results, args.episodes, out_path=args.plot_path)


if __name__ == '__main__':
    main()
