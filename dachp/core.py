"""DACHP encoder, DACHP loss, deep metric-learning baselines and a single shared trainer.

All deep methods (DACHP and the six deep baselines) use the SAME encoder backbone,
optimiser, schedule, batching rule and epoch budget; only the objective differs.
This is the controlled comparison requested by Reviewer #4 (comment 1) and Reviewer #3.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")  # all computation runs on the CPU


# ----------------------------------------------------------------------------- encoder
def hidden_dims(D: int, N: int, rule: str = "adaptive"):
    """Capacity rule, Eqs. (1)-(2). N is the number of TRAINING samples."""
    if rule == "fixed256":
        return 256, 128
    if rule == "uncapped" or N >= D:
        return max(128, 2 * D), max(64, D)
    return max(128, min(2 * D, N // 2)), max(64, min(D, N // 4))


class Encoder(nn.Module):
    def __init__(self, D, N, latent_dim=16, dropout=0.1, residual=True, rule="adaptive",
                 sphere=True, proj_head=True):
        super().__init__()
        H1, H2 = hidden_dims(D, N, rule)
        self.H = (H1, H2)
        self.fc1, self.bn1 = nn.Linear(D, H1), nn.BatchNorm1d(H1)
        self.fc2, self.fc3 = nn.Linear(H1, H2), nn.Linear(H2, latent_dim)
        self.drop = nn.Dropout(dropout)
        self.residual = residual
        # Phi(h1) = W_skip h1  (W_skip in R^{H2 x H1}); identity when H1 == H2
        self.skip = (nn.Linear(H1, H2, bias=False) if H1 != H2 else nn.Identity()) if residual else None
        self.sphere = sphere
        self.proj = nn.Sequential(nn.Linear(latent_dim, 2 * latent_dim), nn.ReLU(),
                                  nn.Linear(2 * latent_dim, latent_dim)) if proj_head else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def raw(self, x):
        h1 = self.drop(F.relu(self.bn1(self.fc1(x))))
        h2 = F.relu(self.fc2(h1))
        if self.residual:
            h2 = h2 + self.skip(h1)
        return self.fc3(h2)

    def forward(self, x):
        z = self.raw(x)
        return F.normalize(z, dim=1) if self.sphere else z

    @torch.no_grad()
    def embed(self, X, bs=4096):
        self.eval()
        out = []
        for i in range(0, len(X), bs):
            out.append(self(torch.as_tensor(X[i:i + bs], device=DEVICE)).cpu())
        return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- losses
def pdist(z):
    """Eq. (9). Valid for normalised and un-normalised z; the eps clamp only guards
    the gradient of sqrt at 0 (d/dx sqrt(x) is unbounded at x=0)."""
    sq = (z * z).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * z @ z.T
    return torch.sqrt(torch.clamp(d2, min=1e-8))


def masks(y):
    same = (y[:, None] == y[None, :]).float()
    eye = torch.eye(len(y), device=y.device)
    pos = same * (1 - eye)
    neg = (1 - same)            # diagonal is already 0 because y_i == y_i
    return pos, neg


def dachp_main_loss(z, y, alpha=1.2, eps_pos=0.05, severe=True, hardneg=True, dmax=2.0):
    D = pdist(z)
    pos, neg = masks(y)
    l_pos = ((D ** 2) * pos * (1 - eps_pos)).sum() / pos.sum().clamp_min(1)
    if hardneg:
        h = torch.clamp(alpha - D, min=0)
        pen = h ** 2 + ((D < alpha / 2).float() * h if severe else 0.0)
    else:
        # ablation: every negative pair is penalised (no margin-based selection);
        # pushes all negatives towards the antipodal distance dmax=2 of the unit sphere
        pen = torch.clamp(dmax - D, min=0) ** 2
    l_neg = (pen * neg).sum() / neg.sum().clamp_min(1)
    return l_pos + l_neg


def supcon_loss(z, y, tau):
    """Supervised contrastive loss (Khosla et al., 2020, L_out), single view."""
    sim = z @ z.T / tau
    n = len(y)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    loss = -(log_prob * pos).sum(1)[valid] / npos[valid]
    return loss.mean() if valid.any() else z.sum() * 0


def triplet_batch_hard(z, y, margin=0.2):
    D = pdist(z)
    pos, neg = masks(y)
    hardest_pos = (D * pos).max(1).values
    hardest_neg = (D + 1e6 * (1 - neg)).min(1).values
    valid = pos.sum(1) > 0
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()


def proxy_anchor_loss(z, y, proxies, alpha=32.0, delta=0.1):
    P = F.normalize(proxies, dim=1)
    cos = z @ P.T                                        # (n, C)
    onehot = F.one_hot(y, P.shape[0]).float()
    pos_exp = torch.exp(-alpha * (cos - delta)) * onehot
    neg_exp = torch.exp(alpha * (cos + delta)) * (1 - onehot)
    with_pos = onehot.sum(0) > 0
    pos_term = torch.log1p(pos_exp.sum(0))[with_pos].sum() / with_pos.sum().clamp_min(1)
    neg_term = torch.log1p(neg_exp.sum(0)).sum() / P.shape[0]
    return pos_term + neg_term


def margin_softmax_loss(z, y, W, s=30.0, m=0.5, kind="arcface"):
    cos = z @ F.normalize(W, dim=1).T
    cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
    onehot = F.one_hot(y, W.shape[0]).bool()
    if kind == "arcface":
        target = torch.cos(torch.acos(cos) + m)
    else:  # cosface
        target = cos - m
    logits = torch.where(onehot, target, cos) * s
    return F.cross_entropy(logits, y)


# ----------------------------------------------------------------------------- trainer
METHODS_DEEP = ["dachp", "supcon", "triplet", "proxyanchor", "arcface", "cosface", "ce_mlp"]


def _collapse_stats(model, Xmon, ymon):
    with torch.no_grad():
        model.eval()
        zr = model.raw(Xmon)
        z = F.normalize(zr, dim=1)
        C = z @ z.T
        n = len(z)
        off = ~torch.eye(n, dtype=torch.bool, device=z.device)
        same = (ymon[:, None] == ymon[None, :]) & off
        diff = ymon[:, None] != ymon[None, :]
        c_off = C[off]
        s = torch.linalg.svdvals(z)
        p = s / s.sum()
        erank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
        model.train()
        return dict(
            cos_mean=c_off.mean().item(), cos_var=c_off.var().item(),
            cos_intra=C[same].mean().item(), cos_inter=C[diff].mean().item(),
            resultant=z.mean(0).norm().item(),          # 1.0 <=> complete angular collapse
            eff_rank=erank.item(), raw_norm=zr.norm(dim=1).mean().item(),
        )


def train_embedder(Xtr, ytr, method="dachp", hp=None, epochs=200, seed=0, monitor_every=0,
                   monitor_n=1000, verbose=False):
    """Train one deep embedding model. Returns (model, info)."""
    hp = dict(hp or {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, D = Xtr.shape
    C = int(ytr.max()) + 1
    latent = hp.get("latent_dim", 16)
    sphere = hp.get("sphere", True) if method == "dachp" else method != "ce_mlp"
    model = Encoder(D, N, latent, hp.get("dropout", 0.1), hp.get("residual", True),
                    hp.get("capacity_rule", "adaptive"), sphere=sphere,
                    proj_head=(method == "dachp" and hp.get("lambda_aux", 0.3) > 0)).to(DEVICE)

    extra, groups = nn.ParameterDict(), []
    if method == "dachp":
        tau0 = hp.get("tau_init", 0.07)
        extra["log_tau"] = nn.Parameter(torch.tensor(math.log(tau0), device=DEVICE),
                                        requires_grad=hp.get("tau_learnable", True))
    elif method == "proxyanchor":
        pr = torch.empty(C, latent, device=DEVICE)
        nn.init.kaiming_normal_(pr, mode="fan_out")
        extra["proxies"] = nn.Parameter(pr)
    elif method in ("arcface", "cosface"):
        w = torch.empty(C, latent, device=DEVICE)
        nn.init.xavier_uniform_(w)
        extra["W"] = nn.Parameter(w)
    elif method == "ce_mlp":
        model.cls = nn.Linear(latent, C).to(DEVICE)

    lr = hp.get("lr", 0.005)
    groups.append({"params": list(model.parameters()), "lr": lr})
    for k, p in extra.items():
        mult = hp.get("proxy_lr_mult", 100) if k == "proxies" else 1.0
        if p.requires_grad:
            groups.append({"params": [p], "lr": lr * mult})
    opt = torch.optim.Adam(groups, betas=tuple(hp.get("betas", (0.9, 0.999))),
                           weight_decay=hp.get("weight_decay", 0.0))
    warm = max(5, epochs // 10)
    floor = hp.get("min_lr", 1e-5) / lr

    def lam(ep):
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(1, epochs - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    Xt = torch.as_tensor(Xtr, device=DEVICE)
    yt = torch.as_tensor(ytr, device=DEVICE, dtype=torch.long)
    thr = hp.get("minibatch_threshold", 2000)
    bs = N if N <= thr else (hp.get("batch_large", 512) if N > 5000 else hp.get("batch_medium", 256))
    g = torch.Generator(device="cpu").manual_seed(seed)
    mon_idx = torch.randperm(N, generator=g)[:monitor_n].to(DEVICE)
    tau_lo, tau_hi = hp.get("tau_clamp", (0.01, 1.0))

    def objective(xb, yb):
        z = model(xb)
        parts = {}
        if method == "dachp":
            l_main = dachp_main_loss(z, yb, hp.get("alpha", 1.2), hp.get("eps_pos", 0.05),
                                     hp.get("severe_term", True), hp.get("hardneg", True))
            parts["main"] = l_main.item()
            lam_aux = hp.get("lambda_aux", 0.3)
            if lam_aux > 0:
                tau = torch.exp(extra["log_tau"])
                if not hp.get("legacy_tau", False):
                    tau = tau.clamp(tau_lo, tau_hi)
                zs = F.normalize(z, dim=1)
                p = F.normalize(model.proj(zs), dim=1)
                l_aux = supcon_loss(torch.cat([zs.detach(), p]), torch.cat([yb, yb]), tau)
                parts["aux"] = l_aux.item()
                return l_main + lam_aux * l_aux, parts
            return l_main, parts
        if method == "supcon":
            return supcon_loss(z, yb, hp.get("tau", 0.1)), parts
        if method == "triplet":
            return triplet_batch_hard(z, yb, hp.get("margin", 0.2)), parts
        if method == "proxyanchor":
            return proxy_anchor_loss(z, yb, extra["proxies"], hp.get("alpha", 32), hp.get("delta", 0.1)), parts
        if method in ("arcface", "cosface"):
            return margin_softmax_loss(z, yb, extra["W"], hp.get("s", 30.0), hp.get("m", 0.5), method), parts
        if method == "ce_mlp":
            return F.cross_entropy(model.cls(z), yb, label_smoothing=hp.get("label_smoothing", 0.0)), parts
        raise KeyError(method)

    t0 = time.perf_counter()
    hist, mon = [], []
    params_all = [p for gr in groups for p in gr["params"]]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE) if bs < N else torch.arange(N, device=DEVICE)
        tot, agg, nb = 0.0, {}, 0
        for b in perm.split(bs):
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss, parts = objective(Xt[b], yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_all, hp.get("grad_clip", 5.0))
            opt.step()
            tot += loss.item()
            nb += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        rec = {"epoch": ep + 1, "total": tot / max(nb, 1), **{k: v / max(nb, 1) for k, v in agg.items()}}
        if "log_tau" in extra:
            rec["tau"] = float(torch.exp(extra["log_tau"]).item())
        hist.append(rec)
        if monitor_every and (ep == 0 or (ep + 1) % monitor_every == 0):
            mon.append({"epoch": ep + 1, **_collapse_stats(model, Xt[mon_idx], yt[mon_idx])})
        if verbose and (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}/{epochs} loss={rec['total']:.4f}")
    peak = float("nan")  # peak accelerator memory is not measured in the CPU-only version
    info = dict(train_seconds=time.perf_counter() - t0, peak_mem_mb=peak, H=model.H, batch_size=bs,
                history=hist, monitor=mon, n_params=sum(p.numel() for p in params_all),
                final_tau=hist[-1].get("tau") if hist else None, device=str(DEVICE))
    return model, info
