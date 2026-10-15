#!/usr/bin/env python
"""Run every experiment of the DACHP revision under one held-out protocol.

Usage:
    pip install -r requirements.txt
    python run_experiments.py                       # everything, all datasets (Experiment 1)
    python run_experiments.py --datasets mnist har  # a subset of datasets, e.g. to run in parallel
    python run_experiments.py --stages main ablation
    python run_experiments.py --config config_v2.yaml --out results_v2 --stages diagnostics tune main  # Experiment 2

Stages (in priority order): diagnostics, tune, main, ablation, permutation, sensitivity.
Every finished run is appended to <out>/runs.jsonl and is skipped on restart, so the
script can be interrupted and resumed without losing work.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback

import numpy as np
import yaml

from dachp.baselines import available, run_classical
from dachp.core import DEVICE, train_embedder
from dachp.data import load_dataset, prepared_split, regime
from dachp.metrics import evaluate, input_diagnostics

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_KEYS = ["latent_dim", "dropout", "residual", "capacity_rule", "lr", "betas", "weight_decay",
              "warmup_frac", "min_lr", "grad_clip", "minibatch_threshold", "batch_large", "batch_medium"]
DEEP_BASELINES = ["supcon", "triplet", "proxyanchor", "arcface", "cosface", "ce_mlp"]
CLASSICAL = ["pca", "lda", "tsne", "pacmap", "trimap", "umap", "sup_umap"]
ABLATION_OVERRIDES = {
    "full": {},
    "no_sphere": {"sphere": False},
    "no_hardneg": {"hardneg": False},
    "no_severe": {"severe_term": False},
    "no_aux": {"lambda_aux": 0.0},
    "no_residual": {"residual": False},
    "uncapped": {"capacity_rule": "uncapped"},
    "fixed256": {"capacity_rule": "fixed256"},
    "legacy_tau": {"tau_init": float(np.exp(2.659)), "legacy_tau": True},   # the original submission's tau bug
}
FIG_METHODS = ["dachp", "supcon", "sup_umap_tuned", "umap", "tsne", "pca"]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class Store:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.out = out
        self.path = os.path.join(out, "runs.jsonl")
        self.recs = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                try:
                    r = json.loads(line)
                    self.recs[r["key"]] = r
                except Exception:
                    pass

    @staticmethod
    def key(**kw):
        return hashlib.sha1(json.dumps(jsonable(kw), sort_keys=True).encode()).hexdigest()[:16]

    def has(self, k):
        return k in self.recs

    def add(self, rec):
        self.recs[rec["key"]] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(jsonable(rec)) + "\n")


class Runner:
    def __init__(self, cfg, store, quick=False):
        self.cfg, self.store, self.quick = cfg, store, quick
        self.ev = cfg["evaluation"]
        self._ds, self._split = {}, {}

    # ---------------------------------------------------------------- data
    def ds(self, key):
        if key not in self._ds:
            self._ds[key] = load_dataset(key, self.cfg["datasets"].get(key, {}))
        return self._ds[key]

    def split(self, key, seed):
        if (key, seed) not in self._split:
            s = self.cfg["split"]
            self._split[(key, seed)] = prepared_split(self.ds(key), seed, (s["train"], s["val"], s["test"]))
        return self._split[(key, seed)]

    def epochs(self, key):
        e = self.cfg["datasets"].get(key, {}).get("epochs", 200)
        return max(5, e // 20) if self.quick else e

    # ---------------------------------------------------------------- hyper-parameters
    def base_hp(self):
        d = self.cfg["dachp"]
        return {k: d[k] for k in TRAIN_KEYS if k in d}

    def dachp_hp(self):
        return {k: v for k, v in self.cfg["dachp"].items() if k != "tune_grid"}

    def deep_hp(self, m):
        b = self.cfg["deep_baselines"][m]
        return {**self.base_hp(), **{k: v for k, v in b.items() if k != "tune_grid"}}

    def grid(self, method):
        if method == "dachp":
            g = self.cfg["dachp"]["tune_grid"]
        elif method == "dachp_v2":
            g = self.cfg["dachp_v2"]["tune_grid"]
        elif method in DEEP_BASELINES:
            g = self.cfg["deep_baselines"][method]["tune_grid"]
        elif method == "sup_umap":
            g = self.cfg["classical_baselines"]["sup_umap"]["tune_grid"]
        else:
            return [{}]
        keys = list(g)
        return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]

    # ---------------------------------------------------------------- single runs
    def _deep(self, stage, dkey, method, hp, seed, eval_on="test", perm=False, save_hist=False, tag=None, name=None):
        name = name or method
        k = Store.key(stage=stage if stage in ("tune",) else "run", d=dkey, m=name, hp=hp, s=seed,
                      e=eval_on, perm=perm, q=self.quick)
        if self.store.has(k):
            rec = self.store.recs[k]
            if tag and tag not in rec.get("tags", []):
                rec.setdefault("tags", []).append(tag)
                self.store.add(rec)
            return rec
        sp = self.split(dkey, seed)
        ytr = sp["ytr"].copy()
        if perm:
            ytr = np.random.RandomState(seed + 1000).permutation(ytr)
        model, info = train_embedder(sp["Xtr"], ytr, method, hp, self.epochs(dkey), seed,
                                     monitor_every=self.ev["monitor_every"] if save_hist else 0,
                                     monitor_n=self.ev["monitor_n"])
        Ztr, Zev = model.embed(sp["Xtr"]), model.embed(sp["X" + ("te" if eval_on == "test" else "va")])
        if method == "ce_mlp":
            Ztr = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True)
            Zev = Zev / np.linalg.norm(Zev, axis=1, keepdims=True)
        yev = sp["y" + ("te" if eval_on == "test" else "va")]
        Xev = sp["X" + ("te" if eval_on == "test" else "va")]
        met = evaluate(Ztr, sp["ytr"], Zev, yev, Xev, k=self.ev["knn_k"], trust_k=self.ev["trust_k"],
                       trust_max_n=self.ev["trust_max_n"], sil_max_n=self.ev["silhouette_max_n"])
        if not save_hist:
            info.pop("history", None)
            info.pop("monitor", None)
        rec = dict(key=k, stage=stage, dataset=dkey, method=name, hp=hp, seed=seed, eval_on=eval_on,
                   perm=perm, metrics=met, info=info, tags=[tag] if tag else [], time=time.time())
        self.store.add(rec)
        if seed == self.cfg["seeds"][0] and eval_on == "test" and not perm and tag in ("main",):
            self._save_fig_embedding(dkey, name, Ztr, Zev, yev)
        return rec

    def _classical(self, stage, dkey, method, params, seed, eval_on="test", name=None, tag=None):
        name = name or method
        k = Store.key(stage=stage if stage == "tune" else "run", d=dkey, m=name, hp=params, s=seed, e=eval_on,
                      q=self.quick)
        if self.store.has(k):
            return self.store.recs[k]
        sp = self.split(dkey, seed)
        Xev = sp["Xte"] if eval_on == "test" else sp["Xva"]
        yev = sp["yte"] if eval_on == "test" else sp["yva"]
        Ztr, Zev, info = run_classical(method, sp["Xtr"], sp["ytr"], Xev, seed, params)
        met = evaluate(Ztr, sp["ytr"], Zev, yev, Xev, k=self.ev["knn_k"], trust_k=self.ev["trust_k"],
                       trust_max_n=self.ev["trust_max_n"], sil_max_n=self.ev["silhouette_max_n"])
        rec = dict(key=k, stage=stage, dataset=dkey, method=name, hp=params, seed=seed, eval_on=eval_on,
                   perm=False, metrics=met, info=info, tags=[tag] if tag else [], time=time.time())
        self.store.add(rec)
        if seed == self.cfg["seeds"][0] and eval_on == "test" and name in FIG_METHODS:
            self._save_fig_embedding(dkey, name, Ztr, Zev, yev)
        return rec

    def _save_fig_embedding(self, dkey, name, Ztr, Zte, yte):
        from sklearn.decomposition import PCA
        Z2 = PCA(2, random_state=0).fit(Ztr).transform(Zte) if Zte.shape[1] > 2 else Zte[:, :2]
        d = os.path.join(self.store.out, "embeddings")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{dkey}__{name}.npz"), Z=Z2.astype(np.float32), y=yte)

    # ---------------------------------------------------------------- stages
    def stage_diagnostics(self, dkey):
        k = Store.key(stage="diag", d=dkey)
        if self.store.has(k):
            return
        ds = self.ds(dkey)
        sp = self.split(dkey, 0)
        diag = input_diagnostics(sp["Xtr"], sp["ytr"])
        tr, va, te = sp["idx"]
        self.store.add(dict(key=k, stage="diag", dataset=dkey, method=None, seed=0, metrics=diag,
                            info=dict(name=ds.name, domain=ds.domain, N=ds.N, D=ds.D, C=ds.C, regime=regime(ds.D),
                                      n_train=len(tr), n_val=len(va), n_test=len(te), note=ds.note,
                                      grouped=ds.groups is not None,
                                      class_counts=np.bincount(ds.y).tolist()),
                            tags=[], time=time.time()))

    def selected(self, dkey, method):
        """Best configuration on the split-0 validation set: max val k-NN acc, tie-break val native Silhouette."""
        s = self.cfg["tuning_seed"]
        best, best_score = None, None
        for g in self.grid(method):
            if method == "dachp":
                hp = {**self.dachp_hp(), **g}
                r = self._deep("tune", dkey, "dachp", hp, s, eval_on="val")
            elif method == "dachp_v2":
                hp = {**self.dachp_hp(), **{k: v for k, v in self.cfg["dachp_v2"].items() if k != "tune_grid"}, **g}
                r = self._deep("tune", dkey, "dachp", hp, s, eval_on="val", name="dachp_v2")
            elif method in DEEP_BASELINES:
                hp = {**self.deep_hp(method), **g}
                r = self._deep("tune", dkey, method, hp, s, eval_on="val")
            elif method == "sup_umap":
                hp = {**self.cfg["classical_baselines"]["sup_umap"]["default"], **g}
                r = self._classical("tune", dkey, "sup_umap", hp, s, eval_on="val", name="sup_umap")
            m = r["metrics"]
            sc = (round(m["knn_acc"], 6), m["sil_native"] if np.isfinite(m["sil_native"]) else -9)
            if best_score is None or sc > best_score:
                best, best_score = hp, sc
        return best

    def stage_tune(self, dkey):
        for m in [*self.cfg.get("deep_methods", ["dachp", *DEEP_BASELINES]), "sup_umap"]:
            self.selected(dkey, m)

    def stage_main(self, dkey):
        cb = self.cfg["classical_baselines"]
        for seed in self.cfg["seeds"]:
            for m in self.cfg.get("deep_methods", ["dachp", *DEEP_BASELINES]):
                train_m = "dachp" if m.startswith("dachp") else m
                self._deep("main", dkey, train_m, self.selected(dkey, m), seed, save_hist=m.startswith("dachp"),
                           tag="main", name=m)
            for m in self.cfg.get("classical_methods", CLASSICAL):
                if not available(m):
                    continue
                if m == "sup_umap":
                    self._classical("main", dkey, "sup_umap", cb["sup_umap"]["default"], seed, name="sup_umap_default", tag="main")
                    tuned = self.selected(dkey, "sup_umap")
                    self._classical("main", dkey, "sup_umap", tuned, seed, name="sup_umap_tuned", tag="main")
                    if cb["sup_umap"].get("also_16d"):
                        self._classical("main", dkey, "sup_umap", {**tuned, "n_components": 16}, seed,
                                        name="sup_umap_tuned_16d", tag="main")
                else:
                    self._classical("main", dkey, m, cb.get(m, {}), seed, tag="main")

    def stage_ablation(self, dkey):
        base = self.selected(dkey, "dachp")
        for v in self.cfg["ablation"]["variants"]:
            hp = {**base, **ABLATION_OVERRIDES[v]}
            for seed in self.cfg["seeds"]:
                self._deep("ablation", dkey, "dachp", hp, seed, save_hist=v in ("full", "no_sphere", "legacy_tau"),
                           tag=f"ablation:{v}")

    def stage_permutation(self, dkey):
        base = self.selected(dkey, "dachp")
        for seed in self.cfg["seeds"]:
            self._deep("permutation", dkey, "dachp", base, seed, perm=True, tag="permutation")

    def stage_sensitivity(self, dkey):
        base = self.selected(dkey, "dachp")
        for p, vals in self.cfg["sensitivity"].items():
            for v in vals:
                hp = {**base, p: v}
                for seed in self.cfg["sensitivity_seeds"]:
                    self._deep("sensitivity", dkey, "dachp", hp, seed, tag=f"sens:{p}={v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--stages", nargs="*", default=["diagnostics", "tune", "main", "ablation", "permutation", "sensitivity"])
    ap.add_argument("--quick", action="store_true", help="tiny epoch budget; for smoke tests only")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seeds:
        cfg["seeds"] = a.seeds
        cfg["sensitivity_seeds"] = [s for s in cfg["sensitivity_seeds"] if s in a.seeds] or a.seeds[:1]
    datasets = a.datasets or list(cfg["datasets"])
    store = Store(a.out)
    json.dump(cfg, open(os.path.join(a.out, "config_used.json"), "w"), indent=1)
    R = Runner(cfg, store, quick=a.quick)
    print(f"device={DEVICE}  datasets={datasets}  stages={a.stages}  quick={a.quick}", flush=True)
    for stage in a.stages:
        for d in datasets:
            t0 = time.time()
            try:
                getattr(R, f"stage_{stage}")(d)
                print(f"[{stage:<12}] {d:<14} done in {time.time()-t0:7.1f}s  (runs stored: {len(store.recs)})", flush=True)
            except Exception as e:
                print(f"[{stage:<12}] {d:<14} FAILED: {e}", flush=True)
                traceback.print_exc()
    print("finished.")


if __name__ == "__main__":
    sys.exit(main())
