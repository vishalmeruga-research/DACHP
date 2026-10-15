#!/usr/bin/env python
"""Aggregate runs.jsonl into every table, statistic and figure of the revised manuscript.

    python make_report.py --results results [results_shard2 ...] --out report

Outputs (report/):
  tables/*.csv  + tables/*.md       every table of the paper
  figures/*.png (300 dpi) + *.pdf   every figure of the paper
  summary.json                      every number quoted in the text / response letter
  REPORT.md                         human-readable digest
All numbers in the manuscript must come from this folder (Reviewer #2, comment 6).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_LABEL = {
    "dachp": "DACHP (ours)", "dachp_v2": "DACHP-v2 (ours)", "supcon": "SupCon (same enc.)", "triplet": "Triplet (batch-hard)",
    "proxyanchor": "Proxy-Anchor", "arcface": "ArcFace", "cosface": "CosFace", "ce_mlp": "CE-MLP (norm.)",
    "sup_umap_tuned": "Sup-UMAP (tuned)", "sup_umap_tuned_16d": "Sup-UMAP (tuned, 16-D)",
    "sup_umap_default": "Sup-UMAP (default)", "umap": "UMAP", "lda": "LDA", "pca": "PCA",
    "tsne": "t-SNE", "pacmap": "PaCMAP", "trimap": "TriMap",
}
METHOD_ORDER = list(METHOD_LABEL)
SUPERVISED = ["dachp", "dachp_v2", "supcon", "triplet", "proxyanchor", "arcface", "cosface", "ce_mlp",
              "sup_umap_tuned", "sup_umap_tuned_16d", "sup_umap_default", "lda"]
DS_ORDER = ["breast_cancer", "digits", "mice_protein", "anneal", "coil20", "cnae9", "mnist",
            "fashion_mnist", "har", "isolet", "olivetti"]
METRIC_LABEL = {
    "knn_acc": "Test 5-NN accuracy", "linear_acc": "Test linear-probe accuracy", "ncm_acc": "Test nearest-class-mean accuracy",
    "sil_native": "Test Silhouette (native dim.)", "sil_2d": "Test Silhouette (2-D)", "trust": "Test trustworthiness (k=10)",
    "db_native": "Test Davies-Bouldin (native)", "ch_native": "Test Calinski-Harabasz (native)",
    "sil_native_train": "TRAIN Silhouette (native, transductive)",
}
PRIMARY = "knn_acc"
ABL_LABEL = {
    "full": "Full DACHP", "no_sphere": "- Hypersphere (no L2-norm.)", "no_hardneg": "- Hard-neg. selection (all negatives)",
    "no_severe": "- Severe-violation term", "no_aux": "- Auxiliary SupCon (lambda=0)", "no_residual": "- Residual skip",
    "uncapped": "- N-cap (uncapped widths)", "fixed256": "Fixed widths 256/128", "legacy_tau": "τ₀ = 14.3 (no clamp)",
}
# Okabe-Ito (CVD-safe); DACHP always vermilion
C_DACHP, C_GREY = "#D55E00", "#8C8C8C"
PAL = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000", "#F0E442"]


# ----------------------------------------------------------------------------- load
def load(paths):
    recs = {}
    for p in paths:
        for f in glob.glob(os.path.join(p, "runs.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                    recs[r["key"]] = r
                except Exception:
                    pass
    return list(recs.values())


def ci95(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan, len(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h, len(x)


def fmt(m, h, d=3):
    if not np.isfinite(m):
        return "–"
    return f"{m:.{d}f}" if not np.isfinite(h) else f"{m:.{d}f} ± {h:.{d}f}"


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def boot_ci(d, B=10000, seed=0):
    d = np.asarray(d, float)
    r = np.random.RandomState(seed)
    bs = d[r.randint(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(bs, [2.5, 97.5])


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)



def _peak_mem(rec):
    """Peak memory in MB stored in a run record (older records use the key 'peak_gpu_mb')."""
    info = rec["info"]
    return float(info.get("peak_mem_mb", info.get("peak_gpu_mb", np.nan)))

class Report:
    def __init__(self, recs, out, focal="dachp"):
        self.recs, self.out, self.focal = recs, out, focal
        self.tdir, self.fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
        os.makedirs(self.tdir, exist_ok=True)
        os.makedirs(self.fdir, exist_ok=True)
        self.S = {}          # summary.json
        self.md = ["# DACHP revision — generated results\n"]
        self.diag = {r["dataset"]: r for r in recs if r["stage"] == "diag"}
        self.datasets = [d for d in DS_ORDER if d in self.diag] + sorted(
            {r["dataset"] for r in recs if r["stage"] == "diag"} - set(DS_ORDER))
        self.dname = {d: self.diag[d]["info"]["name"] for d in self.datasets}
        # main records: method -> dataset -> seed -> metrics
        self.main = defaultdict(lambda: defaultdict(dict))
        self.main_info = defaultdict(lambda: defaultdict(dict))
        for r in recs:
            if "main" in r.get("tags", []) and r["eval_on"] == "test" and not r.get("perm"):
                self.main[r["method"]][r["dataset"]][r["seed"]] = r["metrics"]
                self.main_info[r["method"]][r["dataset"]][r["seed"]] = r
        self.methods = [m for m in METHOD_ORDER if m in self.main]
        self.nseeds = max((len(v) for m in self.main.values() for v in m.values()), default=0)

    def save(self, name, df, caption=""):
        df.to_csv(os.path.join(self.tdir, name + ".csv"), index=False)
        with open(os.path.join(self.tdir, name + ".md"), "w") as f:
            f.write((caption + "\n\n" if caption else "") + md_table(df) + "\n")
        self.md.append(f"\n## {name}\n{caption}\n\n{md_table(df)}\n")

    def savefig(self, fig, name):
        fig.savefig(os.path.join(self.fdir, name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(self.fdir, name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def vals(self, m, d, metric):
        return [v[metric] for v in self.main[m][d].values() if metric in v]

    # ------------------------------------------------------------------ Table: datasets
    def t_datasets(self):
        rows = []
        for d in self.datasets:
            i, g = self.diag[d]["info"], self.diag[d]["metrics"]
            rows.append({"Dataset": i["name"], "Domain": i["domain"], "N": i["N"], "D": i["D"], "C": i["C"],
                         "Regime": i["regime"], "N/D": round(i["N"] / i["D"], 3),
                         "Train/Val/Test": f'{i["n_train"]}/{i["n_val"]}/{i["n_test"]}',
                         "Min class": min(c for c in i["class_counts"] if c > 0),
                         "Note": i["note"]})
        self.save("T2_datasets", pd.DataFrame(rows), "Datasets and held-out splits (split seed 0).")

    # ------------------------------------------------------------------ main result tables
    def t_main(self):
        self.S["main"] = {}
        for metric in ["knn_acc", "linear_acc", "ncm_acc", "sil_native", "sil_2d", "trust", "db_native", "sil_native_train"]:
            rows, summ = [], {}
            for m in self.methods:
                row = {"Method": METHOD_LABEL[m]}
                means = []
                for d in self.datasets:
                    mu, h, n = ci95(self.vals(m, d, metric))
                    row[self.dname[d]] = fmt(mu, h)
                    summ.setdefault(m, {})[d] = {"mean": mu, "ci95": h, "n": n}
                    means.append(mu)
                row["Mean"] = f"{np.nanmean(means):.3f}"
                summ[m]["_mean"] = float(np.nanmean(means))
                rows.append(row)
            self.S["main"][metric] = summ
            # bold-free; winner marked with *
            df = pd.DataFrame(rows)
            for d in self.datasets:
                best = max(self.methods, key=lambda m: (summ[m][d]["mean"] if np.isfinite(summ[m][d]["mean"]) else -9)
                           * (-1 if metric == "db_native" else 1))
                df.loc[df.Method == METHOD_LABEL[best], self.dname[d]] += " *"
            self.save(f"T_main_{metric}", df,
                      f"{METRIC_LABEL[metric]} — mean ± 95% CI (t-interval, n={self.nseeds} held-out splits). * = best mean per dataset.")

    # ------------------------------------------------------------------ statistics
    def t_stats(self, metric=PRIMARY):
        # (a) per-dataset paired tests DACHP vs each method across the 5 splits, Holm within dataset
        rows = []
        per = {}
        for d in self.datasets:
            ps, tmp = [], []
            for m in self.methods:
                if m == self.focal:
                    continue
                a, b = self.main[self.focal][d], self.main[m][d]
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 2:
                    continue
                diff = np.array([a[s][metric] - b[s][metric] for s in seeds])
                if np.allclose(diff, diff[0]):
                    p = 1.0 if diff[0] == 0 else 0.0
                else:
                    p = stats.ttest_rel([a[s][metric] for s in seeds], [b[s][metric] for s in seeds]).pvalue
                lo, hi = boot_ci(diff)
                tmp.append([d, m, diff.mean(), lo, hi, p])
                ps.append(p)
            if ps:
                adj = holm(ps)
                for t, pa in zip(tmp, adj):
                    rows.append({"Dataset": self.dname[t[0]], "vs": METHOD_LABEL[t[1]], "Mean Δ": f"{t[2]:+.3f}",
                                 "95% boot CI": f"[{t[3]:+.3f}, {t[4]:+.3f}]", "p (paired t)": f"{t[5]:.4f}",
                                 "p Holm": f"{pa:.4f}", "Sig.": "yes" if pa < 0.05 else "no"})
                    per.setdefault(t[0], {})[t[1]] = {"delta": t[2], "ci": [t[3], t[4]], "p": t[5], "p_holm": pa}
        self.save(f"T_perdataset_tests_{metric}", pd.DataFrame(rows),
                  f"Per-dataset paired comparison of {METHOD_LABEL[self.focal]} vs each method on {METRIC_LABEL[metric]} over the {self.nseeds} "
                  "held-out splits; Holm correction across methods within each dataset.")
        self.S.setdefault("perdataset", {})[metric] = per

        # (b) across datasets: Friedman + Nemenyi, Wilcoxon + Holm, rank-biserial
        methods = [m for m in self.methods if all(np.isfinite(self.S["main"][metric][m][d]["mean"]) for d in self.datasets)]
        M = np.array([[self.S["main"][metric][m][d]["mean"] for m in methods] for d in self.datasets])
        sign = -1 if metric == "db_native" else 1
        ranks = np.array([stats.rankdata(-sign * row) for row in M])
        avg = ranks.mean(0)
        k, n = len(methods), len(self.datasets)
        fr = stats.friedmanchisquare(*M.T) if k >= 3 and n >= 2 else None
        q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) if k >= 2 else np.nan
        cd = q * np.sqrt(k * (k + 1) / (6 * n))
        rows, ps, tmp = [], [], []
        di = methods.index(self.focal)
        for j, m in enumerate(methods):
            if m == self.focal:
                continue
            diff = M[:, di] - M[:, j]
            try:
                w = stats.wilcoxon(M[:, di], M[:, j], alternative="two-sided", zero_method="zsplit")
                W, p = w.statistic, w.pvalue
            except ValueError:
                W, p = np.nan, 1.0
            lo, hi = boot_ci(diff)
            tmp.append((m, diff, W, p, lo, hi, j))
            ps.append(p)
        adj = holm(ps)
        agg = {}
        for (m, diff, W, p, lo, hi, j), pa in zip(tmp, adj):
            wins = int((diff > 0).sum()); losses = int((diff < 0).sum())
            rows.append({f"{METHOD_LABEL[self.focal]} vs": METHOD_LABEL[m], "Mean Δ": f"{diff.mean():+.3f}", "95% boot CI": f"[{lo:+.3f}, {hi:+.3f}]",
                         "W/T/L": f"{wins}/{n-wins-losses}/{losses}", "Wilcoxon W": f"{W:.1f}", "p": f"{p:.4f}",
                         "p Holm": f"{pa:.4f}", "Rank-biserial r": f"{rank_biserial(diff):+.2f}",
                         "Avg. rank gap": f"{avg[j]-avg[di]:+.2f}", "> CD?": "yes" if abs(avg[j] - avg[di]) > cd else "no"})
            agg[m] = {"delta": float(diff.mean()), "ci": [float(lo), float(hi)], "W": float(W), "p": float(p),
                      "p_holm": float(pa), "r_rb": float(rank_biserial(diff)), "wins": wins, "losses": losses}
        self.save(f"T_across_tests_{metric}", pd.DataFrame(rows),
                  f"Across-dataset comparison on {METRIC_LABEL[metric]} (one mean per dataset, n={n}). Friedman "
                  f"χ²={fr.statistic:.2f}, p={fr.pvalue:.2g}; Nemenyi CD(α=0.05)={cd:.2f}. Wilcoxon two-sided (focal: {METHOD_LABEL[self.focal]}) with Holm "
                  f"correction over {len(ps)} comparisons. NB: with n={n} the smallest attainable two-sided Wilcoxon p is "
                  f"{2/2**n:.4f}." if fr else "")
        rank_df = pd.DataFrame({"Method": [METHOD_LABEL[m] for m in methods], "Average rank": np.round(avg, 2)}).sort_values("Average rank")
        self.save(f"T_avg_rank_{metric}", rank_df, f"Average rank on {METRIC_LABEL[metric]} (1 = best).")
        self.S.setdefault("across", {})[metric] = {"friedman_chi2": float(fr.statistic) if fr else None,
                                                   "friedman_p": float(fr.pvalue) if fr else None, "cd": float(cd),
                                                   "avg_rank": dict(zip(methods, map(float, avg))), "vs": agg, "n": n}
        self.f_cd(methods, avg, cd, metric)

    # ------------------------------------------------------------------ figures: main
    def f_cd(self, methods, avg, cd, metric):
        order = np.argsort(avg)
        fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(methods) + 1.0))
        y = np.arange(len(methods))[::-1]
        for yy, i in zip(y, order):
            c = C_DACHP if methods[i] == self.focal else "#333333"
            ax.plot([avg[i]], [yy], "o", color=c, ms=6)
            ax.text(avg[i] + 0.15, yy, f"{METHOD_LABEL[methods[i]]}  ({avg[i]:.2f})", va="center", fontsize=8,
                    color="#222222", fontweight="bold" if methods[i] == self.focal else "normal")
        best = avg[order[0]]
        ax.axvspan(best, best + cd, color="#dddddd", zorder=0)
        ax.text(best + cd / 2, y[0] + 0.7, f"CD = {cd:.2f}", ha="center", fontsize=8)
        ax.set_yticks([]); ax.set_xlabel("Average rank across datasets (lower is better)")
        ax.set_xlim(0.5, len(methods) + 3.5); ax.set_ylim(-0.8, len(methods) + 0.3)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.set_title(f"Average ranks — {METRIC_LABEL[metric]} (Friedman/Nemenyi)", fontsize=9)
        self.savefig(fig, f"F_cd_{metric}")

    def f_bars(self, metric=PRIMARY):
        show = [m for m in ["dachp", "dachp_v2", "supcon", "arcface", "proxyanchor", "ce_mlp", "sup_umap_tuned", "umap", "pca"] if m in self.methods]
        n = len(self.datasets)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.4 * rows), squeeze=False)
        for ax, d in zip(axs.flat, self.datasets):
            mus = [ci95(self.vals(m, d, metric)) for m in show]
            ax.bar(range(len(show)), [x[0] for x in mus], yerr=[x[1] if np.isfinite(x[1]) else 0 for x in mus],
                   color=[C_DACHP if m == self.focal else ("#E69F00" if m.startswith("dachp") else C_GREY) for m in show], width=0.7, capsize=2,
                   error_kw={"lw": 0.8})
            lo = min(x[0] for x in mus if np.isfinite(x[0]))
            ax.set_ylim(max(-1.0, lo - 0.08) if metric.startswith("sil") else max(0, lo - 0.05), 1.0 + 0.01)
            ax.set_xticks(range(len(show)))
            ax.set_xticklabels([METHOD_LABEL[m].replace(" (ours)", "").replace(" (same enc.)", "") for m in show],
                               rotation=60, ha="right", fontsize=6)
            i = self.diag[d]["info"]
            ax.set_title(f'{i["name"]} (D={i["D"]})', fontsize=8)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(axis="y", alpha=0.3, lw=0.5)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
        for ax in list(axs.flat)[n:]:
            ax.set_visible(False)
        fig.supylabel(METRIC_LABEL[metric], fontsize=8)
        fig.tight_layout()
        self.savefig(fig, f"F_bars_{metric}")

    def f_embeddings(self, emb_dir):
        show = [m for m in ["dachp", "supcon", "sup_umap_tuned", "umap", "tsne", "pca"]
                if glob.glob(os.path.join(emb_dir, f"*__{m}.npz"))]
        if not show:
            return
        fig, axs = plt.subplots(len(self.datasets), len(show), figsize=(1.9 * len(show), 1.7 * len(self.datasets)), squeeze=False)
        cmap = plt.get_cmap("tab20")
        for r, d in enumerate(self.datasets):
            for c, m in enumerate(show):
                ax = axs[r, c]
                f = os.path.join(emb_dir, f"{d}__{m}.npz")
                ax.set_xticks([]); ax.set_yticks([])
                if os.path.exists(f):
                    z = np.load(f)
                    ax.scatter(z["Z"][:, 0], z["Z"][:, 1], c=z["y"] % 20, cmap=cmap, s=2, lw=0, alpha=0.8, rasterized=True)
                    sil = self.S["main"]["sil_2d"].get(m, {}).get(d, {}).get("mean", np.nan)
                    knn = self.S["main"]["knn_acc"].get(m, {}).get(d, {}).get("mean", np.nan)
                    ax.set_title(f"kNN {knn:.3f} | Sil2D {sil:.2f}", fontsize=5.5)
                if r == 0:
                    ax.text(0.5, 1.28, METHOD_LABEL[m], transform=ax.transAxes, ha="center", fontsize=7, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(self.dname[d], fontsize=6.5)
        fig.tight_layout(h_pad=0.4, w_pad=0.2)
        self.savefig(fig, "F_embeddings_test")

    # ------------------------------------------------------------------ ablation
    def ablation(self, metric=PRIMARY):
        ab = defaultdict(lambda: defaultdict(dict))
        for r in self.recs:
            for t in r.get("tags", []):
                if t.startswith("ablation:") and r["eval_on"] == "test":
                    ab[t.split(":", 1)[1]][r["dataset"]][r["seed"]] = r
        if "full" not in ab:
            return
        variants = [v for v in ABL_LABEL if v in ab]
        out = {}
        for met in [metric, "sil_native", "linear_acc"]:
            rows = []
            for v in variants:
                row = {"Variant": ABL_LABEL[v]}
                for d in self.datasets:
                    x = [r["metrics"][met] for r in ab[v][d].values()]
                    mu, h, _ = ci95(x)
                    cell = fmt(mu, h)
                    if v != "full":
                        seeds = sorted(set(ab[v][d]) & set(ab["full"][d]))
                        if len(seeds) >= 2:
                            a = [ab["full"][d][s]["metrics"][met] for s in seeds]
                            b = [ab[v][d][s]["metrics"][met] for s in seeds]
                            dd = np.subtract(a, b)
                            p = 1.0 if np.allclose(dd, dd[0]) and dd[0] == 0 else (0.0 if np.allclose(dd, dd[0]) else stats.ttest_rel(a, b).pvalue)
                            cell += "†" if p < 0.05 / (len(variants) - 1) else ""
                            out.setdefault(met, {}).setdefault(v, {})[d] = {"mean": mu, "delta_vs_full": float(np.mean(b) - np.mean(a)), "p": float(p)}
                    else:
                        out.setdefault(met, {}).setdefault(v, {})[d] = {"mean": mu}
                    row[self.dname[d]] = cell
                rows.append(row)
            self.save(f"T_ablation_{met}", pd.DataFrame(rows),
                      f"Ablation — {METRIC_LABEL[met]}, mean ± 95% CI over {self.nseeds} splits. † = differs from Full DACHP, "
                      f"paired t-test, Bonferroni-adjusted α=0.05/{len(variants)-1}.")
        self.S["ablation"] = out
        for hm_met in [metric, "sil_native"]:
            # heatmap of delta vs full
            M = np.array([[out[hm_met][v][d].get("delta_vs_full", 0.0) for d in self.datasets] for v in variants[1:]])
            fig, ax = plt.subplots(figsize=(0.75 * len(self.datasets) + 2.5, 0.42 * len(variants) + 1))
            lim = max(0.05, np.nanmax(np.abs(M)))
            im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(M[i, j]) > 0.6 * lim else "black")
            ax.set_xticks(range(len(self.datasets))); ax.set_xticklabels([self.dname[d] for d in self.datasets], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(variants) - 1)); ax.set_yticklabels([ABL_LABEL[v] for v in variants[1:]], fontsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label(f"Δ {METRIC_LABEL[hm_met]} vs Full (neg. = component helps)", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            self.savefig(fig, f"F_ablation_heatmap_{hm_met}")
        # collapse dynamics: full vs no_sphere
        self.f_dynamics(ab)

    def f_dynamics(self, ab):
        stats_keys = [("resultant", "Resultant length ‖mean ẑ‖ (1 = collapse)"), ("cos_var", "Var. of pairwise cosine"),
                      ("eff_rank", "Effective rank of ẑ"), ("cos_inter", "Mean inter-class cosine")]
        fig, axs = plt.subplots(1, 4, figsize=(12, 2.8))
        summ = {}
        for v, col in [("full", C_DACHP), ("no_sphere", "#0072B2"), ("legacy_tau", "#009E73")]:
            if v not in ab:
                continue
            for d in self.datasets:
                runs = [r for r in ab[v][d].values() if r["info"].get("monitor")]
                if not runs:
                    continue
                for ax, (k, lab) in zip(axs, stats_keys):
                    ep = [m["epoch"] for m in runs[0]["info"]["monitor"]]
                    y = np.mean([[m[k] for m in r["info"]["monitor"]] for r in runs], 0)
                    ax.plot(ep, y, color=col, lw=0.9, alpha=0.7, label=ABL_LABEL[v] if d == self.datasets[0] else None)
                    ax.set_title(lab, fontsize=8); ax.set_xlabel("epoch", fontsize=7); ax.tick_params(labelsize=6)
                last = {k: float(np.mean([r["info"]["monitor"][-1][k] for r in runs])) for k, _ in stats_keys}
                summ.setdefault(v, {})[d] = last
        axs[0].legend(fontsize=6)
        for ax in axs:
            ax.grid(alpha=0.3, lw=0.5)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        self.savefig(fig, "F_collapse_dynamics")
        self.S["dynamics_final"] = summ
        rows = []
        for d in self.datasets:
            row = {"Dataset": self.dname[d]}
            for v in ["full", "no_sphere"]:
                if v in summ and d in summ[v]:
                    s = summ[v][d]
                    row[f"{v}: resultant"] = f'{s["resultant"]:.3f}'
                    row[f"{v}: cos var"] = f'{s["cos_var"]:.3f}'
                    row[f"{v}: eff. rank"] = f'{s["eff_rank"]:.2f}'
            rows.append(row)
        self.save("T_collapse_final", pd.DataFrame(rows),
                  f"Angular-collapse diagnostics at the last epoch (training monitor subset, mean over {self.nseeds} runs). "
                  "Resultant length 1.0 = all embeddings share one direction; effective rank ≤ 16.")

    # ------------------------------------------------------------------ sensitivity
    def sensitivity(self, metric=PRIMARY):
        se = defaultdict(lambda: defaultdict(list))
        for r in self.recs:
            for t in r.get("tags", []):
                if t.startswith("sens:") and r["eval_on"] == "test":
                    se[t[5:]][r["dataset"]].append(r["metrics"])
        if not se:
            return
        rows, out = [], {}
        for key in sorted(se, key=lambda s: (s.split("=")[0], float(s.split("=")[1]))):
            p, v = key.split("=")
            row = {"Parameter": p, "Value": v}
            for d in self.datasets:
                mu, h, _ = ci95([m[metric] for m in se[key][d]])
                mu2, _, _ = ci95([m["sil_native"] for m in se[key][d]])
                row[self.dname[d]] = f"{mu:.3f} / {mu2:.3f}" if np.isfinite(mu) else "–"
                out.setdefault(p, {}).setdefault(v, {})[d] = {"knn": mu, "sil": mu2}
            rows.append(row)
        self.save("T3_sensitivity", pd.DataFrame(rows),
                  f"One-at-a-time sensitivity, all datasets, 3 splits. Cell = test 5-NN accuracy / test native Silhouette.")
        self.S["sensitivity"] = out
        params = list(out)
        fig, axs = plt.subplots(1, len(params), figsize=(3.0 * len(params), 2.6), squeeze=False)
        for ax, p in zip(axs[0], params):
            xs = sorted(out[p], key=float)
            for i, d in enumerate(self.datasets):
                ax.plot([float(x) for x in xs], [out[p][x][d]["sil"] for x in xs], "-o", ms=2.5, lw=0.9,
                        color=plt.get_cmap("tab20")(i % 20), label=self.dname[d])
            ax.set_xlabel(p, fontsize=8); ax.set_ylabel("Test Silhouette (native)", fontsize=7); ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3, lw=0.5)
        axs[0][-1].legend(fontsize=5, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        self.savefig(fig, "F_sensitivity")

    # ------------------------------------------------------------------ permutation
    def permutation(self):
        rows, out = [], {}
        for d in self.datasets:
            x = [r["metrics"] for r in self.recs if r.get("perm") and r["dataset"] == d]
            if not x:
                continue
            counts = np.array(self.diag[d]["info"]["class_counts"], float)
            chance = counts.max() / counts.sum()
            real = self.S["main"]["knn_acc"]["dachp"][d]["mean"]
            mu, h, _ = ci95([m["knn_acc"] for m in x])
            rows.append({"Dataset": self.dname[d], "True labels": f"{real:.3f}", "Permuted train labels": fmt(mu, h),
                         "Majority-class rate": f"{chance:.3f}", "Uniform chance": f"{1/len(counts[counts>0]):.3f}"})
            out[d] = {"real": real, "perm": mu, "majority": chance}
        if rows:
            self.save("T_permutation", pd.DataFrame(rows),
                      "Label-permutation control: DACHP trained on randomly permuted TRAINING labels; the 5-NN probe is then "
                      "fitted on the true training labels and scored on the true test labels (5 permutations). Accuracy "
                      "that stays high indicates class structure already present in the input and preserved by the "
                      "encoder; accuracy at chance indicates that the structure was learned from the labels.")
            self.S["permutation"] = out

    # ------------------------------------------------------------------ cost
    def cost(self):
        rows, out = [], {}
        for d in self.datasets:
            row = {"Dataset": self.dname[d]}
            for m in ["dachp", "supcon", "sup_umap_tuned", "umap", "tsne"]:
                recs = list(self.main_info[m][d].values())
                if not recs:
                    continue
                t = np.mean([r["info"]["train_seconds"] for r in recs])
                row[f"{METHOD_LABEL[m]} time (s)"] = f"{t:.1f}"
                out.setdefault(d, {})[m] = {"seconds": float(t)}
                if m == "dachp":
                    mems = [_peak_mem(r) for r in recs]
                    mem = np.nanmean(mems) if any(np.isfinite(x) for x in mems) else np.nan
                    row["DACHP peak memory (MB)"] = f"{mem:.0f}" if np.isfinite(mem) else "n/a"
                    row["DACHP params"] = recs[0]["info"].get("n_params")
                    row["DACHP H1/H2"] = "/".join(map(str, recs[0]["info"].get("H", [])))
                    row["Batch"] = recs[0]["info"].get("batch_size")
                    out[d][m].update(peak_mb=float(mem), device=recs[0]["info"].get("device"))
            rows.append(row)
        self.save("T_cost", pd.DataFrame(rows),
                  f"Wall-clock training/fitting time (mean over {self.nseeds} runs) and DACHP peak memory where it was recorded. "
                  "Times depend on the hardware used.")
        self.S["cost"] = out

    # ------------------------------------------------------------------ diagnostics vs gain
    def diagnostics(self):
        rows, keys = [], ["knn_purity", "false_nn_rate", "hub_skew", "dist_cv", "rel_contrast", "n_over_d"]
        gain = {}
        for d in self.datasets:
            g = self.diag[d]["metrics"]
            try:
                gs = self.S["main"]["sil_native"]["dachp"][d]["mean"] - self.S["main"]["sil_native"]["sup_umap_tuned"][d]["mean"]
                gk = self.S["main"]["knn_acc"]["dachp"][d]["mean"] - self.S["main"]["knn_acc"]["sup_umap_tuned"][d]["mean"]
            except KeyError:
                gs = gk = np.nan
            gain[d] = (gk, gs)
            rows.append({"Dataset": self.dname[d], **{k: f"{g[k]:.3f}" for k in keys},
                         "Δ kNN (DACHP−SupUMAP)": f"{gk:+.3f}", "Δ Sil (DACHP−SupUMAP)": f"{gs:+.3f}"})
        self.save("T_input_diagnostics", pd.DataFrame(rows),
                  "Input-space neighbourhood diagnostics (standardised training split): 10-NN label purity, 1-NN "
                  "false-neighbour rate, hubness skewness of 10-occurrence, coefficient of variation of pairwise distances, "
                  "median relative contrast (Dmax−Dmin)/Dmin, and N/D; with DACHP's gain over tuned Supervised UMAP.")
        corr = []
        out = {}
        for k in keys:
            x = [self.diag[d]["metrics"][k] for d in self.datasets]
            for gi, gname in [(0, "Δ kNN"), (1, "Δ Sil")]:
                y = [gain[d][gi] for d in self.datasets]
                ok = np.isfinite(y)
                rho, p = stats.spearmanr(np.array(x)[ok], np.array(y)[ok]) if ok.sum() > 3 else (np.nan, np.nan)
                corr.append({"Diagnostic": k, "Gain": gname, "Spearman ρ": f"{rho:+.2f}", "p": f"{p:.3f}"})
                out[f"{k}|{gname}"] = {"rho": float(rho), "p": float(p)}
        self.save("T_diagnostic_correlation", pd.DataFrame(corr),
                  f"Spearman correlation between input-space diagnostics and DACHP's gain over tuned Sup-UMAP (n={len(self.datasets)} datasets; exploratory, uncorrected).")
        self.S["diagnostic_corr"] = out
        fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))
        for ax, k in zip(axs, ["knn_purity", "hub_skew", "rel_contrast"]):
            x = [self.diag[d]["metrics"][k] for d in self.datasets]
            y = [gain[d][1] for d in self.datasets]
            ax.scatter(x, y, color=C_DACHP, s=14)
            for xi, yi, d in zip(x, y, self.datasets):
                ax.annotate(self.dname[d], (xi, yi), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
            ax.axhline(0, color="#555555", lw=0.6)
            ax.set_xlabel(k, fontsize=8); ax.set_ylabel("Δ test Silhouette vs tuned Sup-UMAP", fontsize=7)
            ax.tick_params(labelsize=6); ax.grid(alpha=0.3, lw=0.5)
            if k == "rel_contrast":
                ax.set_xscale("log")
        fig.tight_layout()
        self.savefig(fig, "F_gain_vs_diagnostics")

    # ------------------------------------------------------------------ convergence
    def convergence(self):
        fig_n = len(self.datasets)
        cols = 4
        rows_ = int(np.ceil(fig_n / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.3 * rows_), squeeze=False)
        taus = {}
        for ax, d in zip(axs.flat, self.datasets):
            runs = [r for r in self.main_info["dachp"][d].values() if r["info"].get("history")]
            if not runs:
                continue
            ep = [h["epoch"] for h in runs[0]["info"]["history"]]
            for k, col, lab in [("total", "#000000", "Total"), ("main", C_DACHP, "Hinge (L_pos+L_neg)"), ("aux", "#0072B2", "Aux. SupCon")]:
                if k in runs[0]["info"]["history"][0]:
                    y = np.array([[h[k] for h in r["info"]["history"]] for r in runs])
                    ax.plot(ep, y.mean(0), color=col, lw=1, label=lab)
                    ax.fill_between(ep, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.15, lw=0)
            if "tau" in runs[0]["info"]["history"][0]:
                t = np.array([[h["tau"] for h in r["info"]["history"]] for r in runs]).mean(0)
                taus[d] = float(t[-1])
                ax.text(0.98, 0.95, f"final τ = {t[-1]:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
            ax.set_title(self.dname[d], fontsize=8); ax.tick_params(labelsize=6); ax.grid(alpha=0.3, lw=0.5)
        for ax in list(axs.flat)[fig_n:]:
            ax.set_visible(False)
        axs[0][0].legend(fontsize=6)
        fig.supxlabel("epoch", fontsize=8); fig.supylabel(f"loss (mean ± 1 s.d., {self.nseeds} runs)", fontsize=8)
        fig.tight_layout()
        self.savefig(fig, "F_convergence")
        self.S["final_tau"] = taus

    # ------------------------------------------------------------------ graphical abstract (real numbers only)
    def graphical_abstract(self, metric=PRIMARY):
        show = [m for m in ["pca", "umap", "sup_umap_tuned", "supcon", "arcface", "dachp"] if m in self.methods]
        vals = [self.S["main"][metric][m]["_mean"] for m in show]
        sil = [self.S["main"]["sil_native"][m]["_mean"] for m in show]
        fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
        for ax, v, lab in [(axs[0], vals, METRIC_LABEL[metric]), (axs[1], sil, METRIC_LABEL["sil_native"])]:
            ax.barh(range(len(show)), v, color=[C_DACHP if m == "dachp" else C_GREY for m in show], height=0.6)
            for i, x in enumerate(v):
                ax.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=7)
            ax.set_yticks(range(len(show))); ax.set_yticklabels([METHOD_LABEL[m] for m in show], fontsize=7)
            ax.set_xlabel(f"{lab}\n(mean over {len(self.datasets)} datasets, held-out test sets)", fontsize=7)
            ax.set_xlim(min(0, min(v) - 0.05), 1.1); ax.tick_params(labelsize=6)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        self.savefig(fig, "F_graphical_abstract_results_panel")

    def dimensionality(self, metric="sil_native"):
        show = [m for m in ["dachp", "supcon", "sup_umap_tuned", "umap", "pca"] if m in self.methods]
        D = [self.diag[d]["info"]["D"] for d in self.datasets]
        o = np.argsort(D)
        fig, ax = plt.subplots(figsize=(6.5, 3))
        for i, m in enumerate(show):
            y = [self.S["main"][metric][m][self.datasets[j]]["mean"] for j in o]
            ax.plot(np.array(D)[o], y, "-o", ms=3.5, lw=1.4 if m == "dachp" else 0.9,
                    color=C_DACHP if m == "dachp" else PAL[1 + i % 7], label=METHOD_LABEL[m])
        ax.set_xscale("log"); ax.set_xlabel("Input dimensionality D (log)", fontsize=8)
        ax.set_ylabel(METRIC_LABEL[metric], fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=6, ncol=2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        self.savefig(fig, f"F_dimensionality_{metric}")

    def run(self, emb_dirs):
        self.t_datasets()
        self.t_main()
        for met in ["knn_acc", "linear_acc", "sil_native", "sil_2d"]:
            self.t_stats(met)
        self.f_bars("knn_acc"); self.f_bars("sil_native")
        for e in emb_dirs:
            if os.path.isdir(e):
                self.f_embeddings(e)
        self.ablation(); self.sensitivity(); self.permutation(); self.cost(); self.diagnostics()
        self.convergence(); self.graphical_abstract(); self.dimensionality("sil_native"); self.dimensionality("knn_acc")
        self.S["focal"] = self.focal
        self.S["datasets"] = {d: self.diag[d]["info"] for d in self.datasets}
        json.dump(self.S, open(os.path.join(self.out, "summary.json"), "w"), indent=1,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        open(os.path.join(self.out, "REPORT.md"), "w").write("\n".join(self.md))
        print(f"report written to {self.out}: {len(os.listdir(self.tdir))} table files, {len(os.listdir(self.fdir))} figure files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results"])
    ap.add_argument("--out", default="report")
    ap.add_argument("--focal", default="dachp")
    a = ap.parse_args()
    recs = load(a.results)
    print(f"{len(recs)} records loaded")
    embs = [os.path.join(p, "embeddings") for p in a.results]
    # merge embedding folders into one for plotting
    merged = os.path.join(a.out, "_emb")
    os.makedirs(merged, exist_ok=True)
    for e in embs:
        for f in glob.glob(os.path.join(e, "*.npz")):
            dst = os.path.join(merged, os.path.basename(f))
            if not os.path.exists(dst):
                os.link(f, dst) if hasattr(os, "link") else None
    Report(recs, a.out, a.focal).run([merged])


if __name__ == "__main__":
    main()
