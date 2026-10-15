"""Classical / graph-based baselines under the same held-out protocol.

* PCA, LDA, UMAP and Supervised UMAP have an out-of-sample transform: fitted on the
  training split only (labels only from the training split), then applied to test.
* t-SNE, PaCMAP and TriMap have no out-of-sample extension: they are fitted
  (without labels) on train+test jointly; the probes are still trained on the
  training part only. This is stated in the manuscript.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


def _pre_pca(Xtr, Xother, n, seed):
    if n in (None, "none") or Xtr.shape[1] <= n:
        return Xtr, Xother
    p = PCA(n_components=min(n, Xtr.shape[0] - 1), random_state=seed).fit(Xtr)
    return p.transform(Xtr), [p.transform(x) for x in Xother]


def run_classical(method, Xtr, ytr, Xte, seed=0, params=None):
    params = dict(params or {})
    t0 = time.perf_counter()
    if method == "pca":
        m = PCA(n_components=min(params.get("n_components", 16), *Xtr.shape), random_state=seed).fit(Xtr)
        Ztr, Zte = m.transform(Xtr), m.transform(Xte)
    elif method == "lda":
        C = len(np.unique(ytr))
        m = LinearDiscriminantAnalysis(n_components=min(C - 1, 16, Xtr.shape[1])).fit(Xtr, ytr)
        Ztr, Zte = m.transform(Xtr), m.transform(Xte)
    elif method in ("tsne", "pacmap", "trimap"):
        Xall = np.vstack([Xtr, Xte])
        n = len(Xall)
        if method == "tsne":
            perp = min(30, max(5, n // 10))
            Z = TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(Xall)
        elif method == "pacmap":
            import pacmap
            Z = pacmap.PaCMAP(n_components=2, n_neighbors=params.get("n_neighbors", 10),
                              random_state=seed).fit_transform(Xall, init="pca")
        else:
            import trimap
            Z = trimap.TRIMAP(n_dims=2).fit_transform(Xall)
        Ztr, Zte = Z[: len(Xtr)], Z[len(Xtr):]
    elif method in ("umap", "sup_umap"):
        import umap
        A, (B,) = _pre_pca(Xtr, [Xte], params.get("pre_pca", 50), seed)
        m = umap.UMAP(n_components=params.get("n_components", 2), n_neighbors=params.get("n_neighbors", 15),
                      min_dist=params.get("min_dist", 0.1), target_weight=params.get("target_weight", 0.5),
                      random_state=seed)
        if method == "sup_umap":
            m.fit(A, y=ytr)
        else:
            m.fit(A)
        Ztr, Zte = m.embedding_, m.transform(B)
    else:
        raise KeyError(method)
    return np.asarray(Ztr, np.float32), np.asarray(Zte, np.float32), {"train_seconds": time.perf_counter() - t0}


def available(method):
    if method == "trimap":
        try:
            import trimap  # noqa
            return True
        except Exception:
            return False
    if method == "pacmap":
        try:
            import pacmap  # noqa
            return True
        except Exception:
            return False
    return True
