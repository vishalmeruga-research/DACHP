"""Held-out evaluation metrics and input-space diagnostics.

Primary metrics (Reviewer #4, 11-12): test-set k-NN accuracy, linear-probe accuracy and
nearest-class-mean accuracy — classifiers are fitted on TRAIN embeddings and scored on
TEST embeddings of unseen samples. Cluster indices (Silhouette, CH, DB) are reported on
the TEST embeddings in their NATIVE dimensionality and, for comparability with the
original submission, after a 2-D PCA fitted on the training embeddings (Reviewer #2, 8).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _sub(n, m, seed=0):
    if n <= m:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, m, replace=False)


def safe(f, *a, **k):
    try:
        return float(f(*a, **k))
    except Exception:
        return float("nan")


def evaluate(Ztr, ytr, Zte, yte, Xte=None, k=5, trust_k=10, trust_max_n=2000, sil_max_n=5000,
             two_d_native=False):
    out = {}
    knn = KNeighborsClassifier(n_neighbors=min(k, len(ytr) - 1)).fit(Ztr, ytr)
    pred = knn.predict(Zte)
    out["knn_acc"] = accuracy_score(yte, pred)
    out["knn_bacc"] = balanced_accuracy_score(yte, pred)
    sc = StandardScaler().fit(Ztr)
    lp = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Ztr), ytr)
    out["linear_acc"] = accuracy_score(yte, lp.predict(sc.transform(Zte)))
    ncm = NearestCentroid().fit(Ztr, ytr)
    out["ncm_acc"] = accuracy_score(yte, ncm.predict(Zte))

    s = _sub(len(yte), sil_max_n)
    out["sil_native"] = safe(silhouette_score, Zte[s], yte[s])
    out["ch_native"] = safe(calinski_harabasz_score, Zte, yte)
    out["db_native"] = safe(davies_bouldin_score, Zte, yte)
    if Zte.shape[1] > 2 and not two_d_native:
        p = PCA(2, random_state=0).fit(Ztr)
        Z2 = p.transform(Zte)
    else:
        Z2 = Zte[:, :2]
    out["sil_2d"] = safe(silhouette_score, Z2[s], yte[s])
    if Xte is not None:
        t = _sub(len(yte), trust_max_n)
        out["trust"] = safe(trustworthiness, Xte[t], Zte[t], n_neighbors=trust_k)
    # training-set Silhouette (transductive; reported only to expose the train/test gap)
    s2 = _sub(len(ytr), sil_max_n)
    out["sil_native_train"] = safe(silhouette_score, Ztr[s2], ytr[s2])
    return out


def input_diagnostics(X, y, k=10, max_n=3000, seed=0):
    """Neighbourhood-quality measures of the (standardised) INPUT space (Reviewer #4, 2)."""
    s = _sub(len(y), max_n, seed)
    X, y = X[s], y[s]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    ind, dist = ind[:, 1:], dist[:, 1:]
    purity = (y[ind] == y[:, None]).mean()
    counts = np.bincount(ind.ravel(), minlength=len(y))
    hub_skew = skew(counts)
    # distance concentration: relative contrast and coefficient of variation of pairwise distances
    r = np.random.RandomState(seed)
    a, b = r.randint(0, len(y), 20000), r.randint(0, len(y), 20000)
    m = a != b
    d = np.linalg.norm(X[a[m]] - X[b[m]], axis=1)
    cv = d.std() / d.mean()
    from sklearn.metrics import pairwise_distances
    q = r.choice(len(y), min(300, len(y)), replace=False)
    d_all = np.sort(pairwise_distances(X[q], X), axis=1)
    dmax, dmin = d_all[:, -1], np.maximum(d_all[:, 1], 1e-12)
    rel_contrast = np.median((dmax - dmin) / dmin)
    # fraction of samples whose single nearest neighbour has a different label
    false_nn = (y[ind[:, 0]] != y).mean()
    return dict(knn_purity=float(purity), hub_skew=float(hub_skew), dist_cv=float(cv),
                rel_contrast=float(rel_contrast), false_nn_rate=float(false_nn),
                n_over_d=float(X.shape[0] / X.shape[1]))
