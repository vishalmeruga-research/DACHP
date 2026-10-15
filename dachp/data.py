"""Dataset loaders and leakage-free train/val/test splitting.

Every preprocessing statistic (imputation medians, StandardScaler mean/variance,
any PCA used by a method) is fitted on the TRAINING split only (Reviewer #4, 7-9).
"""
from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, fetch_olivetti_faces
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_HOME = os.environ.get("DACHP_DATA", os.path.join(os.path.dirname(__file__), "..", "data_cache"))

META = {
    # key: (display name, domain, regime-by-D)
    "breast_cancer": ("Breast Cancer", "Medical (cytology features)"),
    "digits": ("Digits", "Handwritten image"),
    "mice_protein": ("Mice Protein", "Proteomic expression"),
    "anneal": ("Annealing", "Industrial process (steel annealing)"),
    "coil20": ("COIL-20", "Object image (rotation)"),
    "cnae9": ("CNAE-9", "Text (sparse term frequencies)"),
    "mnist": ("MNIST", "Handwritten image"),
    "fashion_mnist": ("Fashion-MNIST", "Natural image"),
    "har": ("HAR", "Inertial sensor signal (time+frequency features)"),
    "isolet": ("ISOLET", "Speech signal (spectral features)"),
    "olivetti": ("Olivetti Faces", "Face image"),
}


def regime(D: int) -> str:
    if D <= 30:
        return "Low"
    if D <= 100:
        return "Moderate"
    if D < 1000:
        return "High"
    return "Ultra-High"


@dataclass
class Dataset:
    key: str
    name: str
    domain: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None
    note: str = ""

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def D(self):
        return self.X.shape[1]

    @property
    def C(self):
        return len(np.unique(self.y))


def _openml(data_id: int):
    return fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=DATA_HOME)


def _frame_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode categorical columns (missing -> own category); numeric NaN kept for the imputer."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        if col.dtype.name in ("category", "object", "bool"):
            codes = col.astype("object").where(col.notna(), "__missing__").astype(str)
            out[c] = LabelEncoder().fit_transform(codes).astype(np.float32)
        else:
            out[c] = pd.to_numeric(col, errors="coerce").astype(np.float32)
    return out.values.astype(np.float32)


def _stratified_subset(y, per_class, seed=42):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def _load_coil20() -> np.ndarray:
    """True COIL-20 (processed): 20 objects x 72 poses, 128x128 -> 32x32 grey (D=1024).

    Looks for a local copy first (any of: data_cache/coil20.npz, coil-20-proc.zip,
    coil-20-proc/ folder), otherwise downloads the official archive from Columbia CAVE.
    """
    from PIL import Image

    os.makedirs(DATA_HOME, exist_ok=True)
    npz = os.path.join(DATA_HOME, "coil20.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["X"], d["y"]
    zpath = os.path.join(DATA_HOME, "coil-20-proc.zip")
    folder = os.path.join(DATA_HOME, "coil-20-proc")
    files = {}
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".png"):
                files[f] = open(os.path.join(folder, f), "rb").read()
    else:
        if not os.path.exists(zpath):
            urls = [
                "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
                "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
            ]
            last = None
            for u in urls:
                try:
                    urllib.request.urlretrieve(u, zpath)
                    break
                except Exception as e:  # pragma: no cover
                    last = e
            if not os.path.exists(zpath):
                raise RuntimeError(
                    f"Could not download COIL-20 ({last}). Download coil-20-proc.zip from "
                    "https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php and place it in " + DATA_HOME
                )
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if n.endswith(".png"):
                    files[os.path.basename(n)] = z.read(n)
    X, y = [], []
    for fname in sorted(files):
        obj = int(fname.split("__")[0].replace("obj", ""))
        img = Image.open(io.BytesIO(files[fname])).convert("L").resize((32, 32), Image.BILINEAR)
        X.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
        y.append(obj - 1)
    X, y = np.stack(X), np.array(y)
    assert X.shape == (1440, 1024) and len(np.unique(y)) == 20, X.shape
    np.savez_compressed(npz, X=X, y=y)
    return X, y


def load_dataset(key: str, cfg: dict | None = None) -> Dataset:
    cfg = cfg or {}
    name, domain = META.get(key, (key, "synthetic"))
    groups, note = None, ""
    if key == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data.astype(np.float32), d.target
    elif key == "digits":
        d = load_digits()
        X, y = d.data.astype(np.float32), d.target
        note = "full 1,797 samples (the original submission used a 1,000-sample subset)"
    elif key == "mice_protein":
        d = _openml(40966)
        X = _frame_to_numeric(d.data)
        y = LabelEncoder().fit_transform(d.target.astype(str))
        # OpenML stores the 1,080 rows as 72 mice x 15 technical replicates, in order.
        g = np.arange(len(y)) // 15
        if all(len(np.unique(y[g == k])) == 1 for k in np.unique(g)) and len(np.unique(g)) == 72:
            groups = g
            note = "split grouped by mouse (replicates never cross train/test)"
        else:
            note = "WARNING: mouse grouping could not be verified; random stratified split used"
    elif key == "anneal":
        d = _openml(2)
        X = _frame_to_numeric(d.data)
        y = LabelEncoder().fit_transform(d.target.astype(str))
        note = "UCI Annealing (OpenML id 2); categorical attributes ordinal-encoded"
    elif key == "coil20":
        X, y = _load_coil20()
    elif key == "cnae9":
        d = _openml(1468)
        X = _frame_to_numeric(d.data)
        y = LabelEncoder().fit_transform(d.target.astype(str))
    elif key in ("mnist", "fashion_mnist"):
        d = _openml(554 if key == "mnist" else 40996)
        X_full = d.data.values.astype(np.float32) / 255.0
        y_full = LabelEncoder().fit_transform(d.target.astype(str))
        idx = _stratified_subset(y_full, cfg.get("subset_per_class", 1000), seed=42)
        X, y = X_full[idx], y_full[idx]
        note = f"stratified subset, {cfg.get('subset_per_class', 1000)} per class"
    elif key == "har":
        d = _openml(1478)
        X = _frame_to_numeric(d.data)
        y = LabelEncoder().fit_transform(d.target.astype(str))
        note = "subject identifiers not available in the OpenML release; random stratified split"
    elif key == "isolet":
        d = _openml(300)
        X = _frame_to_numeric(d.data)
        y = LabelEncoder().fit_transform(d.target.astype(str))
        note = "speaker identifiers not available in the OpenML release; random stratified split"
    elif key == "olivetti":
        d = fetch_olivetti_faces(shuffle=False, data_home=DATA_HOME)
        X, y = d.data.astype(np.float32), d.target
    elif key.startswith("synthetic"):  # used only by the CPU smoke test
        from sklearn.datasets import make_classification
        D = int(key.split("_")[1]) if "_" in key else 50
        X, y = make_classification(n_samples=int(os.environ.get("SYN_N", 600)), n_features=D, n_informative=min(10, D), n_classes=4,
                                   n_clusters_per_class=2, random_state=0)
        X = X.astype(np.float32)
        name, domain = f"Synthetic-{D}", "synthetic"
    else:
        raise KeyError(key)
    return Dataset(key, name, domain, X, np.asarray(y), groups, note)


def make_split(ds: Dataset, seed: int, fr=(0.6, 0.2, 0.2)):
    """Stratified (and, when available, group-aware) train/val/test indices."""
    N = ds.N
    idx = np.arange(N)
    if ds.groups is not None:
        # 5-fold stratified-group: fold (seed %5) = test, next fold = val, rest = train
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [te for _, te in sgk.split(ds.X, ds.y, ds.groups)]
        te = folds[0]
        va = folds[1]
        tr = np.setdiff1d(idx, np.concatenate([te, va]))
        return tr, va, te
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=fr[2], random_state=seed)
    trva, te = next(s1.split(ds.X, ds.y))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=fr[1] / (fr[0] + fr[1]), random_state=seed)
    tr_i, va_i = next(s2.split(ds.X[trva], ds.y[trva]))
    return trva[tr_i], trva[va_i], te


class Preprocessor:
    """Median imputation + standardisation, fitted on the training split only."""

    def fit(self, X):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imp.transform(X)
        self.sc = StandardScaler().fit(Xi)
        return self

    def transform(self, X):
        Z = self.sc.transform(self.imp.transform(X)).astype(np.float32)
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def prepared_split(ds: Dataset, seed: int, fr=(0.6, 0.2, 0.2)):
    tr, va, te = make_split(ds, seed, fr)
    pp = Preprocessor().fit(ds.X[tr])
    return dict(
        Xtr=pp.transform(ds.X[tr]), ytr=ds.y[tr],
        Xva=pp.transform(ds.X[va]), yva=ds.y[va],
        Xte=pp.transform(ds.X[te]), yte=ds.y[te],
        idx=(tr, va, te),
    )
