"""dimension sweep and regularized readouts"""
import numpy as np
from numpy.linalg import svd
from .progress import bar


def pca_geometry(H, k):
    """project the geometry onto its top k principal directions; returns (k, V) matrix"""
    Hc = H - H.mean(1, keepdims=True)
    u, s, vt = svd(Hc, full_matrices=False)
    return (s[:k, None] * vt[:k])


def ridge_holdout(T, H, lams=(1e-2, 1e-1, 1, 10, 100, 1000), folds=5, seed=0):
    """cross validated held out defect with ridge chosen per predicate on inner folds"""
    rng = np.random.default_rng(seed)
    V = H.shape[1]
    Hh = np.vstack([H, np.ones((1, V))])
    perm = rng.permutation(V)
    m = T.shape[0]
    err = np.zeros((len(lams), m))
    norms = np.zeros(m)
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        A = Hh[:, tr].T
        AtA = A.T @ A
        AtY = A.T @ T[:, tr].T
        norms += ((T[:, te] - T[:, te].mean(1, keepdims=True)) ** 2).sum(1)
        for li, lam in enumerate(lams):
            reg = lam * np.eye(A.shape[1])
            reg[-1, -1] = 0.0
            W = np.linalg.solve(AtA + reg, AtY)
            pred = (Hh[:, te].T @ W).T
            err[li] += ((pred - T[:, te]) ** 2).sum(1)
    rel = np.sqrt(err / np.where(norms > 0, norms, np.nan))
    best = rel.argmin(0)
    return rel[best, np.arange(m)], np.array(lams)[best]


def probe_auc(T, H, folds=5, C=1.0, seed=0):
    """cross validated auc and balanced accuracy of logistic probe per predicate"""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    X = H.T
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in bar(T, desc="probes", total=T.shape[0]):
            y = t.astype(int)
            n_min = min(y.sum(), len(y) - y.sum())
            f = int(min(folds, n_min))
            if f < 2:
                out.append((np.nan, np.nan)); continue
            cv = StratifiedKFold(f, shuffle=True, random_state=seed)
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
            p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            out.append((roc_auc_score(y, p), balanced_accuracy_score(y, p > 0.5)))
    return np.array(out)


def forced_zero_angles(dim_H, rank_T1, V):
    """number of principal angles that vanish by dimension counting alone"""
    return max(0, dim_H + rank_T1 - V)


def dimension_sweep(T, H, ks, lams=(1e-1, 1, 10, 100), folds=5, tol=0.25, seed=0):
    """for each k, held out ridge defect of the top k pca geometry"""
    rows = []
    for k in bar(ks, desc="dimension sweep"):
        Hk = pca_geometry(H, k)
        d, _ = ridge_holdout(T, Hk, lams=lams, folds=folds, seed=seed)
        rows.append((k, float(np.nanmedian(d)), float(np.nanmean(d < tol))))
    return rows
