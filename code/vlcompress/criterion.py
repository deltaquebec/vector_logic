"""rank criterion, defect, separability, and principal angles"""
import numpy as np
from numpy.linalg import matrix_rank, svd, lstsq


def augmented_rank(T, tol=None):
    """rank of [T; 1], the minimal exact dimension of cor:mindim."""
    ones = np.ones((1, T.shape[1]))
    return matrix_rank(np.vstack([T, ones]), tol=tol)


def projector(H, rcond=1e-10):
    """orthogonal projector H^+ H onto rs(H), acting on row vectors from the right."""
    return np.linalg.pinv(H, rcond=rcond) @ H


def defect(t, H, P=None, affine=False, relative=True):
    """least squares defect delta(P; H) = min_w ||w H - t||_2 """
    Hh = np.vstack([H, np.ones((1, H.shape[1]))]) if affine else H
    if P is None:
        P = projector(Hh)
    resid = t - t @ P
    d = np.linalg.norm(resid)
    if relative:
        base = np.linalg.norm(t - t.mean()) if affine else np.linalg.norm(t)
        d = d / base if base > 0 else np.nan
    return d


def defect_table(T, features, H, affine=True):
    """defect for every predicate; returns list of (feature, n_true, defect, ones_defect)."""
    P = projector(np.vstack([H, np.ones((1, H.shape[1]))]) if affine else H)
    ones_def = defect(np.ones(H.shape[1]), H, projector(H), affine=False)
    rows = []
    for f, t in zip(features, T):
        rows.append((f, int(t.sum()), defect(t, H, P, affine=affine), ones_def))
    return rows


def holdout_defect(T, H, folds=5, affine=True, ridge=0.0, seed=0):
    """cross validated defect: fit w on training entities, score on held out"""
    rng = np.random.default_rng(seed)
    V = H.shape[1]
    perm = rng.permutation(V)
    Hh = np.vstack([H, np.ones((1, V))]) if affine else H
    out = np.zeros(T.shape[0])
    norms = np.zeros(T.shape[0])
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        A = Hh[:, tr].T
        if ridge > 0:
            W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ T[:, tr].T)
        else:
            W = lstsq(A, T[:, tr].T, rcond=None)[0]
        pred = (Hh[:, te].T @ W).T
        out += ((pred - T[:, te]) ** 2).sum(1)
        norms += ((T[:, te] - T[:, te].mean(1, keepdims=True)) ** 2).sum(1)
    return np.sqrt(out / np.where(norms > 0, norms, np.nan))


def separability(t, H, C=1e4, folds=None, seed=0):
    """thresholded lift: linear separability by logistic regression"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    X = H.T
    y = t.astype(int)
    clf = LogisticRegression(C=C, max_iter=5000)
    if folds is None:
        clf.fit(X, y)
        return clf.score(X, y)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv).mean()


def principal_angles(H, T, deg=True):
    """principal angles between rs(H) and rs[T; 1]"""
    def orth_rows(M):
        u, s, vt = svd(M, full_matrices=False)
        r = (s > s.max() * 1e-10).sum()
        return vt[:r]
    A = orth_rows(H)
    B = orth_rows(np.vstack([T, np.ones((1, T.shape[1]))]))
    s = svd(A @ B.T, compute_uv=False)
    ang = np.arccos(np.clip(s, -1, 1))
    return np.degrees(ang) if deg else ang


def random_geometry(d, V, seed=0):
    """gaussian control geometry of same shape; expected relative defect
    of generic row is about sqrt(1 - (d+1)/V) in affine case"""
    return np.random.default_rng(seed).standard_normal((d, V))


def admissible_dependences(T):
    """basis of rs[T; 1]^perp, the space of admissible dependences"""
    M = np.vstack([T, np.ones((1, T.shape[1]))])
    u, s, vt = svd(M)
    r = (s > s.max() * 1e-10).sum()
    return vt[r:]
