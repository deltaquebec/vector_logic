"""training geometry under mixed distributional and truth conditional objective"""
import numpy as np
from numpy.linalg import svd, lstsq, solve
from .progress import bar


def _center(E):
    return E - E.mean(1, keepdims=True)


def truth_floor(T, d):
    """min over d dimensional row spaces of the relative affine truth"""
    Tc = T - T.mean(1, keepdims=True)
    s = svd(Tc, compute_uv=False)
    tail = np.sqrt((s[d:] ** 2).sum()) if d < len(s) else 0.0
    return tail / np.linalg.norm(Tc)


def dist_retained(E, d, constraint=None):
    """fraction of ||E||^2 captured by the best d dimensional row space"""
    E = _center(E)
    tot = (E ** 2).sum()
    if constraint is None:
        s = svd(E, compute_uv=False)
        return (s[:d] ** 2).sum() / tot
    u, s, vt = svd(constraint, full_matrices=False)
    r = (s > s.max() * 1e-10).sum()
    B = vt[:r]                       # orthonormal basis of rs(constraint), r x V
    Ep = E - (E @ B.T) @ B           # E projected off rs(constraint)
    s2 = svd(Ep, compute_uv=False)
    captured = ((E @ B.T) ** 2).sum() + (s2[:max(0, d - r)] ** 2).sum()
    return captured / tot


def price_of_exactness(E, T, d, seed=0):
    """distributional variance retained at dimension d: unconstrained, exact
    (rs(H) contains rs[T; 1]), and random constraint of the same rank"""
    T1 = np.vstack([T, np.ones((1, T.shape[1]))])
    r = np.linalg.matrix_rank(T1)
    R = np.random.default_rng(seed).standard_normal((r, T.shape[1]))
    return dict(rank=r, unconstrained=dist_retained(E, d),
                exact=dist_retained(E, d, T1), random=dist_retained(E, d, R))


def als_train(E, T, d, lam, iters=100, seed=0, ridge=1e-8, H0=None, desc=None):
    """alternating least squares for the mixed objective; returns (H, history)"""
    rng = np.random.default_rng(seed)
    E = _center(E)
    D, V = E.shape
    m = T.shape[0]
    Tc = T - T.mean(1, keepdims=True)
    nE = (E ** 2).sum()
    nT = (Tc ** 2).sum()
    a, b = (1 - lam) / nE, lam / nT
    H = rng.standard_normal((d, V)) if H0 is None else H0.copy()
    hist = []
    it_range = range(iters) if desc is None else bar(range(iters), desc=desc, leave=False)
    for it in it_range:
        # inner solves
        W = lstsq(H.T, E.T, rcond=None)[0].T                      # D x d
        H1 = np.vstack([H, np.ones((1, V))])
        WT1 = lstsq(H1.T, T.T, rcond=None)[0].T                   # m x (d+1)
        WT, w0 = WT1[:, :d], WT1[:, d:]
        # H solve: (a W^T W + b WT^T WT) H = a W^T E + b WT^T (T - w0 1^T)
        A = a * W.T @ W + b * WT.T @ WT + ridge * np.eye(d)
        Bm = a * W.T @ E + b * WT.T @ (T - w0)
        H = solve(A, Bm)
        # renormalize rows to keep scale (loss is GL(d) invariant)
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
        # evaluate
        H1 = np.vstack([H, np.ones((1, V))])
        WT1 = lstsq(H1.T, T.T, rcond=None)[0].T
        td = np.linalg.norm(WT1 @ H1 - T) / np.sqrt(nT)
        W = lstsq(H.T, E.T, rcond=None)[0].T
        de = np.linalg.norm(W @ H - E) / np.sqrt(nE)
        hist.append((it, td, de))
        if desc is not None and hasattr(it_range, 'set_postfix'):
            it_range.set_postfix(truth=f'{td:.3f}', dist=f'{de:.3f}')
    return H, np.array(hist)


def per_predicate_defect(H, T):
    """in sample affine defect per predicate on trained geometry"""
    V = H.shape[1]
    H1 = np.vstack([H, np.ones((1, V))])
    W = lstsq(H1.T, T.T, rcond=None)[0].T
    R = W @ H1 - T
    base = np.linalg.norm(T - T.mean(1, keepdims=True), axis=1)
    return np.linalg.norm(R, axis=1) / np.where(base > 0, base, np.nan)


def sweep(E, T, ds, lams, iters=60, seed=0):
    """grid over dimension and weight; rows (d, lam, truth defect, dist error,
    truth floor at d, fraction of predicates with defect < 0.05)"""
    rows = []
    grid = [(d, lam) for d in ds for lam in lams]
    for d, lam in bar(grid, desc="als grid", leave=True):
        fl = truth_floor(T, d)
        if True:
            H, hist = als_train(E, T, d, lam, iters=iters, seed=seed, desc=f"d={d} lam={lam:g}")
            pdf = per_predicate_defect(H, T)
            rows.append((d, lam, hist[-1, 1], hist[-1, 2], fl, float(np.nanmean(pdf < 0.05))))
    return rows
