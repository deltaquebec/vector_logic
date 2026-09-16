"""defect under a monotone of affine readout

three readouts per predicate
  affine
  sigmoid
  isotonic
"""
import warnings
import numpy as np
from .progress import bar


def _rel(pred, t):
    base = np.linalg.norm(t - t.mean())
    return np.linalg.norm(pred - t) / base if base > 0 else np.nan


def monotone_defects(T, H, folds=5, seed=0, Cs=(0.01, 0.1, 1, 10), lams=(1, 10, 100, 1000)):
    """returns array (m, 3): held out relative defect of affine, sigmoid, isotonic readouts"""
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
    rng = np.random.default_rng(seed)
    X = H.T
    V, m = X.shape[0], T.shape[0]
    out = np.zeros((m, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(bar(T, desc="monotone readouts", total=m)):
            y = t.astype(int)
            n_min = min(y.sum(), V - y.sum())
            f = int(min(folds, n_min))
            if f < 2:
                out[i] = np.nan; continue
            pa = np.zeros(V); ps = np.zeros(V); pi = np.zeros(V)
            cv = StratifiedKFold(f, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y):
                Xtr, Xte, ytr = X[tr], X[te], y[tr]
                # affine
                rr = RidgeCV(alphas=lams).fit(Xtr, ytr)
                pa[te] = rr.predict(Xte)
                # sigmoid
                lr = LogisticRegressionCV(Cs=Cs, cv=min(3, int(min(ytr.sum(), len(ytr) - ytr.sum()))),
                                          max_iter=3000, class_weight=None, scoring="neg_log_loss")
                lr.fit(Xtr, ytr)
                ps[te] = lr.predict_proba(Xte)[:, 1]
                # isotonic on cross fitted affine scores of the training fold
                inner = KFold(3, shuffle=True, random_state=seed)
                str_ = cross_val_predict(RidgeCV(alphas=lams), Xtr, ytr, cv=inner)
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(str_, ytr)
                pi[te] = iso.predict(rr.predict(Xte))
            out[i] = (_rel(pa, t), _rel(ps, t), _rel(pi, t))
    return out


def nonlinear_defect(T, H, folds=3, seed=0, hidden=(64,), alpha=1e-1, max_iter=800):
    """held out defect of a small mlp readout, scored like the monotone readouts"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = H.T
    V, m = X.shape[0], T.shape[0]
    out = np.zeros((m, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(bar(T, desc="mlp readouts", total=m)):
            y = t.astype(int)
            n_min = min(y.sum(), V - y.sum())
            f = int(min(folds, n_min))
            if f < 2:
                out[i] = np.nan; continue
            p = np.zeros(V)
            cv = StratifiedKFold(f, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y):
                clf = make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                                  max_iter=max_iter, early_stopping=False,
                                                  n_iter_no_change=20, random_state=seed))
                clf.fit(X[tr], y[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            out[i] = (_rel(p, t), roc_auc_score(y, p))
    return out
