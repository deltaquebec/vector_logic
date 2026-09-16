"""bilinear factorization defect for binary relations"""
import numpy as np


def bilinear_defect(TR, H, rcond=1e-10, relative=True):
    """min_A ||H^T A H - T_R||_F; optimal A = H^{+T} T_R H^{+}
    """
    Hp = np.linalg.pinv(H, rcond=rcond)
    A = Hp.T @ TR @ Hp
    resid = np.linalg.norm(H.T @ A @ H - TR)
    if relative:
        resid /= np.linalg.norm(TR)
    return resid, A


def identity_defect(H, **kw):
    """defect of the identity relation; nonzero whenever rank H < V"""
    return bilinear_defect(np.eye(H.shape[1]), H, **kw)[0]


def equivalence_defect(labels, H, **kw):
    """defect of the equivalence relation induced by a labeling; rank = number of classes"""
    labels = np.asarray(labels)
    TR = (labels[:, None] == labels[None, :]).astype(float)
    return bilinear_defect(TR, H, **kw)[0]
