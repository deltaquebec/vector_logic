"""parallelogram (analogy) defects for feature pairs"""
import numpy as np
from itertools import combinations
from .progress import bar


def parallelogram_defects(T, features, H, max_pairs=200, max_quads=50, min_cell=5, groups=None, seed=0):
    """for feature pairs (P, Q), entities partitioned into the four cells
    (1,1), (1,0), (0,1), (0,0); each choice of one entity per cell gives a
    parallelogram vector c with entries +1, -1, -1, +1; report
    ||H c|| / mean ||h||, the normalized failure of h1 - h2 = h3 - h4
    """
    rng = np.random.default_rng(seed)
    scale = np.linalg.norm(H, axis=0).mean()
    V = H.shape[1]
    out = []
    pairs = list(combinations(range(len(features)), 2))
    rng.shuffle(pairs)
    for i, j in bar(pairs[:max_pairs], desc="parallelograms"):
        p, q = T[i], T[j]
        cells = [np.where((p == a) & (q == b))[0] for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))]
        if any(len(c) < min_cell for c in cells):
            continue
        ds, cs = [], []
        for _ in range(max_quads):
            if groups is not None:
                g = rng.choice(np.asarray(groups)[cells[0]])
                gc = [c[np.asarray(groups)[c] == g] for c in cells]
                if any(len(c) == 0 for c in gc):
                    continue
                e = [rng.choice(c) for c in gc]
            else:
                e = [rng.choice(c) for c in cells]
            v = H[:, e[0]] - H[:, e[1]] - H[:, e[2]] + H[:, e[3]]
            ds.append(np.linalg.norm(v) / scale)
            e4 = rng.integers(V)
            v = H[:, e[0]] - H[:, e[1]] - H[:, e[2]] + H[:, e4]
            cs.append(np.linalg.norm(v) / scale)
        if len(ds) >= 5:
            out.append((features[i], features[j], len(ds), float(np.mean(ds)), float(np.mean(cs))))
    return out
