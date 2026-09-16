"""run compression diagnostics on trained embeddings against feature norms

usage:
\tpython run_diagnostics.py --glove glove.6B.300d.txt --w2v GoogleNews-vectors-negative300.bin \\
\t\t--mcrae CONCS_FEATS_concstats_brm.txt --out results/
\tpython run_diagnostics.py --synthetic   (smoke test without data)
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import warnings, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)
from vlcompress import (
    load_glove, load_word2vec, geometry_matrix, load_mcrae, load_binder,
    truth_matrix, augmented_rank, defect_table, principal_angles,
    random_geometry, identity_defect, equivalence_defect, parallelogram_defects,
    ridge_holdout, probe_auc, forced_zero_angles, dimension_sweep, monotone_defects, nonlinear_defect,
)


def synthetic(V=200, d=50, m=12, seed=0):
    """boolean feature lexicon on V entities and a geometry that carries it
    exactly (rank m + 1) plus noise; a smoke test for the pipeline"""
    rng = np.random.default_rng(seed)
    T = (rng.random((m, V)) < 0.35).astype(float)
    feats = [f"f{k}" for k in range(m)]
    ents = [f"e{i}" for i in range(V)]
    W = rng.standard_normal((d, m + 1))
    H = W @ np.vstack([T, np.ones((1, V))]) + 0.05 * rng.standard_normal((d, V))
    return H, ents, T, feats


def analyze(name, H, ents, T, feats, out, taxonomic=None, ks=(5, 10, 20, 40, 80, 160, 300), parallelograms=True, calibrate=False, nonlinear=False):
    d, V = H.shape
    r = augmented_rank(T)
    rH = np.linalg.matrix_rank(H)
    print(f"\n{name}: d = {d}, V = {V}, |P| = {T.shape[0]}, rank[T;1] = {r}, rank H = {rH}")
    regime = "free" if rH == V else "compressed"
    print(f"\tregime: {regime}; exact monadic compression needs d >= {r}")
    fz = forced_zero_angles(rH, r, V)
    if fz > 0:
        print(f"\twarning: {fz} principal angles vanish by dimension counting; angles uninformative,")
        print(f"\t         raise --min_true or reduce the lexicon until rank H + rank[T;1] < V")

    rows = defect_table(T, feats, H, affine=True)
    df = pd.DataFrame(rows, columns=["feature", "n_true", "defect_affine", "ones_defect"])
    hd, lam = ridge_holdout(T, H)
    df["defect_holdout"] = hd
    df["ridge_lambda"] = lam
    Hr = random_geometry(d, V)
    df["defect_control"] = [x[2] for x in defect_table(T, feats, Hr, affine=True)]
    df["defect_holdout_control"] = ridge_holdout(T, Hr)[0]
    pa = probe_auc(T, H)
    df["probe_auc"] = pa[:, 0]
    df["probe_bal_acc"] = pa[:, 1]
    if calibrate:
        md = monotone_defects(T, H)
        df["defect_affine_cv"] = md[:, 0]
        df["defect_sigmoid"] = md[:, 1]
        df["defect_isotonic"] = md[:, 2]
        mdr = monotone_defects(T, Hr)
        df["defect_isotonic_control"] = mdr[:, 2]
        if nonlinear:
            nd = nonlinear_defect(T, H)
            df["defect_mlp"] = nd[:, 0]
            df["mlp_auc"] = nd[:, 1]
    df = df.sort_values("defect_holdout")
    df.to_csv(os.path.join(out, f"{name}_defects.csv"), index=False)
    if calibrate:
        print(f"\tmonotone readouts, median held out defect: affine {df.defect_affine_cv.median():.3f}; "
              f"sigmoid {df.defect_sigmoid.median():.3f}; isotonic {df.defect_isotonic.median():.3f}; "
              f"isotonic control {df.defect_isotonic_control.median():.3f}")
        print(f"\tfraction with isotonic defect < 0.5: {(df.defect_isotonic < 0.5).mean():.3f}; < 0.25: {(df.defect_isotonic < 0.25).mean():.3f}")
        if nonlinear:
            print(f"\tmlp readout, median held out defect: {df.defect_mlp.median():.3f}; median mlp auc {df.mlp_auc.median():.3f}; "
                  f"median (isotonic - mlp): {(df.defect_isotonic - df.defect_mlp).median():.3f}")
            print(f"\tfraction with mlp defect < 0.5: {(df.defect_mlp < 0.5).mean():.3f}")
        q = df.defect_isotonic.quantile([.1,.25,.5,.75,.9]).values
        print(f"\tisotonic defect quantiles 10/25/50/75/90: {np.round(q, 3)}")
        print("\tbest under isotonic squash:")
        for _, r_ in df.sort_values("defect_isotonic").head(6).iterrows():
            extra = f" mlp={r_.defect_mlp:.3f}" if nonlinear else ""
            print(f"\t\t{r_.feature:30s} n={r_.n_true:5d} affine={r_.defect_affine_cv:.3f} sigmoid={r_.defect_sigmoid:.3f} isotonic={r_.defect_isotonic:.3f}{extra} auc={r_.probe_auc:.3f}")
    print(f"\tmedian defect (in sample, affine): {df.defect_affine.median():.3f}; control {df.defect_control.median():.3f}")
    print(f"\tmedian defect (held out, ridge cv): {df.defect_holdout.median():.3f}; control {df.defect_holdout_control.median():.3f}")
    print(f"\tfraction of predicates with held out defect < 0.5: {(df.defect_holdout < 0.5).mean():.3f}; control {(df.defect_holdout_control < 0.5).mean():.3f}")
    print(f"\tmedian probe auc (cv, balanced): {df.probe_auc.median():.3f}; balanced acc {df.probe_bal_acc.median():.3f}")
    print(f"\tones row defect (linear, no bias): {df.ones_defect.iloc[0]:.3f}")
    print("\tbest carried predicates:")
    for _, r_ in df.head(8).iterrows():
        print(f"\t\t{r_.feature:30s} n={r_.n_true:4d} holdout={r_.defect_holdout:.3f} auc={r_.probe_auc:.3f}")
    print("\tworst carried predicates:")
    for _, r_ in df.tail(5).iterrows():
        print(f"\t\t{r_.feature:30s} n={r_.n_true:4d} holdout={r_.defect_holdout:.3f} auc={r_.probe_auc:.3f}")

    ang = principal_angles(H, T)
    angr = principal_angles(Hr, T)
    np.savetxt(os.path.join(out, f"{name}_angles.txt"), ang)
    k = min(10, len(ang))
    print(f"\tprincipal angles rs(H) vs rs[T;1], first informative ten (deg): {np.round(ang[fz:fz + k], 1)}")
    print(f"\t                    control                                     : {np.round(angr[fz:fz + k], 1)}")

    ks = [k for k in ks if k <= d]
    if not ks:
        print(f"\tidentity relation defect: {identity_defect(H):.3f} (control {identity_defect(Hr):.3f})")
        return
    sw = dimension_sweep(T, H, ks)
    swr = dimension_sweep(T, Hr, ks)
    pd.DataFrame(sw, columns=["k", "median_defect", "frac_carried"]).to_csv(os.path.join(out, f"{name}_sweep.csv"), index=False)
    print("\tdimension sweep (top k pca directions): k, median held out defect, fraction carried (< 0.25); control in brackets")
    for (k, m, f), (_, mr, fr) in zip(sw, swr):
        print(f"\t\tk={k:4d}  {m:.3f}  {f:.3f}   [{mr:.3f}  {fr:.3f}]")

    print(f"\tidentity relation defect: {identity_defect(H):.3f} (control {identity_defect(Hr):.3f})")
    if taxonomic is not None:
        print(f"\tequivalence relation defect ({taxonomic[0]}): {equivalence_defect(taxonomic[1], H):.3f}")

    if not parallelograms:
        return
    groups = taxonomic[1] if taxonomic is not None else None
    par = parallelogram_defects(T, feats, H, max_pairs=500, min_cell=5, groups=groups)
    if par:
        pdf = pd.DataFrame(par, columns=["P", "Q", "n", "defect", "control"]).sort_values("defect")
        pdf.to_csv(os.path.join(out, f"{name}_parallelograms.csv"), index=False)
        print(f"\tparallelogram defect, median over pairs: {pdf.defect.median():.3f}; control {pdf.control.median():.3f}")
        for _, r_ in pdf.head(5).iterrows():
            print(f"\t\t{r_.P:24s} x {r_.Q:24s} defect={r_.defect:.3f} control={r_.control:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove"); ap.add_argument("--w2v")
    ap.add_argument("--mcrae"); ap.add_argument("--binder")
    ap.add_argument("--wordnet", type=int, default=0,
                    help="if > 0, build a wordnet hypernym lexicon over this many frequent nouns (needs nltk wordnet)")
    ap.add_argument("--wn_min_members", type=int, default=30)
    ap.add_argument("--wn_max_members", type=int, default=500)
    ap.add_argument("--wn_min_depth", type=int, default=4)
    ap.add_argument("--wn_polysemous", action="store_true", help="keep polysemous nouns (first sense labels)")
    ap.add_argument("--wn_sense_share", type=float, default=None)
    ap.add_argument("--min_prod_freq", type=int, default=5)
    ap.add_argument("--min_true", type=int, default=15,
                    help="drop predicates true of fewer entities; keeps rank[T;1] well below V")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="also fit sigmoid and isotonic readouts (slower)")
    ap.add_argument("--no_sweep", action="store_true")
    ap.add_argument("--nonlinear", action="store_true", help="with --calibrate, also fit an mlp readout")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.synthetic:
        analyze("synthetic", *synthetic(), out=a.out, calibrate=a.calibrate, nonlinear=a.nonlinear)
        return

    norms = []
    if a.mcrae:
        norms.append(("mcrae", load_mcrae(a.mcrae, a.min_prod_freq)))
    if a.binder:
        norms.append(("binder", load_binder(a.binder)))
    if not norms and not a.wordnet:
        sys.exit("need --mcrae, --binder, or --wordnet (or --synthetic)")

    words = set()
    for _, df in norms:
        words |= set(df["word"])
    embs = []
    if a.glove:
        embs.append(("glove", load_glove(a.glove, vocab=None if a.wordnet else words)))
    if a.w2v:
        embs.append(("w2v", load_word2vec(a.w2v, vocab=None if a.wordnet else words)))
    if not embs:
        sys.exit("need --glove or --w2v")

    if a.wordnet:
        from vlcompress.norms import load_wordnet, frequent_nouns
        for ename, vecs in embs:
            nouns = frequent_nouns(vecs, n=a.wordnet)
            df = load_wordnet(nouns, min_members=a.wn_min_members, max_members=a.wn_max_members,
                              min_depth=a.wn_min_depth, monosemous=not a.wn_polysemous,
                              sense_share=a.wn_sense_share)
            ents = sorted(set(df["word"]))
            H, ents, _ = geometry_matrix(vecs, ents)
            T, feats = truth_matrix(df, ents, min_pos=a.min_true, min_neg=a.min_true)
            print(f"\n[wordnet x {ename}] {len(ents)} nouns, {len(feats)} hypernym predicates")
            analyze(f"wordnet_{ename}", H, ents, T, feats, a.out, parallelograms=False, calibrate=a.calibrate, nonlinear=a.nonlinear, ks=() if a.no_sweep else (5, 10, 20, 40, 80, 160, 300))

    for nname, df in norms:
        ents_all = sorted(set(df["word"]))
        for ename, vecs in embs:
            H, ents, missing = geometry_matrix(vecs, ents_all)
            print(f"\n[{nname} x {ename}] {len(ents)} entities matched, {len(missing)} missing")
            T, feats = truth_matrix(df, ents, min_pos=a.min_true, min_neg=a.min_true)
            tax = None
            if nname == "mcrae":
                # equivalence relation from the taxonomic feature each concept lists first, if any
                lab = []
                for e in ents:
                    fs = df[(df.word == e) & df.feature.str.startswith(("a_", "an_"))].feature
                    lab.append(fs.iloc[0] if len(fs) else "none")
                tax = ("taxonomic a_/an_ feature", lab)
            analyze(f"{nname}_{ename}", H, ents, T, feats, a.out, taxonomic=tax, calibrate=a.calibrate, nonlinear=a.nonlinear, ks=() if a.no_sweep else (5, 10, 20, 40, 80, 160, 300))


if __name__ == "__main__":
    main()
