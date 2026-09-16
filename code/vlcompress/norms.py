"""feature norms as truth matrices T in {0,1}^{|P| x V}"""
import numpy as np
import pandas as pd


def load_mcrae(path, min_prod_freq=5, drop_taxonomic=False):
    """McRae et al. 2005 CONCS_FEATS_concstats_brm.txt (tab separated)"""
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    cols = {c.lower(): c for c in df.columns}
    concept = cols.get("concept")
    feature = cols.get("feature")
    freq = cols.get("prod_freq")
    df = df[[concept, feature, freq]].rename(
        columns={concept: "concept", feature: "feature", freq: "freq"})
    df = df[df["freq"] >= min_prod_freq]
    if drop_taxonomic:
        df = df[~df["feature"].str.startswith(("a_", "an_"))]
    # McRae uses e.g. bat_(animal); strip disambiguation for lookup, keep raw
    df["word"] = df["concept"].str.replace(r"_\(.*\)$", "", regex=True)
    df["word"] = df["word"].str.replace("_", " ")
    return df[["concept", "word", "feature"]].reset_index(drop=True)


def load_binder(path, threshold=3.0, sheet=0):
    """Binder et al. 2016 WordSet1_Ratings.xlsx; graded ratings"""
    df = pd.read_excel(path, sheet_name=sheet)
    cols = {c.lower(): c for c in df.columns}
    wcol = cols.get("word") or cols.get("lemma") or df.columns[0]
    numeric = [c for c in df.columns
               if c != wcol and pd.api.types.is_numeric_dtype(df[c])]
    skip = {"n", "n.", "wordnum", "cscore", "count"}
    numeric = [c for c in numeric if str(c).lower() not in skip]
    rows = []
    for _, r in df.iterrows():
        w = str(r[wcol]).strip().lower()
        for c in numeric:
            v = r[c]
            if pd.notna(v) and float(v) >= threshold:
                rows.append((w, w, str(c)))
    return pd.DataFrame(rows, columns=["concept", "word", "feature"])


def truth_matrix(long_df, entities, min_pos=3, min_neg=3):
    """binary truth matrix over the given entity order"""
    idx = {e: i for i, e in enumerate(entities)}
    feats = sorted(long_df["feature"].unique())
    T = np.zeros((len(feats), len(entities)), dtype=np.float64)
    fidx = {f: i for i, f in enumerate(feats)}
    for w, f in zip(long_df["word"], long_df["feature"]):
        if w in idx:
            T[fidx[f], idx[w]] = 1.0
    pos = T.sum(1)
    keep = (pos >= min_pos) & (len(entities) - pos >= min_neg)
    return T[keep], [f for f, k in zip(feats, keep) if k]


def load_wordnet(words, min_members=30, max_members=500, min_depth=4, max_depth=None,
                 monosemous=True, sense_share=None):
    """hypernym lexicon over a large noun set; requires nltk with wordnet data"""
    from nltk.corpus import wordnet as wn
    from collections import defaultdict
    from .progress import bar
    members = defaultdict(set)
    for w in bar(words, desc="wordnet hypernyms", total=len(words)):
        syns = wn.synsets(w, pos=wn.NOUN)
        if not syns:
            continue
        if monosemous and len(syns) > 1:
            continue
        s0 = syns[0]
        if sense_share is not None and len(syns) > 1:
            counts = [sum(l.count() for l in s.lemmas() if l.name().lower() == w) for s in syns]
            tot = sum(counts)
            if tot == 0 or counts[0] / tot < sense_share:
                continue
        for path in s0.hypernym_paths():
            for h in path:
                dep = h.min_depth()
                if dep < min_depth or (max_depth is not None and dep > max_depth):
                    continue
                if h == s0:
                    continue
                members[h.name()].add(w)
    rows = []
    for h, ms in members.items():
        if len(ms) < min_members:
            continue
        if max_members is not None and len(ms) > max_members:
            continue
        for w in ms:
            rows.append((w, w, h))
    return pd.DataFrame(rows, columns=["concept", "word", "feature"])


def frequent_nouns(vecs, n=5000, min_len=3):
    """first n vocabulary items of an embedding dict that wordnet lists as nouns"""
    from nltk.corpus import wordnet as wn
    from .progress import bar
    out = []
    for w in bar(vecs, desc="scanning nouns", total=len(vecs)):
        if len(w) < min_len or not w.isalpha():
            continue
        if wn.synsets(w, pos=wn.NOUN):
            out.append(w)
        if len(out) >= n:
            break
    return out
