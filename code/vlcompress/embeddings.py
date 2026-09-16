"""loading trained embeddings and forming the geometry matrix H in R^{d x V}"""
import numpy as np
from .progress import bar


def load_glove(path, vocab=None, lowercase=True):
    """read glove text format; returns dict word -> vector
    vocab: optional set restricting which words to keep, to save memory
    """
    vecs = {}
    with open(path, encoding="utf8", errors="ignore") as f:
        for line in bar(f, desc="reading glove"):
            parts = line.rstrip().split(" ")
            w = parts[0]
            if lowercase:
                w = w.lower()
            if vocab is not None and w not in vocab:
                continue
            if len(parts) < 3:
                continue
            vecs[w] = np.asarray(parts[1:], dtype=np.float64)
    return vecs


def load_word2vec(path, vocab=None, binary=True):
    """read word2vec (GoogleNews style) via gensim; returns dict word -> vector
    vocab: optional set; words are looked up as given, then lowercased,
      then capitalized, since GoogleNews is case sensitive
    """
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format(path, binary=binary)
    vecs = {}
    words = vocab if vocab is not None else kv.index_to_key
    for w in bar(words, desc="matching word2vec", total=len(words)):
        for cand in (w, w.lower(), w.capitalize()):
            if cand in kv:
                vecs[w] = np.asarray(kv[cand], dtype=np.float64)
                break
    return vecs


def geometry_matrix(vecs, entities):
    """stack entity vectors as columns; returns (H, kept, missing)
    H has shape (d, V) with V = len(kept)
    """
    kept, missing = [], []
    for e in entities:
        (kept if e in vecs else missing).append(e)
    if not kept:
        raise ValueError("no entities found in embedding vocabulary")
    H = np.stack([vecs[e] for e in kept], axis=1)
    return H, kept, missing
