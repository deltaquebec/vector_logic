"""strict linear separability of each predicate over a geometry decided by a linear feasibility problem

usage:
\tpython run_separability.py --glove G.txt --w2v W.bin --mcrae CONCS_FEATS_concstats_brm.txt
\tpython run_separability.py --glove G.txt --wordnet 20000 --min_true 30
"""
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlcompress import (load_glove, load_word2vec, geometry_matrix, load_mcrae,
                        truth_matrix, random_geometry)

log = logging.getLogger("separability")
PATH_FLAGS = {"glove", "w2v", "mcrae"}
LABEL_EXCLUDED = {"root", "name", "label"}


def find_root(start):
    """first ancestor holding pyproject.toml or .git, else the start directory."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start


def slug(text):
    """lowercase, characters outside letters digits hyphen period become hyphens."""
    return "".join(c if c.isalnum() or c in "-." else "-" for c in str(text).lower())


def derive_label(argv, stem):
    """label from the command line in the order passed; --label replaces it."""
    parts, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:]
            has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
            value = argv[i + 1] if has_value else None
            if key == "label" and value is not None:
                return slug(value)
            if key not in LABEL_EXCLUDED:
                if value is None:
                    parts.append(slug(key))
                elif key in PATH_FLAGS:
                    parts.append(f"{slug(key)}-{slug(Path(value).stem)}")
                else:
                    parts.append(f"{slug(key)}-{slug(value)}")
            i += 2 if has_value else 1
        else:
            parts.append(slug(Path(tok).stem) if os.sep in tok or "." in tok else slug(tok))
            i += 1
    return "_".join(parts) if parts else f"output_{stem}"


def make_run_folder(base, stamp, label):
    """create outputs/NAME/STAMP_LABEL, appending a two digit ordinal on collision."""
    base.mkdir(parents=True, exist_ok=True)
    k, width = 1, 2
    while True:
        name = f"{stamp}_{label}" if k == 1 else f"{stamp}_{label}_{k:0{width}d}"
        try:
            (base / name).mkdir(exist_ok=False)
            return base / name
        except FileExistsError:
            k += 1
            if k >= 10 ** width:
                width += 1


def point_latest(base, run):
    """replace base/latest with a link to run; junction fallback on Windows."""
    latest, tmp = base / "latest", base / "latest.tmp"
    if os.name == "nt":
        for p in (latest, tmp):
            if p.exists() or p.is_symlink():
                p.rmdir() if p.is_dir() and not p.is_symlink() else p.unlink()
        try:
            os.symlink(run, latest, target_is_directory=True)
        except OSError:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(latest), str(run)],
                           check=True, capture_output=True)
        return
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(run, tmp, target_is_directory=True)
    os.replace(tmp, latest)


def git_state(root):
    """commit hash and dirty flag when root is under git, else nulls."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                               capture_output=True, text=True).stdout.strip() != ""
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def separable(t, H):
    """lp feasibility of y_i (w h_i + b) >= 1 with y in {-1, +1}; returns (bool, status)."""
    d, V = H.shape
    y = np.where(t > 0.5, 1.0, -1.0)
    X = np.hstack([H.T, np.ones((V, 1))])
    A = -y[:, None] * X
    res = linprog(c=np.zeros(d + 1), A_ub=A, b_ub=-np.ones(V),
                  bounds=[(None, None)] * (d + 1), method="highs")
    return res.status == 0, res.message


def check_cell(name, H, T, feats, rows):
    """append one row per predicate for the geometry and for a gaussian control."""
    d, V = H.shape
    Hr = random_geometry(d, V)
    ok, okr = 0, 0
    for t, f in tqdm(list(zip(T, feats)), desc=name, leave=False):
        s, msg = separable(t, H)
        sr, _ = separable(t, Hr)
        ok += s
        okr += sr
        rows.append((name, f, int(t.sum()), V, d, int(s), int(sr), msg))
    log.info("%s: V = %d, d = %d, |P| = %d, 2(d + 1) = %d; separable %d of %d, control %d of %d",
             name, V, d, len(feats), 2 * (d + 1), ok, len(feats), okr, len(feats))


def main(args, run, meta):
    embs = []
    if args.glove:
        embs.append(("glove", load_glove(args.glove, vocab=None)))
    if args.w2v:
        embs.append(("w2v", load_word2vec(args.w2v, vocab=None)))
    if not embs:
        raise SystemExit("need --glove or --w2v")
    rows = []
    for ename, vecs in embs:
        if args.mcrae:
            df = load_mcrae(args.mcrae, args.min_prod_freq)
            ents = sorted(set(df["word"]))
            H, ents, missing = geometry_matrix(vecs, ents)
            log.info("mcrae x %s: %d entities matched, %d missing", ename, len(ents), len(missing))
            T, feats = truth_matrix(df, ents, min_pos=args.min_true, min_neg=args.min_true)
            check_cell(f"mcrae_{ename}", H, T, feats, rows)
        if args.wordnet:
            from vlcompress.norms import load_wordnet, frequent_nouns
            nouns = frequent_nouns(vecs, n=args.wordnet)
            df = load_wordnet(nouns, min_members=args.wn_min_members)
            ents = sorted(set(df["word"]))
            H, ents, _ = geometry_matrix(vecs, ents)
            T, feats = truth_matrix(df, ents, min_pos=args.min_true, min_neg=args.min_true)
            check_cell(f"wordnet_{ename}", H, T, feats, rows)
    out = pd.DataFrame(rows, columns=["cell", "feature", "n_true", "V", "d", "separable",
                                      "separable_control", "status"])
    out.to_csv(run / "tab_separability.csv", index=False)
    summary = out.groupby("cell")[["separable", "separable_control"]].mean()
    summary.to_csv(run / "tab_separability_summary.csv")
    log.info("fraction separable per cell:\n%s", summary.to_string())


def entry():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove")
    ap.add_argument("--w2v")
    ap.add_argument("--mcrae")
    ap.add_argument("--wordnet", type=int, default=0)
    ap.add_argument("--wn_min_members", type=int, default=30)
    ap.add_argument("--min_prod_freq", type=int, default=5)
    ap.add_argument("--min_true", type=int, default=15)
    ap.add_argument("--root")
    ap.add_argument("--name")
    ap.add_argument("--label")
    args = ap.parse_args()
    script = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else find_root(script.parent)
    name = args.name or script.stem
    label = derive_label(sys.argv[1:], script.stem)
    base = root / "outputs" / name
    run = make_run_folder(base, stamp, label)
    tmp = run / "tmp"
    tmp.mkdir()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(run / "log_run.txt", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    head, dirty = git_state(root)
    meta = {"args": vars(args), "root": str(root), "run": str(run), "command": sys.argv,
            "git_commit": head, "git_dirty": dirty, "stamp": stamp}
    (run / "meta_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    status = "completed"
    try:
        with logging_redirect_tqdm(loggers=[log]):
            main(args, run, meta)
        point_latest(base, run)
    except BaseException as e:
        status = f"{type(e).__name__}: {e}"
        log.exception("uncaught exception")
        raise
    finally:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        log.info("status: %s", status)


if __name__ == "__main__":
    entry()
