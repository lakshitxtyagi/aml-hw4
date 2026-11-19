"""Time-honest backtests of model.build on 2025 data (design tool; not needed to make the tables).

T1: each 2025H2 listing with >=3 first-quarter holders, model built at its listing date.
T2: each pair of such listings, model built at the later listing date with both treated as cold.
T3: decision 2025-08-15, base Jun-2025 holdings, outcome = new positions reported for Dec-2025.
"""
import itertools
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from hw4eval import auc, cos_matrix, listing_outcomes, load, test3
from make_vectors import build

h, c, f, m = load()
LISTED = c.set_index("stock_id").listing_date
OUT = listing_outcomes(h, f, c, c.loc[c.status == "ipo_2025H2", "stock_id"])
EST = c.loc[c.status == "established", "stock_id"].tolist()
L25 = c.loc[c.status == "ipo_2025H2", "stock_id"].tolist()


def sic_onehot(digits=2):
    idx = pd.read_csv("cache/filing_index.csv", dtype={"sic": str}).set_index("stock_id").loc[c.stock_id]
    codes = idx.sic.fillna("0000").str[:digits]
    return pd.get_dummies(codes).to_numpy(float)


def load_emb(name):
    """'random', a model name, 'sic', or 'model+sicW' (e.g. 'bge+sic0.5') to append weighted SIC one-hots."""
    if "|" in name:  # concatenation of unit-normalised embeddings from several models
        return np.hstack([load_emb(n) for n in name.split("|")])
    if name == "random":
        return np.random.default_rng(0).normal(size=(len(c), 64))
    if name == "sic":
        return sic_onehot()
    if "+sic" in name:
        base, w = name.split("+sic")
        S = sic_onehot()
        return np.hstack([np.load(f"cache/emb_{base}.npy"), float(w or 1) * S / np.sqrt(2)])
    return np.load(f"cache/emb_{name}.npy")


def run(emb, params=None, verbose=False, which=("t1", "t2", "t3")):
    key = lambda D, cold: (D, tuple(sorted(cold)))
    cache = {}

    def C_at(D, cold=()):
        k = key(D, cold)
        if k not in cache:
            U, V = build(h, f, c, m, emb, D, cold, params)
            cache[k] = cos_matrix(U, V)
        return cache[k]

    res = {}
    if "t1" in which:
        a = {}
        for s, (q, filers, holders) in OUT.items():
            C = C_at(LISTED[s], (s,))
            a[s] = auc(C.loc[filers, s].values, [x in holders for x in filers])
        res["t1"] = np.mean(list(a.values()))
        q3 = [s for s in a if LISTED[s] <= "2025-09-30"]
        res["t1_q3"] = np.mean([a[s] for s in q3])
        res["t1_q4"] = np.mean([a[s] for s in a if s not in q3])
        if verbose:
            print({s: round(v, 3) for s, v in a.items()})
    if "t2" in which:
        vals = []
        for A, B in itertools.combinations(OUT, 2):
            I = sorted(OUT[A][2] - OUT[B][2])
            J = sorted(OUT[B][2] - OUT[A][2])
            if not I or not J:
                continue
            C = C_at(max(LISTED[A], LISTED[B]), (A, B))
            d = (C.loc[I, A].values[:, None] + C.loc[J, B].values[None, :]) - (
                C.loc[I, B].values[:, None] + C.loc[J, A].values[None, :])
            vals.append(((d > 0) + 0.5 * (d == 0)).mean())
        res["t2"] = np.mean(vals)
        res["t2_n"] = len(vals)
    if "t3" in which:
        C = C_at("2025-08-15")
        res["t3"] = test3(C, h, f, "2025-06-30", "2025-12-31", EST)[0]
        C = C_at("2025-11-15")
        cands = EST + [s for s in L25 if LISTED[s] <= "2025-09-30"]
        res["t3b"] = test3(C, h, f, "2025-09-30", "2025-12-31", cands)[0]
    return res


def cv_final(emb, params=None, n_folds=5, seed=0):
    """Diagnostic (not time-honest): at the real decision date 2026-03-02, hide the holdings of a fold
    of 2025H2 listings, treat them as cold, and score T1 / T2 on them."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(sorted(OUT))
    t1, t2 = [], []
    for fold in np.array_split(ids, n_folds):
        U, V = build(h, f, c, m, emb, "2026-03-02", tuple(fold), params)
        C = cos_matrix(U, V)
        for s in fold:
            _, filers, holders = OUT[s]
            t1.append(auc(C.loc[filers, s].values, [x in holders for x in filers]))
        for A, B in itertools.combinations(fold, 2):
            I, J_ = sorted(OUT[A][2] - OUT[B][2]), sorted(OUT[B][2] - OUT[A][2])
            if I and J_:
                d = (C.loc[I, A].values[:, None] + C.loc[J_, B].values[None, :]) - (
                    C.loc[I, B].values[:, None] + C.loc[J_, A].values[None, :])
                t2.append(((d > 0) + 0.5 * (d == 0)).mean())
    return {"cv_t1": float(np.mean(t1)), "cv_t2": float(np.mean(t2)), "cv_t2_n": len(t2)}


def fmt(r):
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items())


if __name__ == "__main__":
    emb = load_emb(sys.argv[1] if len(sys.argv) > 1 else "random")
    print(fmt(run(emb, verbose=True)))
