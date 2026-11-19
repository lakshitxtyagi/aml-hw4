"""Local re-implementation of the three HW4 tests, plus time-honest backtest setups.

Not part of the submission pipeline; used only to choose the design.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA = "data"


def load():
    h = pd.read_csv(f"{DATA}/holdings.csv")
    c = pd.read_csv(f"{DATA}/companies.csv")
    f = pd.read_csv(f"{DATA}/filings.csv")
    m = pd.read_csv(f"{DATA}/manager_info.csv")
    return h, c, f, m


def auc(scores, y):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(scores)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def unit(X):
    X = np.asarray(X, float)
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def cos_matrix(U, V):
    """U: DataFrame managers x k, V: DataFrame stocks x k -> DataFrame of cosines."""
    return pd.DataFrame(unit(U.values) @ unit(V.values).T, index=U.index, columns=V.index)


def first_qe(date):
    for q in ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]:
        if date <= q:
            return q
    return None


def listing_outcomes(h, f, c, stock_ids, min_holders=3):
    """For each listing: (quarter, filers, holders) at first quarter-end on/after listing."""
    lst = c.set_index("stock_id").listing_date
    out = {}
    for s in stock_ids:
        q = first_qe(lst[s])
        filers = set(f.loc[f.quarter_end == q, "manager_id"])
        holders = set(h.loc[(h.quarter_end == q) & (h.stock_id == s), "manager_id"]) & filers
        if len(holders) >= min_holders:
            out[s] = (q, sorted(filers), holders)
    return out


def test1(C, outcomes):
    res = {}
    for s, (q, filers, holders) in outcomes.items():
        sc = C.loc[filers, s].values
        res[s] = auc(sc, [m in holders for m in filers])
    return float(np.nanmean(list(res.values()))), res


def test2(C, outcomes):
    vals = []
    for a, b in itertools.combinations(outcomes, 2):
        _, fa, ha = outcomes[a]
        _, fb, hb = outcomes[b]
        I = sorted(ha - hb)
        J = sorted(hb - ha)
        if not I or not J:
            continue
        ci_a = C.loc[I, a].values[:, None]
        ci_b = C.loc[I, b].values[:, None]
        cj_a = C.loc[J, a].values[None, :]
        cj_b = C.loc[J, b].values[None, :]
        d = (ci_a + cj_b) - (ci_b + cj_a)
        vals.append(((d > 0) + 0.5 * (d == 0)).mean())
    return float(np.mean(vals)) if vals else np.nan


def test3(C, h, f, base_q, target_q, candidates, min_new=3, n_groups=4):
    """Per-investor AUC for new holdings at target_q among candidates not held at base_q."""
    base = h[h.quarter_end == base_q]
    tgt = h[h.quarter_end == target_q]
    filers = sorted(set(f.loc[f.quarter_end == target_q, "manager_id"]))
    held_b = set(zip(base.manager_id, base.stock_id))
    held_t = set(zip(tgt.manager_id, tgt.stock_id))
    newh = {(m, s) for (m, s) in held_t if (m, s) not in held_b and m in filers}
    nnew = pd.Series([s for _, s in newh]).value_counts()
    cands = [s for s in candidates if nnew.get(s, 0) >= min_new]
    pop = base.groupby("stock_id").size().reindex(cands).fillna(0)
    # four popularity groups (quantile bins on candidate popularity)
    grp = pd.qcut(pop.rank(method="first"), n_groups, labels=False)
    per_inv = []
    for m in filers:
        aucs = []
        for g in range(n_groups):
            ss = [s for s in grp.index[grp == g] if (m, s) not in held_b]
            if len(ss) < 3:
                continue
            y = [(m, s) in newh for s in ss]
            a = auc(C.loc[m, ss].values, y)
            if not np.isnan(a):
                aucs.append(a)
        if aucs:
            per_inv.append(np.mean(aucs))
    return float(np.mean(per_inv)), len(per_inv), len(cands)


def public(h, D):
    """Holdings rows public strictly before decision date D."""
    return h[h.filing_date < D]
