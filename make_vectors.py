"""make_vectors.py -- builds stocks.csv and managers.csv (16-d vectors, scored by cosine).

Runs offline from the supplied files in data/ and external.parquet:
    python make_vectors.py            # writes stocks.csv and managers.csv next to this file

Decision date: 2026-03-02.  Only holdings rows with filing_date < 2026-03-02 are used.  External
data (external.parquet) holds, per stock, a business description taken from one SEC filing dated
before its own cutoff (annual report filed before 2025-07-01 for established stocks; registration
statement filed before the listing date for listings), that filing's SIC code, and embeddings of the
description from approved local models (all-MiniLM-L6-v2, bge-base-en-v1.5 at the pinned revisions).

Vector layout (same for investors u and stocks v):
    d1        IPO axis      u: fitted log-odds that the investor buys a new listing (minus a centre)
                            v: constant for every listing, 0 for established stocks
    d2..d9    text taste    v: stock's description embedding, PCA to 8 dims, unit length
                            u: (shrunk) average of the text vectors of the stocks it holds
    d10..d16  holdings CF   truncated SVD of the investor x stock holdings matrix (all public
                            quarters, weight 0.5^k for the k-th quarter before the investor's latest
                            report); a listing with no holdings gets its CF part by ridge regression
                            from its description embedding.
All randomness is absent (closed-form SVD/ridge, convex logistic fit run to tight tolerance), and
SVD signs are fixed, so the output is reproducible.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXTERNAL = os.path.join(HERE, "external.parquet")
CUTOFF = "2026-03-02"
K = 16
QUARTERS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
EMB_MODELS = ["minilm", "bge"]

DEFAULT = dict(
    k_text=4,          # dims of the text-taste block
    w_ipo=1.0,         # weight of the IPO-appetite axis
    w_text=1.0,        # weight of the text-taste block
    w_cf=1.0,          # weight of the collaborative-filtering block
    decay=0.5,         # weight of each older quarter relative to the next newer one
    svd_alpha=1.0,     # investor factors = U * s^alpha, stock factors = V * s^(1-alpha)
    ipo_mode="logit",  # count: log((n + kappa)/kappa); logit: fitted log-odds (see ipo_logit)
    ipo_C=1.0,         # inverse L2 strength of the logistic fit
    ipo_scale=1.0,
    ipo_kappa=1.0,
    ipo_center=0.0,    # subtracted from the investor IPO value
    listing_flag=1.0,  # IPO-axis value of every listing (established stocks get 0)
    est_taste=0.2,     # weight of established holdings (vs listings) in the investor taste vector
    taste_shrink=2.0,  # taste = sum(w * z) / (sum(w) + taste_shrink)
    cold_cf="ridge",   # how a cold listing gets its CF block: ridge | knn | zero
    ridge_lam=1.0,
    knn_k=10,
    cold_cf_scale=1.0,
)

# chosen on the time-honest 2025 backtests (see design.md)
FINAL = dict(decay=0.5, ipo_center=-3.0, k_text=8, cold_cf="ridge", ridge_lam=0.3, w_ipo=0.5,
             taste_shrink=0.5)


def svd_fixed(A):
    """SVD with a deterministic sign: the largest-magnitude entry of each right vector is positive."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    sg = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
    sg[sg == 0] = 1
    return U * sg, s, Vt * sg[:, None]


def unit_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return np.where(n > 1e-12, A / np.where(n > 1e-12, n, 1), 0.0)


def ipo_logit(f, c, m, mids, Xq, known_lst, D, p):
    """Investor log-odds of holding a new listing at its first quarter-end.

    Logistic regression over (investor, past listing) pairs whose outcome is public at D: label =
    investor reported the listing at the listing's first quarter-end, among investors that filed
    for that quarter.  Features: log(1 + listings held), any listing held, log(1 + established
    stocks held) -- the listing being predicted is left out of its own count -- and, once every
    Dec-2025 report is public (D > 2026-02-27), the Dec-2025 totals in manager_info.
    """
    from sklearn.linear_model import LogisticRegression

    X = Xq.max(0)                                    # held in any public quarter
    n_lst = X[:, known_lst].sum(1)
    est = (c.status == "established").to_numpy()
    log_est = np.log1p(X[:, est].sum(1))
    if D > "2026-02-27":
        mi = m.set_index("manager_id").loc[mids]
        tot = mi.total_positions_dec2025.to_numpy(float).clip(1)
        extra = np.column_stack([np.log(tot), mi.positions_in_197_stocks_dec2025.to_numpy(float) / tot,
                                 np.log(mi.total_value_usd_dec2025.to_numpy(float).clip(1))])
    else:
        extra = np.zeros((len(mids), 0))

    def feats(n):
        return np.column_stack([np.log1p(n), (n > 0) * 1.0, log_est, extra])

    fp = f[f.filing_date < D]
    filed_q = fp.groupby("quarter_end").manager_id.apply(set).to_dict()
    lst_date = c.listing_date.to_numpy()
    rows, ys = [], []
    for j in np.where(known_lst)[0]:
        qe = next(q for q in QUARTERS if q >= lst_date[j])
        filers = np.array([mid in filed_q.get(qe, ()) for mid in mids])
        if filers.sum() == 0:
            continue
        F = feats(n_lst - X[:, j])                   # leave this listing out of its own feature
        rows.append(F[filers])
        ys.append(Xq[QUARTERS.index(qe), :, j][filers])
    if not rows or np.concatenate(ys).sum() < 3:
        return np.log1p(n_lst)                      # not enough history: fall back to the count
    Xf = np.vstack(rows)
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=p["ipo_C"], tol=1e-10, max_iter=100000)
    clf.fit((Xf - mu) / sd, np.concatenate(ys))
    return clf.decision_function((feats(n_lst) - mu) / sd) * p["ipo_scale"]


def build(h, f, c, m, emb, D, cold_extra=(), params=None):
    """(U, V): DataFrames of 200 investors x 16 and 197 stocks x 16 at decision date D.

    h, f, c, m: holdings, filings, companies, manager_info; emb: (197 x p) text embeddings aligned
    with c; only holdings/filings with filing_date < D are used; cold_extra: stock ids whose
    holdings are ignored (treated as not yet listed; used by the backtests).
    """
    p = dict(DEFAULT, **(params or {}))
    mids = m.manager_id.tolist()
    sids = c.stock_id.tolist()
    nM, nS = len(mids), len(sids)
    hp = h[(h.filing_date < D) & ~h.stock_id.isin(set(cold_extra))]
    mi = pd.Index(mids).get_indexer(hp.manager_id)
    si = pd.Index(sids).get_indexer(hp.stock_id)
    qi = pd.Index(QUARTERS).get_indexer(hp.quarter_end)
    Xq = np.zeros((len(QUARTERS), nM, nS))
    Xq[qi, mi, si] = 1.0

    # holdings matrix: each public quarter weighted decay^(age relative to the investor's latest)
    filed = np.zeros((nM, len(QUARTERS)), bool)
    filed[mi, qi] = True
    last = np.where(filed.any(1), len(QUARTERS) - 1 - np.argmax(filed[:, ::-1], 1), -1)
    Y = np.zeros((nM, nS))
    for q in range(len(QUARTERS)):
        w = np.where(q <= last, p["decay"] ** (last - q).clip(0), 0.0)
        Y += w[:, None] * Xq[q]

    is_listing = (c.status != "established").to_numpy()
    known = Y.sum(0) > 0                  # stocks with public holdings at D
    cold = ~known                         # listings with no holdings yet (cold start)

    # ---- block 1: IPO appetite --------------------------------------------------------------
    if p["ipo_mode"] == "count":
        n_ipo = (Y[:, is_listing & known] > 0).sum(1)
        ipo_inv = np.log((n_ipo + p["ipo_kappa"]) / p["ipo_kappa"])
    else:
        ipo_inv = ipo_logit(f, c, m, mids, Xq, is_listing & known, D, p)
    ipo_inv = ipo_inv - p["ipo_center"]
    ipo_stk = np.where(is_listing, p["listing_flag"], 0.0)

    # ---- block 2: text taste ----------------------------------------------------------------
    kt = p["k_text"]
    E = np.asarray(emb, float)
    mu = E[known].mean(0)
    _, _, Wt = svd_fixed(E[known] - mu)   # PCA fitted on stocks with public holdings only
    Z = unit_rows((E - mu) @ Wt[:kt].T) if kt else np.zeros((nS, 0))
    wts = np.where(is_listing, 1.0, p["est_taste"])[None, :] * (Y > 0)
    taste = (wts @ Z) / (wts.sum(1, keepdims=True) + p["taste_shrink"])

    # ---- block 3: collaborative filtering ---------------------------------------------------
    kc = K - 1 - kt
    Uu, s, Vt = svd_fixed(Y[:, known])
    Ucf = Uu[:, :kc] * s[:kc] ** p["svd_alpha"]
    Vk = Vt[:kc].T * s[:kc] ** (1 - p["svd_alpha"])
    Vcf = np.zeros((nS, kc))
    Vcf[known] = Vk
    if cold.any():
        Ek, Ec = E[known] - mu, E[cold] - mu
        if p["cold_cf"] == "ridge":
            B = np.linalg.solve(Ek.T @ Ek + p["ridge_lam"] * np.eye(Ek.shape[1]), Ek.T @ Vk)
            Vcf[cold] = Ec @ B
        elif p["cold_cf"] == "knn":
            S = unit_rows(Ec) @ unit_rows(Ek).T
            idx = np.argsort(-S, 1)[:, : p["knn_k"]]
            for r, ci in enumerate(np.where(cold)[0]):
                w = S[r, idx[r]].clip(0)
                Vcf[ci] = w @ Vk[idx[r]] / max(w.sum(), 1e-9)
        Vcf[cold] *= p["cold_cf_scale"]
    Ucf, Vcf = unit_rows(Ucf), Vcf / np.median(np.linalg.norm(Vk, axis=1))

    U = np.hstack([p["w_ipo"] * ipo_inv[:, None], p["w_text"] * taste, p["w_cf"] * Ucf])
    V = np.hstack([p["w_ipo"] * ipo_stk[:, None], p["w_text"] * Z, p["w_cf"] * Vcf])
    U[np.abs(U).sum(1) < 1e-12, -1] = 1e-6   # cosine is undefined for a zero vector
    V[np.abs(V).sum(1) < 1e-12, -1] = 1e-6
    cols = [f"d{j}" for j in range(1, K + 1)]
    return pd.DataFrame(U, index=mids, columns=cols), pd.DataFrame(V, index=sids, columns=cols)


def load_data():
    h = pd.read_csv(os.path.join(DATA, "holdings.csv"))
    f = pd.read_csv(os.path.join(DATA, "filings.csv"))
    c = pd.read_csv(os.path.join(DATA, "companies.csv"))
    m = pd.read_csv(os.path.join(DATA, "manager_info.csv"))
    return h, f, c, m


def load_embeddings(c, models=EMB_MODELS):
    ext = pd.read_parquet(EXTERNAL).set_index("stock_id").loc[c.stock_id]
    return np.hstack([np.vstack(ext[f"emb_{k}"].to_numpy()).astype(float) for k in models])


def main():
    h, f, c, m = load_data()
    U, V = build(h, f, c, m, load_embeddings(c), CUTOFF, params=FINAL)
    V.rename_axis("stock_id").to_csv(os.path.join(HERE, "stocks.csv"), float_format="%.10f")
    U.rename_axis("manager_id").to_csv(os.path.join(HERE, "managers.csv"), float_format="%.10f")
    print("wrote stocks.csv", V.shape, "and managers.csv", U.shape)


if __name__ == "__main__":
    main()
