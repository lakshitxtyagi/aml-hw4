"""Coordinate search over model.build parameters on the time-honest backtests (design tool)."""
import json, sys
import numpy as np
from backtest import run, fmt, load_emb

def J(r):
    return r["t1"] + r["t2"] + 0.5 * (r["t3"] + r["t3b"])

def search(emb, start, space, rounds=2, log=None):
    best = dict(start); rb = run(emb, best); jb = J(rb)
    print("start", round(jb, 4), fmt(rb), flush=True)
    for _ in range(rounds):
        improved = False
        for k, vals in space.items():
            for v in vals:
                if best.get(k) == v:
                    continue
                cand = dict(best, **{k: v}); r = run(emb, cand); j = J(r)
                if j > jb + 0.002:
                    best, rb, jb, improved = cand, r, j, True
                    print(f"  {k}={v}  J={j:.4f}", fmt(r), flush=True)
        if not improved:
            break
    return best, rb, jb

if __name__ == "__main__":
    emb = load_emb(sys.argv[1])
    start = dict(decay=1.0, ipo_center=-3.0, k_text=4, cold_cf="ridge")
    space = dict(k_text=[3, 4, 6, 8], ridge_lam=[0.1, 0.3, 1.0, 3.0], w_text=[0.5, 1.0, 2.0],
                 w_ipo=[0.5, 1.0, 2.0], est_taste=[0.0, 0.2, 0.5, 1.0], taste_shrink=[0.5, 2.0, 5.0],
                 cold_cf_scale=[0.5, 1.0, 2.0], listing_flag=[0.5, 1.0, 2.0], ipo_center=[-5.0, -3.0, -1.0, 0.0])
    best, r, j = search(emb, start, space)
    print("BEST", sys.argv[1], round(j, 4), json.dumps(best), fmt(r))
