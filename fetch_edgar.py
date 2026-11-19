"""Step 1 of building external.parquet: download dated public filings from SEC EDGAR.

For every stock in companies.csv we pick ONE document, chosen only from filings dated
before a cutoff, and record its SIC code from that filing's own SEC header:

* established stocks: the latest annual report (10-K / 20-F / 40-F) filed before 2025-07-01
  (earlier than both the 2026-03-02 holdings cutoff and every backtest decision date);
* 2025H2 and 2026 listings: the latest registration statement (S-1, F-1, S-11, Form 10 and
  their amendments) filed strictly before the listing date.

Raw responses are cached under cache/ (not part of the submission).
Run once:  python fetch_edgar.py
SEC fair-access rules: a User-Agent with name and email, fewer than 10 requests per second.
"""
import gzip
import json
import os
import re
import sys
import time

import pandas as pd
import requests

UA = "Lakshit Tyagi lakshit2005tyagi@gmail.com"
CACHE = "cache"
EST_CUTOFF = "2025-07-01"
ANNUAL = ["10-K", "20-F", "40-F"]
REG = ["S-1", "S-1/A", "F-1", "F-1/A", "S-11", "S-11/A", "10-12B", "10-12B/A", "10-12G", "10-12G/A"]

sess = requests.Session()
sess.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
_last = [0.0]


def get(url, binary=False):
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[1])[:200] + ".gz")
    if os.path.exists(key):
        with gzip.open(key, "rb") as fh:
            data = fh.read()
    else:
        wait = 0.25 - (time.time() - _last[0])  # <= 4 requests per second
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            _last[0] = time.time()
            r = sess.get(url, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (403, 404):
                raise RuntimeError(f"{r.status_code} {url}")
            time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"failed {url}")
        data = r.content
        os.makedirs(CACHE, exist_ok=True)
        with gzip.open(key, "wb") as fh:
            fh.write(data)
    return data if binary else data.decode("utf-8", errors="replace")


def all_filings(cik, window=("2024-01-01", "2026-07-01")):
    """Filing list; older paginated history is fetched only for pages overlapping `window`."""
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        if extra["filingTo"] >= window[0] and extra["filingFrom"] <= window[1]:
            frames.append(pd.DataFrame(json.loads(get("https://data.sec.gov/submissions/" + extra["name"]))))
    fl = pd.concat(frames, ignore_index=True)
    return sub, fl[["accessionNumber", "filingDate", "form", "primaryDocument"]]


def header_sic(cik, acc):
    """SIC code as stated in the SEC header of this specific filing."""
    nod = acc.replace("-", "")
    txt = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nod}/{acc}-index-headers.html")
    mt = re.search(r"STANDARD INDUSTRIAL CLASSIFICATION:\s*([^\[\n<]*)\[(\d{4})\]", txt)
    return (mt.group(2), mt.group(1).strip()) if mt else (None, None)


def main():
    c = pd.read_csv("data/companies.csv", dtype={"sec_cik": int})
    rows = []
    for r in c.itertuples():
        try:
            sub, fl = all_filings(r.sec_cik)
        except Exception as e:  # noqa: BLE001
            print("!!", r.stock_id, r.ticker, e, flush=True)
            rows.append({"stock_id": r.stock_id, "error": str(e)})
            continue
        if r.status == "established":
            cand = fl[fl.form.isin(ANNUAL) & (fl.filingDate < EST_CUTOFF)]
        else:
            cand = fl[fl.form.isin(REG) & (fl.filingDate < r.listing_date)]
        cand = cand.sort_values("filingDate", ascending=False)
        if cand.empty:
            print("!! no candidate filing", r.stock_id, r.ticker, r.status, sorted(fl.form.unique())[:40], flush=True)
            rows.append({"stock_id": r.stock_id, "error": "no candidate", "sub_sic": sub.get("sic")})
            continue
        # download up to 4 most recent candidates (latest amendment first), keep the largest-text one later
        docs = []
        for f in cand.head(4).itertuples():
            url = f"https://www.sec.gov/Archives/edgar/data/{int(r.sec_cik)}/{f.accessionNumber.replace('-', '')}/{f.primaryDocument}"
            try:
                html = get(url)
            except Exception as e:  # noqa: BLE001
                print("   doc fail", r.ticker, url, e, flush=True)
                continue
            docs.append((f.accessionNumber, f.filingDate, f.form, url, len(html)))
            if len(html) > 400_000:  # substantive document found
                break
        best = max(docs, key=lambda d: d[4]) if docs else None
        if best is None:
            rows.append({"stock_id": r.stock_id, "error": "no doc"})
            continue
        sic, sic_desc = header_sic(r.sec_cik, best[0])
        rows.append({"stock_id": r.stock_id, "ticker": r.ticker, "accession": best[0], "filing_date": best[1],
                     "form": best[2], "url": best[3], "html_bytes": best[4], "sic": sic, "sic_desc": sic_desc,
                     "error": None})
        print(r.stock_id, r.ticker, r.status, best[2], best[1], best[4], sic, sic_desc, flush=True)
    pd.DataFrame(rows).to_csv("cache/filing_index.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
