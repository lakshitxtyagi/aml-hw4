"""Step 4: pack everything brought in from outside into external.parquet (one row per stock).

Columns: provenance of the filing used (form, filing date, accession, URL), its SEC-header SIC code,
the extracted business description, and the text embeddings from each approved model
(model outputs, so make_vectors.py runs offline without loading any language model).
Pipeline:  fetch_edgar.py -> extract_text.py -> embed_text.py -> build_external.py
"""
import html
import os

import numpy as np
import pandas as pd

from embed_text import MODELS

INCLUDE = ["minilm", "bge"]  # embeddings used by make_vectors.py


def main():
    c = pd.read_csv("data/companies.csv")
    idx = pd.read_csv("cache/filing_index.csv", dtype={"sic": str})
    txt = pd.read_parquet("cache/texts.parquet")
    df = c[["stock_id"]].merge(idx.drop(columns=["ticker"], errors="ignore"), on="stock_id", how="left")
    df = df.merge(txt, on="stock_id", how="left")
    df["sic_desc"] = df.sic_desc.map(lambda x: html.unescape(html.unescape(x)) if isinstance(x, str) else x)
    assert (df.stock_id.values == c.stock_id.values).all()
    for name in [n for n in MODELS if n in INCLUDE]:
        path = f"cache/emb_{name}.npy"
        if os.path.exists(path):
            E = np.load(path)
            df[f"emb_{name}"] = [row.astype(np.float32) for row in E]
    keep = ["stock_id", "form", "filing_date", "accession", "url", "sic", "sic_desc", "text", "n_words", "method"]
    keep += [k for k in df.columns if k.startswith("emb_")]
    df[keep].to_parquet("external.parquet", index=False)
    print("external.parquet", df.shape, round(os.path.getsize("external.parquet") / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
