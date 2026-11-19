# HW4: Who Should Be in the Room (solution)

The solution gives each of 197 stocks and 200 institutional investors a 16-d vector in one shared space, and scores a match by cosine. See `design.md` for the method and `answers.md` for the six questions.

## Reproduce the tables (offline)

Put the four supplied files in `data/`, which is not committed, then run:

```bash
pip install numpy pandas scikit-learn pyarrow
python make_vectors.py                          # writes stocks.csv and managers.csv
python check_tables.py stocks.csv managers.csv  # format check (supplied by the course)
```

`make_vectors.py` reads only `data/*.csv` and `external.parquet`. It uses no randomness: SVD, PCA and ridge are closed-form, the logistic fit is convex and solved to tolerance 1e-10, and SVD signs are fixed. It reproduces the tables exactly.

## What is in `external.parquet`

There is one row per stock, and every row comes from one SEC EDGAR filing dated before its cutoff:

- **Established stocks:** the latest 10-K or 20-F filed before 2025-07-01.
- **Listings:** the latest S-1/A or F-1/A filed before the listing date.

Each row records the form, filing date, accession number and URL. It also holds the SIC code from that filing's own SEC header, the extracted business description (up to 2,500 words), and that description's embeddings from two approved models run locally at their pinned revisions: `all-MiniLM-L6-v2` (`emb_minilm`) and `bge-base-en-v1.5` (`emb_bge`). The models only ever embedded the extracted filing text. They were never asked anything about a company.

## How `external.parquet` was built (needs network)

```bash
python fetch_edgar.py      # SEC EDGAR, under 10 req/s, User-Agent with name and email; raw files cached in cache/
python extract_text.py     # business section / prospectus summary
python embed_text.py minilm bge
python build_external.py
```

## Design tools, not needed for the tables

- `hw4eval.py` re-implements the three tests.
- `backtest.py` runs the time-honest 2025 backtests and the final-date cross-validation.
- `tune.py` runs the coordinate search used to choose the parameters.
