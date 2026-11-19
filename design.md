# Design: one 16-d space for investors and stocks

## Data and cutoffs
- **Holdings:** only rows filed before 2026-03-02, which covers all four 2025 quarters (every December report was filed by 27 February). Quarters are weighted 0.5 per quarter back from each investor's own latest report.
- **manager_info:** the December totals come from those same reports, so they are public at the cutoff.
- **External data: one SEC filing per company.**
  - Established stocks: the latest 10-K or 20-F filed before 2025-07-01. This precedes every backtest date too.
  - Listings: the latest S-1/A or F-1/A filed before the listing date.
  - From each filing I keep its business section or prospectus summary (up to 2,500 words) and the SIC code from its SEC header.
- **Embeddings:** MiniLM-L6-v2 and bge-base-v1.5 at their pinned revisions, run locally on filing text only. Each averages twelve 200-word chunks; the two are concatenated.

## Vector layout

| dims | investor | stock |
|---|---|---|
| d1: IPO axis | 0.5(ℓᵢ+3) | 0.5 for listings, 0 otherwise |
| d2–d9: text taste | shrunk mean of held stocks' z | z: 8-d PCA of embedding, unit length |
| d10–d16: CF | rank-7 SVD factor, unit length | SVD factor; cold listings: ridge from text |

- **IPO axis.** ℓᵢ is the log-odds that investor i holds a new listing at its first quarter-end. It comes from an L2 logistic regression over (investor, 2025H2 listing) pairs among that quarter's filers. Features: log(1 + listings held, excluding the one being predicted), any listing held, log(1 + established stocks held), and the December totals (positions, value, share in the 197).
- **Text taste.** Σwz / (Σw + 0.5), with weight 1 for listings and 0.2 for established stocks. The PCA is fitted only on stocks with holdings.
- **CF block.** SVD of the weighted 0/1 matrix over the 153 stocks with holdings; stock factors are divided by their median norm.
- **Cold start.** A 2026 listing's CF factors come from ridge regression (λ = 0.3) of embeddings onto SVD factors. It is trained on those 153 stocks, including the 2025H2 listings, so it learns how IPO-buyer patterns read in prospectus language.

**Cosine-compatibility.** Test 1 ignores stock norms; Test 3 ignores investor norms. The IPO axis's listing-wide term largely cancels in Test 2, where matching comes from the taste and CF blocks.

## Testing (time-honest)
- **T1.** The 22 listings from 2025H2 with at least 3 first-quarter holders, each scored by a model rebuilt at its listing date from filings public then.
- **T2.** Their 230 usable pairs, built at the later listing date with both listings' holdings hidden.
- **T3.** From 2025-08-15, June holdings predict new December positions (two quarters ahead, as in the real test). **T3b** runs from 2025-11-15, September to December, and includes Q3 listings.
- **Regimes.** Only Q4 listings had IPO history to learn from, as the real test will.
- **CV diagnostic.** 5-fold cross-validation at the real cutoff, hiding folds of 2025H2 listings. Not time-honest; used only as a full-information check.

| configuration | T1 | T1 Q3/Q4 | T2 | T3 | T3b | CV T1 | CV T2 |
|---|---|---|---|---|---|---|---|
| CF only | .546 | .58/.47 | .500 | .629 | .644 | .471 | .500 |
| + IPO axis | .543 | .46/.77 | .500 | .629 | .651 | .874 | .500 |
| + text→CF ridge | .747 | .70/.89 | .674 | .629 | .651 | .895 | .688 |
| **+ 8-d taste (final)** | **.779** | .75/.87 | **.747** | .632 | .645 | .901 | .759 |

## Reaching 16 dimensions
- **IPO axis: one dimension**, because appetite is essentially a single number.
- **CF:** in CF-only tests, T3 rose with rank (rank 4 → 16 took it from 0.58 to 0.63).
- **Text:** helps T1 and T2 up to 8 dimensions (4-d: T1 .765, T2 .730). At 12, only 3 CF dimensions remain and every test drops (T1 .749, T3 .609). So the split is 1 + 8 + 7.
- **Other parameters** (ridge λ, shrinkage, IPO weight and centre) came from a coordinate search on T1 + T2 + mean(T3, T3b). Quarter decay of 0.5 vs 1.0 is within noise; I took the better time-honest score.
- **Rejected:**
  - ALS, value-weighted holdings, kNN cold start: no better than binary SVD and ridge.
  - SIC one-hots: hurt, since codes like 7389 are too broad; they are kept only as provenance.
  - A single embedding model: slightly worse (T2: MiniLM .746, bge .723).

## Limitations
- **Noise:** 22 test listings give roughly ±0.02, and choosing among ~40 configurations flatters the backtest.
- **13F snapshots** miss positions traded within a quarter.
- **Unusual 2026 listings** (SpaceX, small miners) may draw unfamiliar holders.
