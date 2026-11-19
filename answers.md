# HW4 answers

**1. AUC and ranks.** AUC is the share of (holder, non-holder) pairs in which the holder's cosine is higher. It depends only on those pairwise orderings, so any strictly increasing transform of the cosines leaves it unchanged. A non-rank alternative is the Brier score of cosines mapped to probabilities by a fitted logistic calibration. That would show how likely a called investor is to buy, and so how many investors to call, which AUC cannot. It misleads because holders are rare (about 3% per listing), so predicting the base rate scores well, and it depends on the calibration fit.

**2. Objective.** There is no single objective; each block solves its own:
- **CF block:** a rank-7 SVD, min ‖Y − UVᵀ‖²_F, Y = quarterly 0/1 holdings weighted 0.5 per quarter back.
- **IPO axis:** an L2-penalised logistic regression, min Σ −log p(held | investor features) + ‖w‖²/2C, over (investor, past listing) pairs.
- **Cold listings:** ridge regression, min ‖E B − V‖² + 0.3‖B‖², mapping text embeddings to CF factors.
- **Text taste:** 8-d PCA of the embeddings.

A missing holding is a 0 in Y, a weak negative. Sixteen dimensions lose individual positions, sizes and fine sector detail.

**3. Identical stock vectors.** Yes, it could do well on Test 1. With a single stock vector v, every listing ranks investors by the same cos(uᵢ, v), which amounts to ranking investors by IPO appetite. That appetite persists strongly: our IPO axis alone scores about 0.88 cross-validated AUC. On Test 2 it would score exactly 0.5, because cos(i,A) + cos(j,B) = cos(i,B) + cos(j,A) for every comparison, so all of them tie. A high Test 1 score can therefore reflect who buys IPOs in general rather than which company suits which investor. Test 2 is needed to show real matching.

**4. Other uses.**
- **Follow-on offerings and block trades.** The same ranking could pick which investors to call when a large holder sells shares. Holdings-based taste plus IPO appetite should predict who absorbs supply. Before trusting it, I'd want past block-allocation records showing that the predicted buyers actually took stock.
- **Investor-relations targeting.** For a listed company, the vectors give investors whose holdings resemble its shareholder base but who don't yet own it (a Test 3-style ranking). Before trusting it, I'd want meeting-to-position conversion rates, ideally from a randomised pilot.

**5. Holdings versus interest.**
- **Quarter-end snapshots.** A 13F shows one day per quarter, so investors who flip an IPO before quarter-end look uninterested.
- **Constraints and non-conviction positions.** Oversubscribed IPOs ration allocations, so an interested investor may hold nothing. Conversely, positions held for hedging, arbitrage or index tracking (and shorts, which 13Fs omit) don't express a preference for the company.

A better test would use the bank's book-building records (indications of interest and order sizes) as outcomes. A second option is to count only positions that are kept or increased after the first quarter.

**6. Widespread adoption.** If every bank trains similar tools on the same public 13F data, the rankings converge and the tools stop giving a relative advantage; the edge moves to private data such as order books and roadshow feedback. The predicted IPO buyers would be crowded with calls and gain bargaining power over allocation and pricing, while overlooked investors are called less. That creates a feedback loop: predicted buyers get allocations, appear in 13Fs, and are predicted again, concentrating shareholder bases. Companies might tune prospectus wording, and investors might manage what their disclosures signal.
