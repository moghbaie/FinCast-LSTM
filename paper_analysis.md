# Paper analysis — Fischer & Krauss (2018)

**Reference.** Fischer, T., Krauss, C. (2018). Deep learning with long short-term memory networks
for financial market predictions. *European Journal of Operational Research* 270(2), 654–669.
DOI 10.1016/j.ejor.2017.11.054.

**Source actually read.** FAU Discussion Papers in Economics No. 11/2017, version dated
10 May 2017, EconStor handle 10419/157808 (34 PDF pages, retrieved 2026-11-13).
**Page references (p. N) are the printed page numbers of that manuscript** (PDF page = N + 2;
the abstract page is unnumbered and cited as p. 1). The EJOR version was not read: its pagination
and possibly some numbers differ (see open item O-1).

Labels used throughout:

- **[P]** stated in the paper (with page).
- **[A]** our implementation assumption where the paper is silent or ambiguous.
- **[M]** proposed modification (used only outside the faithful Phase A reproduction, or as a
  labelled sensitivity analysis).

---

## 1. Hypotheses, objectives, contributions

| # | Content | Source |
|---|---|---|
| 1 | LSTM networks, suited to sequence learning, have not been applied to a large, liquid, survivor-bias-free stock universe; the paper applies them to all S&P 500 constituents 1992–2015. | [P] p. 2–3 |
| 2 | Implicit hypothesis H1: LSTM beats memory-free classifiers (RAF, DNN, LOG) in accuracy and returns. | [P] p. 1, 3 |
| 3 | H2: the returns are economically and statistically significant after 5 bps/half-turn costs. | [P] p. 13, 16–18 |
| 4 | Objective 2: open the black box: what do top/flop stocks have in common? (high volatility, below-mean momentum, short-term reversal). | [P] p. 3, 21–24 |
| 5 | Objective 3: a simple rules-based short-term reversal strategy that explains part of the LSTM returns (R² 52.1 % long, 53.6 % short; remaining alpha ≈ 31 bps/day before costs). | [P] p. 3, 24–25 |
| 6 | Builds on Krauss, Do & Huck (2017, EJOR 259(2)) "on the same data sample for the sake of comparability". | [P] p. 2 |

## 2. Data

| Item | Content | Source |
|---|---|---|
| Provider | Thomson Reuters (proprietary). | [P] p. 4 |
| Universe | S&P 500 constituents. Month-end constituent lists Dec 1989 – Sep 2015, consolidated into a binary matrix "indicating whether the stock is an index constituent in the subsequent month"; this "approximately" reproduces the index at any point in time. | [P] p. 4 |
| Prices | Daily **total return indices** (cum-dividend, adjusted for corporate actions and splits) Jan 1990 – Oct 2015, for every stock ever a constituent in the period. | [P] p. 4 |
| Survivorship control | All stocks ever in the index are downloaded; delisted stocks are traded until their last price. | [P] p. 4, 6 (fn. 3) |
| Membership rule per study period | n_i = stocks that are constituents on the **last day of the training period**; training uses all n_i stocks with whatever history they have; trading uses the same n_i stocks; a stock leaving the index during the trading period is **not** removed; a stock without further prices is traded until its last price. | [P] p. 5–6 (fn. 3) |
| Descriptive stats | Table 1: monthly equal-weighted sector portfolios; "All": 499.7 stocks, mean 1.04 %/month, sd 4.78 %. | [P] p. 5 |
| Sector scheme | Caption says GICS, but the sector names (Oil & Gas, Basic Materials, Consumer Services, …) are ICB/TRBC-style. Inconsistency. | [P] p. 5; noted by us |
| Software | Python 3.5, numpy, pandas, keras on TensorFlow (LSTM), scikit-learn (RAF, LOG), H2O (DNN), R PerformanceAnalytics (evaluation). LSTM on NVIDIA GPUs. | [P] p. 4 |
| Missing values inside a series | Not specified. | — (O-4) |
| Delisting returns | Not specified (the total return index simply ends). | — (O-5) |

## 3. Study-period design

| Item | Content | Source |
|---|---|---|
| Study period | 1,000 trading days: 750 training + 250 trading. | [P] p. 5 |
| Number | 23 study periods with **non-overlapping trading periods** (training windows overlap; step = 250 days). | [P] p. 5 |
| Out-of-sample span | Dec 1992 – Oct 2015; 23 × 250 = 5,750 trading days. | [P] p. 14–15 |
| Samples | ≈ 380,000 sequences per study period (760 per stock × ≈500 stocks); ≈ 255,000 training, ≈ 125,000 trading. | [P] p. 6–7 (fn. 5) |
| Day count | "days" = trading days (implied by 5,750 days over ≈ 23 years). | [A] |

## 4. Features and targets

| Item | Content | Source |
|---|---|---|
| Return | R^{m,s}_t = P^s_t / P^s_{t−m} − 1 on the total return index. | [P] p. 6, eq. (1) |
| LSTM feature | Single feature: standardized one-day return R̃ = (R − µ_train)/σ_train, with µ, σ "obtained from the training set" to avoid look-ahead. | [P] p. 6, eq. (2), fn. 4 |
| Pooled or per stock? | Notation µ^m_train has no stock index → **pooled over all stocks and training days**. | [A] (O-6) |
| Sequence | 240 consecutive standardized one-day returns {R̃_{t−239}, …, R̃_t}, for each stock and each t ≥ 240 of the study period, built by sorting by stock then date (Fig. 1). | [P] p. 6–7 |
| Input shape | (240 timesteps, 1 feature). | [P] p. 11 |
| Target | Binary: Y^s_{t+1} = 1 if R^{1,s}_{t+1} ≥ cross-sectional median of all stocks' returns on t+1, else 0. Follows Takeuchi & Lee (2013). | [P] p. 7 |
| Cross-section for the median | "all stocks s in period t+1", i.e. the n_i stocks of the study period. | [P] p. 7; [A] that stocks with no price on t+1 are excluded |
| Benchmark features (RAF, DNN, LOG) | 31 multi-period returns R^s_{t,m}, m ∈ {1,…,20} ∪ {40, 60, …, 240}. Standardized (eq. 2) for LOG only; DNN standardized internally by H2O; RAF unstandardized. | [P] p. 11 (fn. 7) |

## 5. LSTM

| Item | Content | Source |
|---|---|---|
| Cell | Standard LSTM with forget, input, output gates, eqs. (3)–(8) (no peepholes). | [P] p. 8–9 |
| Topology | Input 1 feature × 240 steps → **1 LSTM layer, h = 25** → Dense(2, softmax). | [P] p. 11 |
| Parameters | LSTM 4(h(i+1)+h²) = 2,700; with dense 2,752 total. Text says "approximately 93 parameters per observation"; it is ≈ 93 **observations per parameter** (255,000/2,752). | [P] p. 10–11, eq. (9); wording error noted by us |
| Dropout | 0.16, "within the recurrent layer … both at the input gates and the recurrent connections", following Gal & Ghahramani (2016), i.e. keras `dropout=0.16, recurrent_dropout=0.16` (variational: one mask per sequence). Chosen low because "in initial experiments" the LSTM was sensitive to higher values. | [P] p. 10–11 (fn. 6); keras mapping [A] |
| Loss | Cross-entropy. | [P] p. 10 |
| Optimizer | RMSprop. **Learning rate not stated** → keras default 0.001, ρ = 0.9. | [P] p. 10; lr [A] (O-2) |
| Batch size | **Not stated** (keras default would be 32). | (O-2) |
| Early stopping | 80 % train / 20 % validation of the training samples; max 1,000 epochs; patience 10 on validation loss; restore the weights with lowest validation loss. | [P] p. 10–11 |
| How the 20 % is chosen | **Not stated.** Text cites Granger's "post-sample" data (suggests chronological). If keras `validation_split=0.2` was used on data sorted by stock then date, the validation set would be the last ≈ 20 % of **stocks**, not of dates. | (O-3) |
| Initialisation, seeds, shuffling | Not stated → keras defaults (Glorot uniform input weights, orthogonal recurrent weights, forget-gate bias 1, shuffle each epoch). | [A] |
| Ensembling / seeds | Single model per study period; number of seeds not stated. | [P] implicit; (O-7) |

## 6. Benchmarks

| Model | Specification | Source |
|---|---|---|
| RAF | scikit-learn; B = 1,000 trees, max depth J = 20, max_features = √p (default). | [P] p. 11–12 |
| DNN | H2O feed-forward 31-31-10-5-2, maxout (2 channels), softmax output, dropout 0.5, L1 = 1e-5; "see Krauss et al. (2017) for further details" (epochs, input dropout, ADADELTA defaults are in that paper / H2O defaults). | [P] p. 12; rest (O-8) |
| LOG | scikit-learn logistic regression, L2 penalty chosen among 100 values log-spaced 1e-4 … 1e4 by 5-fold CV on the training set, L-BFGS, max_iter 100. | [P] p. 12 |
| Is the grid C or λ? | Symmetric grid, so immaterial. CV folds not stated as time-ordered (sklearn default: stratified, unshuffled K-fold). | [A] |
| Monkey | 100,000 random long-short portfolios (10 + 10 stocks per day). | [P] p. 13, 16 |

## 7. Trading rule

| Item | Content | Source |
|---|---|---|
| Signal | P̂(Y^s_{t+1} = 1 | info up to t) for every stock. | [P] p. 12 |
| Portfolio | Rank descending; long top k, short flop k, equal weight, held one day; k ∈ {10, 50, 100, 150, 200}; focus k = 10. | [P] p. 12–13 |
| Execution | Close-to-close: signal on close t, return R_{t+1} from close t to close t+1, i.e. trade at the same close used for the signal. | [P] implied by eq. (1) and p. 12; flagged as optimistic by us |
| Portfolio return | Mean return = long-leg return + short-leg return (Table 3: 0.0029 + 0.0017 = 0.0046), i.e. return per $1 long **and** $1 short (2 × gross exposure per unit). | [P] p. 17; interpretation [A] |
| Costs | 5 bps per half-turn (Avellaneda & Lee, 2010). Table 3 net = gross − 0.0020/day = 4 half-turns × 5 bps → assumes **full turnover of both legs every day**, no borrow cost, no market impact. | [P] p. 13, 17; inference [A] |

## 8. Evaluation and reported results

Statistical tests [P]: Diebold–Mariano on the 0/1 classification-error vectors of the k = 10 stocks
(one-sided, Bonferroni for 3 comparisons) p. 14–15; Pesaran–Timmermann p. 15; binomial test of
54.3 % accuracy on 115,000 picks (p = 2.77e-187) p. 15–16; Newey–West (1 lag) t-statistics p. 17.
The DM test pools 115,000 picks and ignores cross-sectional correlation within a day (our note).

**Table 3 (k = 10, Dec 1992 – Oct 2015, daily unless noted)** [P] p. 17:

| Metric | LSTM | RAF | DNN | LOG | LSTM net | RAF net | DNN net | LOG net | MKT |
|---|---|---|---|---|---|---|---|---|---|
| Mean return (long) | 0.0029 | 0.0030 | 0.0022 | 0.0021 | 0.0019 | 0.0020 | 0.0012 | 0.0011 | – |
| Mean return (short) | 0.0017 | 0.0012 | 0.0010 | 0.0005 | 0.0007 | 0.0002 | 0.0000 | −0.0005 | – |
| Mean return | 0.0046 | 0.0043 | 0.0032 | 0.0026 | 0.0026 | 0.0023 | 0.0012 | 0.0006 | 0.0004 |
| NW t-stat | 16.93 | 14.11 | 8.95 | 7.00 | 9.58 | 7.52 | 3.37 | 1.67 | 2.83 |
| Share > 0 | 0.6148 | 0.6078 | 0.5616 | 0.5584 | 0.5574 | 0.5424 | 0.5146 | 0.5070 | 0.5426 |
| Std dev (daily) | 0.0209 | 0.0215 | 0.0262 | 0.0269 | same | same | same | same | 0.0117 |
| VaR 1 % | −0.0525 | −0.0475 | −0.0676 | −0.0746 | −0.0545 | −0.0495 | −0.0696 | −0.0766 | −0.0320 |
| Max drawdown | 0.4660 | 0.3187 | 0.5594 | 0.5595 | 0.5233 | 0.7334 | 0.9162 | 0.9884 | 0.5467 |
| Return p.a. | 2.0127 | 1.7749 | 1.0610 | 0.7721 | 0.8229 | 0.6787 | 0.2460 | 0.0711 | 0.0925 |
| Std dev p.a. | 0.3323 | 0.3408 | 0.4152 | 0.4266 | same | same | same | same | 0.1852 |
| Sharpe p.a. | 5.8261 | 5.0001 | 2.4288 | 1.7038 | 2.3365 | 1.8657 | 0.5159 | 0.1024 | 0.3486 |
| Sortino p.a. | 10.02 | 9.56 | 4.20 | 2.96 | 3.85 | 3.41 | 0.92 | 0.26 | 0.71 |

Other reported results [P]:

- k = 10 accuracy of LSTM: **54.3 %** (p. 15). Fig. 3 (p. 14) gives returns, sd, Sharpe and accuracy
  for all k; the PDF text layer of Fig. 3 is scrambled, so values for k > 10 and benchmark
  accuracies will be digitized from the rendered figure in M5 (O-9).
- DM p-values (H0: row model not more accurate than column model), first row LSTM vs RAF/DNN/LOG:
  0.0143 / 0.0037 / 0.0000; RAF vs DNN 0.3180; PT p-values all 0.0000 (Table 2, p. 15).
- Monkey portfolios: mean ≈ 0, best of 100,000 ≈ 0.05 %/day (p. 16).
- Sub-periods (Fig. 5, p. 18–20, after costs): 1993–2000 strong (cumulative > 11 USD LSTM,
  > 8 USD RAF on 1 USD/day); 2001–2009 moderation (≈ 4 USD LSTM, 5.6 USD RAF, RAF strong in
  2008–09); 2010–2015 deterioration: RAF loses > 1 USD, LSTM roughly flat after costs.
- Patterns (Figs. 7–8, p. 22–24): top/flop stocks have high volatility and beta, below-mean
  12-month momentum, extreme last-5/20-day returns (top: recent losers, flop: recent winners),
  positive short-term-reversal factor loading, negative momentum loading.
- Simplified strategy (p. 24–25, Table 4): long 10 stocks with most negative 5-day return, short
  10 with most positive: 0.23 %/day gross, ≈ 0.03 %/day net (p. 25 writes "bps"; the
  abstract/conclusion p. 1, 26 say percent: units typo). Regression R² 0.5211 (long) /
  0.5356 (short); intercepts 0.0019 / 0.0012.

## 9. Limitations stated or evident

| Limitation | Source |
|---|---|
| Profitability decays over time; near zero after costs 2010–2015. | [P] p. 18–20 |
| Limits to arbitrage, short-selling costs/bans, liquidity not modelled. | [P] p. 20 |
| Membership "approximately" reproduced from month-end lists. | [P] p. 4 |
| Dropout chosen from "initial experiments": the data used for them is not stated, so possible model-selection look-ahead. | [P] p. 11 fn. 6; concern ours |
| Last training sequences' labels (t = 750 → return on day 751) fall in the trading period: a 1-day label overlap between training and trading. | inferred from p. 5–7; ours |
| Same-close execution; full-turnover cost assumption; no borrow fees. | inferred; ours |
| Single run per period, no seed variance reported. | inferred; ours |

## 10. Open / ambiguous items (feed into plan.md decisions)

| ID | Item | Default assumption for Phase A |
|---|---|---|
| O-1 | EJOR published version not read (paywalled); numbers/pages may differ. | Use discussion paper; cross-check if you can provide the EJOR PDF. |
| O-2 | Batch size, learning rate. | RMSprop lr 1e-3, ρ 0.9 (keras default); batch size per decision D3. |
| O-3 | Composition of the 20 % validation set. | Chronological last 20 % of training dates (primary); keras-style last 20 % of stock-sorted rows (sensitivity). |
| O-4 | Missing prices inside a series. | No interpolation; a sequence is valid only if its 240 returns + target are present. |
| O-5 | Delisting returns. | Total return index ends; position closed at last price (no delisting return); sensitivity with −30 % for performance-related delistings if CRSP codes are available. |
| O-6 | Pooled vs per-stock standardization. | Pooled over the training window. |
| O-7 | Seeds / run-to-run variance. | 5 seeds per period; report each seed and the seed mean. |
| O-8 | DNN details beyond p. 12. | Taken from Krauss, Do & Huck (2017); to be extracted in M2. |
| O-9 | Fig. 3 values for k > 10 and benchmark accuracies. | Digitize from the rendered figure. |
| O-10 | Cross-section definition for the median when some stocks lack a price on t+1. | Median over stocks with a valid return on t+1. |
| O-11 | Exact trading-day calendar of Thomson Reuters series (holidays, stale prices). | NYSE calendar; stale price runs flagged in data audit. |
