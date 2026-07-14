# KAN Market Regime Strategy

A regime-conditional equity return forecasting system built on Kolmogorov-Arnold Networks (KANs) and Hidden Markov Models. Inspired by the [KASPER paper](https://arxiv.org/abs/2507.18983) (arXiv 2507.18983, 2025).

---

## The core idea

Most ML models applied to markets train only one model and implicitly assume the data-generating process is stationary. But the statistical relationships between features and returns shift meaningfully across market regimes. Equity momentum has different predictive power in a high-volatility drawdown than it does in a calm trending environment.

This project tests a direct response to that: first identify which regime the market is in using a Hidden Markov Model, then route to a specialist Kolmogorov-Arnold Network trained *only* on days that resembled that regime. The KAN architecture is the key choice here — because KANs replace fixed activation functions with learnable spline curves, the model's learned relationships are directly visualizable. You can literally plot how each input feature affects the output prediction. The goal wasn't to beat buy-and-hold; it was to build something interpretable enough that you can *see* what the model learned, and then ask whether what it learned makes economic sense.

---

## Architecture decisions

### Why KANs over MLPs

Standard MLPs apply fixed, non-learnable activations (ReLU, sigmoid) at each neuron. KANs move the learnable parameters to the *edges* of the network, where each connection is parameterized as a B-spline curve. This means the function each feature computes is directly inspectable after training — you get a plot of the input-output relationship for every feature in every regime. That interpretability is the entire point of this project. A black-box model might learn the same relationships, but you'd never know what they were.

### Why HMM over Gumbel-Softmax

The KASPER paper uses a differentiable Gumbel-Softmax regime detector so the whole system can be trained end-to-end. I explored this approach (`src/regime_model.py`) but found it produced unstable regime assignments during walk-forward retraining — the detector would collapse to a near-uniform distribution over regimes on some windows, making the per-regime KANs effectively untrained. Switching to a Gaussian HMM decoupled regime detection from forecasting, which made the pipeline easier to validate and the regimes easier to interpret. The tradeoff is that the HMM and KANs are no longer jointly optimized, but for a first pass the stability was worth it.

### Why four ETFs

SPY alone is insufficient — you'd be modeling equity dynamics in a vacuum. Adding QQQ captures growth/tech-sector behavior that doesn't fully track SPY. GLD and TLT are the important ones: gold and long-duration Treasuries are the canonical flight-to-safety assets, so their momentum and volatility carry information about cross-asset positioning that leads equity returns. The four-asset design is what makes the TLT finding possible (see below).

---

## Implementation

### Feature engineering

For each of SPY, QQQ, GLD, and TLT, I compute four daily features: raw return, 20-day rolling volatility, 20-day momentum, and 14-day RSI. That produces a 16-dimensional feature vector per day. The 20-day window is a deliberate choice — short enough to be reactive to regime transitions, long enough to avoid fitting to noise.

### Regime detection

A Gaussian HMM with 3 hidden states is fit to the full feature matrix. The number of states was chosen empirically: 2 states conflated meaningfully different market environments (calm bull vs. low-volatility sideways), while 4 states produced too few training samples per regime for reliable KAN training. The trained model's diagonal transition matrix has entries of approximately 0.97, which is a useful sanity check — it means the model learned that regimes are persistent, which matches how markets actually behave. Regime flipping day-to-day would be a sign something was wrong.

The three identified regimes map loosely to: high-volatility/risk-off (Regime 0), choppy/transitional (Regime 1), and calm bull (Regime 2).

### Per-regime KAN forecasters

After labeling each day with its regime, I train a separate KAN for each regime on only the days belonging to that regime. Each KAN takes the 16 features as input and outputs a scalar: predicted next-day SPY return. Training uses the `pykan` library built on PyTorch, with spline order 3 and a grid size tuned per regime based on the number of available training samples. For Regime 2 (calm bull), which has the most samples, I could afford a finer spline grid and more layers without overfitting.

---

## Key finding: bond momentum in calm bull regimes

This is the result I find most interesting, and it's only visible because of KAN interpretability.

In Regime 2 (calm bull), the spline visualization shows that `TLT_mom` — 20-day Treasury momentum — has a stronger and more monotonic relationship with next-day SPY returns than `SPY_mom` or `QQQ_mom`. Specifically, when bond momentum is high (money is flowing into long-duration Treasuries), the model assigns a lower predicted SPY return for the following day. When bond momentum is low or negative (outflows from Treasuries), predicted equity returns are higher.

This makes intuitive sense from a cross-asset flow perspective: in a calm bull regime, large rotations into Treasuries are unusual and tend to signal defensive positioning by institutional players — a precursor to equity softness, even without an obvious catalyst. The signal is weak enough that it gets averaged out when you train on all regimes together, but when the model is restricted to calm bull days only, it becomes the dominant learned relationship.

A standard MLP would have no mechanism to expose this. You'd have a trained model, an aggregate feature importance score, and no view into how the relationship behaves across the input range. The KAN spline plot shows the shape of the relationship directly.

---

## Backtest methodology and the leakage correction

Walk-forward out-of-sample validation. Both the HMM and all three KANs are retrained from scratch at each step using only historical data; predictions are generated only on days the model has never seen. Training window: 4 years (approximately 1,008 trading days). Retraining frequency: every 252 trading days.

**The leakage story is worth telling explicitly.** An earlier version of this backtest produced a Sharpe ratio of 6.28 — an immediately suspicious number. Investigating it revealed a look-ahead bias in how the HMM regime labels were being assigned: the HMM was being fit to the full dataset before the walk-forward loop, meaning future data was leaking into the regime labels used as training targets. The fix was to move the HMM fitting entirely inside the walk-forward loop, so at each step the regime model sees only the same historical window as the KANs. The corrected Sharpe is 0.38. I report the corrected number, not the inflated one — but I think the catch itself is worth showing, because finding and fixing a subtle leakage bug in a walk-forward backtest is exactly the kind of thing that separates rigorous quantitative work from curve-fit garbage.

---

## Results

Walk-forward out-of-sample backtest, 2014–2024:

| Metric | KAN Strategy | Buy & Hold SPY |
|---|---|---|
| Sharpe Ratio | 0.384 | 0.800 |
| Annualized Return | 3.9% | 13.2% |
| Max Drawdown | -28.3% | -33.7% |
| Annualized Volatility | 11.8% | 17.5% |
| Days in Market | ~44% | 100% |

The strategy underperforms buy-and-hold on raw return. The signal is conservative — the model is in cash roughly 56% of the time — which compresses volatility and drawdown but misses significant upside during sustained bull runs. The regime detector is doing something real (the risk-off periods it identifies largely correspond to 2015–16 volatility, the 2018 Q4 selloff, and the 2020 COVID crash), but the KAN forecasters need stronger signal or richer features to justify more aggressive positioning. Underperformance is reported, not hidden.

---

## What I'd do next

**Short selling in bear regimes.** Right now the strategy goes flat (cash) on negative predictions. Adding a short SPY position in Regime 0 would let the strategy profit from the drawdowns it's already correctly identifying rather than just avoiding them. This is the highest-expected-value extension.

**Richer macro features.** The current 16 features are purely price-derived. Adding VIX (implied vs. realized vol spread), the 2s10s yield curve slope, and investment-grade credit spreads would give the HMM substantially more signal about the macro environment, likely producing cleaner regime separation.

**Regime label consistency across walk-forward windows.** HMMs have a label-switching problem: Regime 0 in window *t* may not correspond to the same market environment as Regime 0 in window *t+1*. I handle this with a simple heuristic (sorting regimes by mean volatility), but a more principled fix would use constrained HMM initialization that anchors states to consistent market characteristics across windows.

**Longer training windows.** Four years per window may be insufficient for the per-regime KANs, especially for Regime 0 (high-volatility/bear), which has the fewest training days by construction. Expanding to 6–8 year windows would increase sample sizes at the cost of slower adaptation.

---

## Project structure

```
KAN_regimes/
│
├── src/
│   ├── data_loader.py            # ETF data fetching and feature engineering
│   ├── regime_model.py           # PyTorch Gumbel-Softmax detector (explored, replaced by HMM)
│   └── kan_forecaster.py         # KAN construction and training utilities
│
├── notebooks/
│   ├── 01_eda.ipynb              # Data exploration and feature visualization
│   ├── 02_regime_detection.ipynb # HMM training, regime labeling, transition matrix analysis
│   ├── 03_kan_training.ipynb     # Per-regime KAN training and spline visualization
│   └── 04_backtest.ipynb         # Walk-forward backtest and performance analysis
│
├── data/processed/
│   └── regime_labels.csv         # HMM regime label for each trading day
│
├── models/
│   ├── hmm_regime_detector.pkl   # Trained HMM
│   └── kan_regime_{0,1,2}.pt     # Trained KAN weights per regime
│
├── outputs/
│   ├── regime_labels.png         # Regime timeline overlaid on SPY price
│   ├── kan_splines_regime_{0,1,2}.png  # Spline visualizations per regime
│   └── equity_curve.png          # Cumulative return vs. buy-and-hold
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/AlexRadovich/KAN_regimes.git
cd KAN_regimes

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run notebooks in order (01 → 04) from the `notebooks/` directory. `data/` and `models/` are not committed — running the notebooks regenerates everything from scratch. Notebook 04 (walk-forward backtest) takes approximately 15–20 minutes on CPU.

---

## Dependencies

- `pykan` — KAN implementation
- `hmmlearn` — Gaussian Hidden Markov Model
- `yfinance` — ETF price data
- `torch` — PyTorch backend for KAN training
- `scikit-learn` — feature preprocessing
- `pandas`, `numpy`, `matplotlib`

---

## Based on

[KASPER: KAN-Based Architecture for Stock Price and Explainable Regime Detection](https://arxiv.org/abs/2507.18983) (arXiv 2507.18983, 2025). This project extends the KASPER approach to a multi-asset ETF portfolio, replaces the Gumbel-Softmax regime detector with a Gaussian HMM for more stable walk-forward behavior, and adds rigorous out-of-sample validation not present in the original paper.
