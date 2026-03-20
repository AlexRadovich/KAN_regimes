# KAN Market Regime Strategy
 
Using Kolmogorov-Arnold Networks (KANs) to build an interpretable, regime-conditional equity return forecasting system. Inspired by the KASPER paper (2025).
 
---
 
## Motivation
 
Most ML trading strategies treat all market conditions the same — they train one model on everything and hope it generalizes. But markets have distinct personalities: calm bull runs, choppy sideways periods, and high-volatility crises all have different statistical properties and different predictive signals.
 
This project tests whether explicitly modeling those regimes — and training a separate forecasting model for each one — produces more interpretable and robust predictions than a single global model.
 
---
 
## What this project does
 
**Step 1 — Feature engineering across four ETFs**
 
Rather than looking at one asset in isolation, the model ingests daily data across SPY (S&P 500), QQQ (Nasdaq 100), GLD (gold), and TLT (long-term Treasuries). For each asset it computes daily return, rolling 20-day volatility, 20-day momentum, and RSI — 16 features total. The multi-asset design captures cross-market dynamics that single-asset models miss.
 
**Step 2 — Unsupervised regime detection via Hidden Markov Model**
 
A Gaussian HMM identifies three hidden market regimes from the feature data without any labeled examples. The model learns a transition matrix encoding how likely the market is to stay in each regime day-to-day. The trained model achieved diagonal transition probabilities of ~0.97, meaning each regime is highly persistent — consistent with how real market environments actually behave.
 
**Step 3 — Regime-conditional KAN forecasters**
 
A separate KAN is trained for each regime on only the days belonging to that regime. Each KAN takes the 16 features as input and outputs a predicted next-day SPY return. Because KANs use learnable spline functions on each connection rather than fixed activations, the learned relationships are directly visualizable — you can plot exactly how each feature affects the prediction in each market environment.
 
**Step 4 — Walk-forward out-of-sample backtest**
 
The strategy is evaluated using strict walk-forward validation: both the HMM and the KANs are retrained from scratch at each step using only past data, and signals are generated only on days the model has never seen. This eliminates data leakage and produces honest out-of-sample results.
 
---
 
## Key finding
 
In calm bull market regimes (Regime 2), the model's spline visualization showed that bond momentum (`TLT_mom`) has a stronger and more consistent relationship with next-day equity returns than equity momentum itself. When bond momentum is high — meaning money is flowing into Treasuries — the model predicts lower SPY returns the following day. This cross-asset relationship only becomes visible when the model is allowed to specialize per regime rather than averaging across all market conditions.
 
This is the kind of insight that emerges naturally from the interpretability of KANs. A black-box model might have learned the same relationship, but you would never know it was there.
 
---
 
## Results
 
Walk-forward out-of-sample backtest, 2014–2024:
 
| Metric | KAN Strategy | Buy & Hold SPY |
|---|---|---|
| Sharpe Ratio | 0.384 | 0.800 |
| Annualized Return | 3.9% | 13.2% |
| Max Drawdown | -28.3% | -33.7% |
| Annualized Volatility | 11.8% | 17.5% |
 
*All results are fully out-of-sample. The model is retrained every 252 trading days using only historical data.*
 
**Honest interpretation:** The strategy underperforms buy-and-hold on raw return. The signal generation is conservative — the model spends roughly 44% of days in cash — which reduces volatility and drawdown but also misses significant upside. The regime detector is identifying genuine risk-off periods, but the KAN forecasters need stronger signal to justify more active positioning.
 
This is a first-pass implementation, not a production strategy. The results are meaningful because they are honest.
 
---
 
## What I'd do next
 
- **Short selling in bear regimes** — currently the strategy goes to cash on negative predictions. Adding a short position in bear regimes would let the strategy profit from downturns rather than just avoiding them, likely improving the Sharpe significantly.
- **Richer feature set** — adding macro features like the VIX, yield curve slope, and credit spreads would give the regime detector more information to work with.
- **Longer training windows** — the current setup uses 4 years of training data per window. More data per window may produce more stable KAN forecasters.
- **Regime label consistency** — the label switching problem in HMMs (where Regime 0 in one window may not correspond to the same market condition as Regime 0 in the next) is a known limitation worth addressing with constrained initialization.
 
---
 
## Project structure
 
```
KAN_regimes/
│
├── src/
│   ├── data_loader.py        # ETF data fetching and feature engineering
│   ├── regime_model.py       # PyTorch Gumbel-Softmax detector (explored but replaced)
│   └── kan_forecaster.py     # KAN building and training helpers
│
├── notebooks/
│   ├── 01_eda.ipynb          # Data exploration and visualization
│   ├── 02_regime_detection.ipynb  # HMM training and regime labeling
│   ├── 03_kan_training.ipynb      # Per-regime KAN training and spline plots
│   └── 04_backtest.ipynb          # Walk-forward backtest and performance analysis
│
├── data/processed/
│   └── regime_labels.csv     # HMM regime label for each trading day
│
├── models/
│   ├── hmm_regime_detector.pkl    # Trained HMM
│   └── kan_regime_{0,1,2}.pt      # Trained KAN weights per regime
│
├── outputs/
│   ├── regime_labels.png          # Regime timeline overlaid on SPY price
│   ├── kan_splines_regime_{0,1,2}.png  # Spline visualizations per regime
│   └── equity_curve.png           # Cumulative return vs buy-and-hold
│
├── requirements.txt
└── README.md
```
 
---
 
## Setup
 
```bash
# Clone the repo
git clone https://github.com/AlexRadovich/KAN_regimes.git
cd KAN_regimes
 
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
```
 
Then run the notebooks in order (01 through 04) from the `notebooks/` directory.
 
**Note:** `data/` and `models/` are not committed to the repo. Running the notebooks in sequence will regenerate all data and trained models from scratch. Notebook 04 (the walk-forward backtest) takes approximately 15–20 minutes to complete on CPU.
 
---
 
## Dependencies
 
- `pykan` — KAN implementation
- `hmmlearn` — Hidden Markov Model
- `yfinance` — ETF price data
- `torch` — PyTorch for KAN training
- `scikit-learn` — preprocessing
- `pandas`, `numpy`, `matplotlib`
 
---
 
## Based on
 
Inspired by [KASPER: KAN-Based Architecture for Stock Price and Explainable Regime Detection](https://arxiv.org/abs/2507.18983) (2025). This project extends the paper's approach to a multi-asset ETF portfolio and replaces the Gumbel-Softmax regime detector with a Hidden Markov Model for more stable regime identification. All backtest results use walk-forward validation not present in the original paper.
 