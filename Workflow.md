Below is a structured, action-oriented checklist that turns every element of the proposal into concrete, trackable tasks.  I’ve grouped items by logical phase so you can drop them straight into a Kanban board or milestone tracker.  (Where a task spans two phases, I list it where it starts.)

---

### Phase 0  — Project Setup

1. **Create version-controlled workspace**

   * New git repo (or dedicated branch) with `data/`, `src/`, `notebooks/`, `reports/`.
   * Add a `conda`/`pip` lock file listing: `pandas`, `statsmodels`, `pmdarima`, `arch`, `torch`, `sklearn`, `matplotlib`, `seaborn`, `scipy`, `kaggle`, `statsforecast`, `prophet` (optional) etc.
2. **Secure dataset access**

   * Obtain Kaggle API token; download “NASDAQ + Macroeconomic Indicators 2010-2024”.
   * Store raw CSVs under `data/raw/`; record SHA-256 hashes for provenance.
3. **Establish experiment-tracking & logging**

   * Decide on MLflow / Weights & Biases / simple CSV logs.
   * Template a logging wrapper for all model runs.


---

### Phase 1  — Data Engineering & Exploratory Analysis

1. **Data cleaning pipeline**

   * Parse dates, align trading days across series, forward-fill macro indicators on holidays.
   * Handle missing values, outliers (e.g., flash-crash spikes).
2. **Feature engineering**

   * Compute daily returns and % changes; optionally add technical indicators (SMA, RSI, etc.).
   * Create nondimensional features (ratios, z-scores) to satisfy the dimensional-analysis step.
3. **Exploratory Data Analysis (EDA)**

   * Plot price level, log-returns, ACF/PACF, rolling variance.
   * Check stationarity (ADF, KPSS) pre- and post-differencing.
   * Inspect correlations between NASDAQ and each exogenous series.
4. **Persist tidy, feature-ready dataset** to `data/processed/`.

---

### Phase 2  — Classical Models

1. **ARIMA family**

   * Automated order search (`pmdarima.auto_arima`) for ARIMA, ARIMAX (exogenous inputs).
   * Fit, save parameters, generate one-step-ahead forecasts.
2. **SARIMA**

   * Seasonality diagnostics (STL/periodogram), grid-search seasonal orders.
3. **GARCH(1,1) volatility model**

   * Fit on log-return residuals from best ARIMA/ARIMAX model.
   * Store conditional variance forecasts for later analysis.
4. **Residual export**

   * Save in-sample residual series for each model → feeds Phase 4.

---

### Phase 3  — Pure Deep-Learning Baseline

1. **Prepare supervised sequences**

   * Choose look-back window (e.g., 30 days) and forecast horizon (1 day ahead).
2. **LSTM architecture & training loop**

   * Layers, hidden units, dropout, learning-rate schedule, early stopping.
   * Hyperparameter search (grid or Bayesian; limit epochs to avoid overfit).
3. **Baseline evaluation**

   * Forecast on validation/test sets; log MAE, RMSE, MAPE; save weights.

---

### Phase 4  — Hybrid Model (Residual LSTM)

1. **Dataset assembly**

   * Input = \[residual\_t-30 … residual\_t-1]; target = residual\_t.
   * Optionally concatenate exogenous variables.
2. **Train & tune residual-correction LSTM**

   * Same search procedure; monitor improvement vs. classical residuals.
3. **Combine forecasts**

   * `ŷ_hybrid = ŷ_classical + ŷ_residual_LSTM`.
   * Save full forecast series for comparison.

---

### Phase 5 — Advanced Deep-Learning Models  *(new)*  

1. **Transformer Family (TFT)**  
   * Implement Temporal Fusion Transformer or equivalent sequence-to-sequence attention model with exogenous inputs.  
   * Tune heads, hidden size, dropout, and learning-rate schedule (Optuna or random search ≤ 25 trials).  
   * Early stopping on rolling-origin validation MAE.

2. **N-BEATS**  
   * Train generic & interpretable stacks on 30-day windows.  
   * Tune block width, #stacks, trend/seasonality mode; monitor validation loss.  

3. **Temporal Convolutional Network (TCN)**  
   * Dilated causal convolutions (kernel 3, residual blocks) with exogenous channels.  
   * Hyper-params: layers (4-6), filters (32-128), dilation growth.  

4. **Evaluation**  
   * Produce 1-step-ahead forecasts; record MAE / RMSE / MAPE.  
   * Diebold-Mariano test vs. ARIMAX and LSTM.  
   * Save weights → `artifacts/models/` and forecasts → `artifacts/forecasts/`.

---

### Phase 6 — Machine-Learning Regressors  *(new)*  

1. **Feature Matrix Construction**  
   * Sliding window of lags (≤ 30) for NASDAQ & macro; add rolling stats & tech indicators.  
   * Time-series cross-validation splits (no shuffling).

2. **XGBoost / LightGBM**  
   * Grid/Bayesian search for trees, depth, learning rate.  
   * Capture feature importance; log SHAP values.

3. **Random Forest**  
   * 500-tree baseline; tune max_depth / min_leaf for robustness.

4. **Support Vector Regression (SVR)**  
   * RBF kernel; grid search on C, γ, ε with standardized features.

5. **Evaluation**  
   * Rolling 1-day forecasts; MAE / RMSE / MAPE; DM test vs. ARIMAX.  
   * Persist models & metrics.

---

### Phase 7 — Advanced Classical & Probabilistic Models  *(new)*  

1. **Dynamic Linear / State-Space Regression**  
   * Local-level + time-varying regression coefficients (Kalman filter).  
   * Compare to static ARIMAX on structural-break periods.

2. **VARMAX (Small System)**  
   * Model NASDAQ + 2-3 key macro series jointly; conditional forecasts.  
   * Check Granger causality & impulse-response for interpretability.

3. **Bayesian Structural TS / Prophet**  
   * Trend + changepoints + regressors with priors; sample posterior forecasts.  
   * Report prediction-interval coverage & CRPS.

4. **Evaluation**  
   * Same point-forecast metrics; add interval coverage & calibration plots.  
   * Summarize explainability benefits vs. accuracy.

---

### Phase 8  — Statistical Evaluation & Sensitivity

1. **Metric computation**

   * MAE, RMSE, MAPE for all models; tabulate.
2. **Diebold-Mariano significance tests**

   * Pairwise compare hybrid vs. each baseline.
3. **Perturbation (first-order) sensitivity**

   * Vary ARIMA coefficients ±ε; compute forecast deltas.
   * Vary key LSTM inputs; capture gradient-based sensitivities.
4. **Robustness checks**

   * Rolling-origin evaluation (walk-forward).
   * Down-sampled weekly data ablation.

---

### Phase 9  — Documentation & Deliverables

1. **Maintain experiment notebook(s)**

   * Narrative EDA, model snippets, plots.
2. **Write mathematical appendix**

   * Explain dimensional analysis steps, perturbation math.
3. **Draft final report**

   * Structure: Intro, Methods, Results, Discussion, Limitations.
   * Embed tables, figures, significance results.
4. **Prepare slide deck**

   * 10-12 slides emphasizing motivation, hybrid gains, key plots.
5. **Rehearse & deliver presentation**

---

### Phase 10  — Project Management & Risk Mitigation

1. **Weekly milestone reviews** (align with proposal’s Week 5-10 timetable).
2. **Re-training schedule for non-stationarity** (decide on rolling window length).
3. **Regularize deep models** (dropout/L2, monitor validation gap).
4. **Automated backups** of data, code, and experiment artifacts.

---

### Optional Enhancements (time permitting)

* Temporal Fusion Transformer (TFT) or N-BEATS baseline for comparison.
* Probabilistic forecasting (quantiles, prediction intervals).
* Simple trading-strategy back-test to demonstrate practical impact.

---

Use this list as a living backlog: break each bullet into specific issues (code module, notebook, or experiment config) and tag them with the matching week number from the original timeline to keep progress visible. Good luck turning the proposal into results!
