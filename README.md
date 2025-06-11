# ai_timeseries_forecasting

Great, I’ll generate a detailed repo layout based on the current repository structure and the descriptions of workflows 1 through 4. I’ll then design the implementation details and corresponding file layout for workflows 5 through 7 as described in `workflow.md`.

I’ll let you know once the layout and design are ready for review.


* **AI\_TimeSeries\_Forecasting/**

  * `data/` *(raw, interim, and processed datasets)*

    * `raw/`

      * `nasdaq.csv` – NASDAQ index daily prices dataset (Workflow 1 input)
      * `economic_indicators.csv` – Macroeconomic indicators dataset (Workflow 1 input)
      * `CHECKSUMS.txt` – SHA-256 checksums for original raw data files (Workflow 1)
    * `interim/`

      * `aligned.parquet` – Combined NASDAQ & macro data aligned on trading days (Workflow 1)
      * `clean.parquet` – Aligned data after cleaning missing values & outliers (Workflow 1)
    * `processed/`

      * `nasdaq_features.parquet` – Feature-enhanced dataset (log returns, tech indicators, ratios) for modeling (Workflow 1)
  * `notebooks/` *(Jupyter notebooks for EDA and model development)*

    * `01_data_exploration.ipynb` – Exploratory analysis (plots price trends, log-returns, ACF/PACF, stationarity tests) (Workflow 1)
    * `02_classical_models.ipynb` – Runs ARIMA/SARIMA models, analyzes residuals, and stores forecasts for classical models (Workflow 2)
    * `03_dl_baseline.ipynb` – Develops LSTM baseline model, logs training metrics and plots one-step forecast vs actual (Workflow 3)
    * `04_hybrid_model.ipynb` – Builds hybrid ARIMAX + LSTM model, generates combined forecasts and compares vs ARIMAX baseline (Workflow 4)
  * `src/` *(Python source code modules)*

    * `data/`

      * `make_dataset.py` – Loads raw CSVs (e.g. via Kaggle API), aligns trading days & merges NASDAQ with exogenous macro data (Workflow 1)
      * `clean.py` – Cleans merged data (drops non-trading days, filters outliers, fills missing values) and outputs cleaned dataset (Workflow 1)
    * `features/`

      * `build_features.py` – Computes features (daily log returns, % change, SMA, RSI, z-scores, ratios) and saves the processed feature dataset (Workflow 1)
    * `models/`

      * `arima.py` – Trains ARIMA and ARIMAX models on log-returns; saves fitted model objects and forecasts to CSV (Workflow 2)
      * `sarima.py` – Performs seasonal ARIMA (SARIMA) grid search for weekly seasonality; saves the best model, its one-step forecasts, and residuals (Workflow 2)
      * `garch_mod.py` – Fits a GARCH(1,1) model on ARIMAX residuals to model volatility; outputs saved model and 1-day variance forecast (Workflow 2)
      * `lstm_baseline.py` – Defines and trains an LSTM network for one-step daily return forecasting; saves model weights (`.pt`) and test set predictions (Workflow 3)
      * `residual_lstm.py` – Trains an LSTM on ARIMAX residual series (30-day window) to correct classical model errors; outputs residual forecasts and hybrid (corrected) predictions (Workflow 4)
      * `tune_resid_lstm.py` – Uses Optuna to tune residual LSTM hyperparameters (sequence length, hidden units, dropout, LR etc.); saves best config to JSON (Workflow 4)
      * `retrain_resid_lstm.py` – Retrains residual LSTM using the best hyperparameters from tuning; saves final model weights and updated forecasts (Workflow 4)
      * `tft.py` – **(Planned)** Implements a Temporal Fusion Transformer for multivariate time series; uses exogenous inputs, tunes heads/hidden/dropout (e.g. Optuna ≤25 trials), early-stops on validation MAE, and saves model weights & 1-day forecasts (Workflow 5)
      * `nbeats.py` – **(Planned)** Implements N-BEATS deep forecasting model (stacked trend/seasonality basis functions); trains on 30-day windows, tunes block width and number of stacks, and outputs the trained model and its forecasts (Workflow 5)
      * `tcn.py` – **(Planned)** Implements a Temporal Convolutional Network with dilated causal convolutions; tunes network depth (layers, filters, dilation growth); saves the trained TCN model and forecast results (Workflow 5)
      * `xgboost_regressor.py` – **(Planned)** Constructs a feature matrix of lagged values (up to 30-day lags) and technical indicators; trains an XGBoost regressor with time-series CV and hyperparameter search, then saves the model, SHAP feature importances, and forecast CSV (Workflow 6)
      * `lightgbm_regressor.py` – **(Planned)** Similar to XGBoost but using LightGBM; trains a LightGBM gradient boosting model on the lagged feature matrix, tunes tree parameters (depth, learning rate, etc.), and outputs the model and forecasts (Workflow 6)
      * `random_forest.py` – **(Planned)** Trains a Random Forest regressor on the lagged feature matrix as a baseline ML model (e.g. 500 trees, tuned max\_depth/min\_samples\_leaf for robustness); saves the fitted model and its one-day-ahead predictions (Workflow 6)
      * `svr.py` – **(Planned)** Performs Support Vector Regression on the feature matrix with an RBF kernel; grid-searches hyperparams (C, γ, ε), then saves the trained SVR model and its forecast results (Workflow 6)
      * `dynamic_regression.py` – **(Planned)** Fits a dynamic linear state-space regression model (Kalman filter) with time-varying coefficients to NASDAQ vs. exogenous series; saves the fitted state-space model and its forecasts (Workflow 7)
      * `varmax.py` – **(Planned)** Fits a VARMAX multivariate time series model on NASDAQ and key macro series to capture joint dynamics; outputs the saved VARMAX model and NASDAQ forecast for the test period (Workflow 7)
      * `prophet.py` – **(Planned)** Uses Facebook Prophet (a Bayesian structural time series model) with trend changepoints and regressors; produces forecasted values with confidence intervals and saves the model and forecast outputs (Workflow 7)
  * `reports/` *(analysis plots and visualizations)*

    * `plots/` *(generated figures for EDA and results – no manual editing, no descriptions needed)*

      * *Subfolders for each workflow (e.g. `eda/`, `classical/`, `dl/`, `hybrid/`) contain saved charts illustrating data and model forecasts.*
  * `artifacts/` *(serialized models, forecasts, residuals, metrics)*

    * `models/`

      * `arima.pkl` – Pickled ARIMA model object fitted on training data (Workflow 2)
      * `arimax.pkl` – Pickled ARIMAX model (ARIMA with exogenous macro inputs) (Workflow 2)
      * `sarima.pkl` – Saved SARIMA model with best seasonal order (Workflow 2)
      * `garch11_scaled.pkl` – Fitted GARCH(1,1) model on scaled residuals (Workflow 2)
      * `lstm_baseline.pt` – Trained LSTM baseline model weights (PyTorch) (Workflow 3)
      * `resid_lstm.pt` – Trained residual LSTM model weights for the hybrid model (PyTorch) (Workflow 4)
      * `tft.pth` – **(Planned)** Trained TFT model weights (PyTorch) for advanced DL forecast (Workflow 5)
      * `nbeats.pth` – **(Planned)** Trained N-BEATS model weights (PyTorch) (Workflow 5)
      * `tcn.pth` – **(Planned)** Trained TCN model weights (PyTorch) (Workflow 5)
      * `xgboost.model` – **(Planned)** Saved XGBoost model file (binary or pickle format) (Workflow 6)
      * `lightgbm.txt` – **(Planned)** Saved LightGBM model in text format (Workflow 6)
      * `random_forest.pkl` – **(Planned)** Pickled Random Forest model (Workflow 6)
      * `svr.pkl` – **(Planned)** Pickled SVR model (Workflow 6)
      * `dynamic_reg.pkl` – **(Planned)** Saved dynamic regression state-space model (Workflow 7)
      * `varmax.pkl` – **(Planned)** Pickled VARMAX multivariate model (Workflow 7)
      * `prophet.pkl` – **(Planned)** Saved Prophet model object with learned parameters (Workflow 7)
    * `forecasts/`

      * `arima_test.csv` – ARIMA model one-step-ahead forecast vs actuals on test set (Workflow 2)
      * `arimax_test.csv` – ARIMAX model one-step forecast for test period (Workflow 2)
      * `sarima_test.csv` – SARIMA model forecast on the test set (Workflow 2)
      * `garch_variance_scaled.csv` – One-day-ahead variance forecast from GARCH model (scaled back to original units) (Workflow 2)
      * `lstm_test.csv` – LSTM baseline model predictions on test data (Workflow 3)
      * `hybrid_test.csv` – Hybrid model (ARIMAX + residual LSTM) combined forecast on test data (Workflow 4)
      * `tft_test.csv` – **(Planned)** TFT model one-day forecasts on test set (Workflow 5)
      * `nbeats_test.csv` – **(Planned)** N-BEATS model one-day forecasts on test set (Workflow 5)
      * `tcn_test.csv` – **(Planned)** TCN model one-day forecasts on test set (Workflow 5)
      * `xgboost_test.csv` – **(Planned)** XGBoost model one-step forecast outputs on test set (Workflow 6)
      * `lightgbm_test.csv` – **(Planned)** LightGBM model one-step forecast outputs on test set (Workflow 6)
      * `random_forest_test.csv` – **(Planned)** Random Forest model one-step forecast outputs on test set (Workflow 6)
      * `svr_test.csv` – **(Planned)** SVR model one-step forecast outputs on test set (Workflow 6)
      * `dynamic_reg_test.csv` – **(Planned)** Dynamic regression model one-step forecast on test set (Workflow 7)
      * `varmax_test.csv` – **(Planned)** VARMAX model forecast for NASDAQ (and other series) on test set (Workflow 7)
      * `prophet_test.csv` – **(Planned)** Prophet model forecast for test set (point forecasts with confidence intervals) (Workflow 7)
    * `residuals/`

      * `arimax_residuals.csv` – In-sample residual series from the ARIMAX model (Workflow 2)
      * `sarima_residuals.csv` – In-sample residual series from the best SARIMA model (Workflow 2)
    * `metrics/`

      * `dl_metrics.csv` – LSTM baseline performance metrics (e.g. MAE, RMSE on validation/test sets) (Workflow 3)
      * `hybrid_metrics.csv` – Hybrid model performance metrics comparing hybrid vs ARIMAX (Workflow 4)
      * `advanced_dl_metrics.csv` – **(Planned)** Metrics (MAE, RMSE, MAPE, etc.) for advanced DL models (TFT, N-BEATS, TCN) (Workflow 5)
      * `ml_regressor_metrics.csv` – **(Planned)** Metrics for machine learning regressors (XGBoost, LightGBM, Random Forest, SVR) (Workflow 6)
      * `advanced_classical_metrics.csv` – **(Planned)** Metrics for advanced classical models (Dynamic regression, VARMAX, Prophet) including interval scores (Workflow 7)
    * `hparams/`

      * `best_resid_lstm.json` – Best hyperparameters found for residual LSTM (saved from tuning phase) (Workflow 4)
  * `requirements1.txt` – Base environment dependencies (pandas, numpy, scipy, matplotlib, seaborn, statsmodels, etc.) for data prep and EDA (Workflow 0/1)
  * `requirements2.txt` – Additional dependencies for classical modeling (e.g. `pmdarima` for auto-ARIMA, `arch` for GARCH, joblib) (Workflow 2)
  * `requirements3.txt` – Additional dependencies for deep learning (PyTorch, torchmetrics, tqdm, optuna for tuning) (Workflow 3/4)
  * **(Planned)** `requirements4.txt` – Additional dependencies for advanced deep learning models (e.g. `pytorch-forecasting` for TFT, any needed transformer libs) (Workflow 5)
  * **(Planned)** `requirements5.txt` – Additional dependencies for ML regressors (e.g. `xgboost`, `lightgbm`, and `shap` for feature importance) (Workflow 6)
  * **(Planned)** `requirements6.txt` – Additional dependencies for advanced classical/probabilistic models (e.g. `prophet` for Bayesian structural forecasting) (Workflow 7)
  * `README.md` – Project overview, setup instructions, and usage examples (Workflow 0)
  * `Workflow.md` – Detailed phase-by-phase project plan and task checklist for workflows 1–9
