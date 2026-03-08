"""
Price forecasting module.

Primary:  LightGBM + MAPIE conformal prediction (no PyTorch dependency).
          Works reliably with 60–90 days of weekly data.

Chronos-Bolt-Tiny is the preferred option per the spec but requires PyTorch.
The LightGBM+MAPIE path is the robust default; Chronos is opt-in.

Audit fixes (2026-03):
  • Comb method: statsmodels.ExponentialSmoothing(optimized=True) replaces
    fixed α=0.3, β=0.1, φ=0.9 (Hyndman recommends MLE optimization)
  • MAPIE: compatibility shim for v0.9.x → v1.x API renames
  • Chronos: updated to reference bolt-tiny (9M params, 18MB)

Public API:
  forecast_prices(price_series, horizon, method) → ForecastResult
  make_features(series)                          → feature DataFrame
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Holds a price forecast with uncertainty intervals."""
    dates: List[str]                # forecast date strings (ISO)
    mean: List[float]               # point forecast ($/ton)
    ci_80_lo: List[float]           # 80% lower bound
    ci_80_hi: List[float]           # 80% upper bound
    ci_95_lo: List[float]           # 95% lower bound
    ci_95_hi: List[float]           # 95% upper bound
    method: str = "unknown"
    commodity: str = ""
    last_actual_date: str = ""
    last_actual_price: float = 0.0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_features(series: pd.Series, freq: str = "W") -> pd.DataFrame:
    """
    Build lag + rolling-window features from a price time series.

    Parameters
    ----------
    series  Price series indexed by date.
    freq    Temporal frequency hint: "W" (weekly) or "D" (daily).

    Returns
    -------
    DataFrame of features, aligned with series index.
    NaN rows (from lags) are dropped.
    """
    df = pd.DataFrame({"price": series})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Lag features
    for lag in [1, 2, 3, 4, 8, 13]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    # Rolling statistics
    for w in [4, 8, 13]:
        df[f"roll_mean_{w}"] = df["price"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["price"].shift(1).rolling(w).std()

    # Price momentum
    df["pct_chg_1"] = df["price"].pct_change(1)
    df["pct_chg_4"] = df["price"].pct_change(4)

    # Calendar features
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter

    return df.dropna()


# ---------------------------------------------------------------------------
# LightGBM + MAPIE forecaster (primary)
# ---------------------------------------------------------------------------

def _forecast_lgbm_mapie(
    series: pd.Series,
    horizon: int,
    alpha_levels: Tuple[float, float] = (0.20, 0.05),
) -> ForecastResult:
    """
    LightGBM + MAPIE 0.9.1 conformal prediction forecast.

    Parameters
    ----------
    series          Weekly (or daily) price series, sorted ascending.
    horizon         Number of steps ahead to forecast.
    alpha_levels    Significance levels: (α₁=0.20 → 80% CI, α₂=0.05 → 95% CI).
    """
    try:
        import lightgbm as lgb
        from sklearn.pipeline import Pipeline

        # MAPIE v0.9.x → v1.x compatibility shim
        # v1.x renames: MapieTimeSeriesRegressor → TimeSeriesRegressor,
        #               alpha → confidence_level, partial_fit → update
        _MAPIE_V1 = False
        try:
            from mapie.regression import MapieTimeSeriesRegressor
            from mapie.subsample import BlockBootstrap
        except ImportError:
            try:
                from mapie.time_series_regression import TimeSeriesRegressor as MapieTimeSeriesRegressor
                from mapie.subsample import BlockBootstrap
                _MAPIE_V1 = True
            except ImportError:
                raise ImportError("MAPIE not installed.")
    except ImportError as e:
        raise ImportError(
            f"LightGBM or MAPIE not installed: {e}. "
            "Run: pip install lightgbm mapie==0.9.1"
        )

    feat_df = make_features(series)
    if len(feat_df) < 15:
        raise ValueError(f"Insufficient data for MAPIE: {len(feat_df)} rows after feature engineering (need ≥15).")

    feature_cols = [c for c in feat_df.columns if c != "price"]
    X = feat_df[feature_cols].values
    y = feat_df["price"].values

    # Train/test split. MAPIE requires n_test ≥ ceil(1/alpha) for each alpha.
    # With alpha=[0.20, 0.05], need n_test ≥ 20. With alpha=[0.20], need n_test ≥ 5.
    min_alpha = min(alpha_levels)
    min_n_test = int(np.ceil(1.0 / min_alpha)) + 1

    n_test = max(min_n_test, len(feat_df) // 5)
    if n_test >= len(feat_df) - 5:
        # Not enough data for requested alphas; relax to single looser alpha
        alpha_levels = (max(alpha_levels),)
        min_alpha = min(alpha_levels)
        min_n_test = int(np.ceil(1.0 / min_alpha)) + 1
        n_test = max(min_n_test, len(feat_df) // 5)
    if n_test >= len(feat_df) - 5:
        raise ValueError(f"Too few samples ({len(feat_df)}) to fit MAPIE.")
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    n_blocks = max(2, min(6, len(X_train) // 6))
    cv_boot = BlockBootstrap(
        n_resamplings=n_blocks,
        n_blocks=n_blocks,
        overlapping=False,
        random_state=42,
    )

    mapie = MapieTimeSeriesRegressor(lgb_model, method="enbpi", cv=cv_boot, n_jobs=-1)
    mapie.fit(X_train, y_train)

    # Online update through test window
    y_pred_test = np.zeros(n_test)
    y_pis_test  = np.zeros((n_test, 2, 2))
    for step in range(n_test):
        p, pis = mapie.predict(X_test[step:step+1], alpha=list(alpha_levels), ensemble=True)
        y_pred_test[step]   = p[0]
        y_pis_test[step]    = pis[0]
        if step > 0:
            # MAPIE v1.x renamed partial_fit → update
            if _MAPIE_V1:
                mapie.update(X_test[step-1:step], y_test[step-1:step])
            else:
                mapie.partial_fit(X_test[step-1:step], y_test[step-1:step])

    # Multi-step ahead forecast (iterative / recursive)
    # Build a synthetic extension of the series and extract features.
    last_date   = feat_df.index[-1]
    freq_delta  = _infer_freq_delta(series)
    forecast_series = series.copy()

    means, lo80s, hi80s, lo95s, hi95s, dates = [], [], [], [], [], []

    for step in range(horizon):
        feat = make_features(forecast_series)
        if feat.empty:
            break
        x_next = feat[feature_cols].iloc[-1:].values
        p, pis = mapie.predict(x_next, alpha=list(alpha_levels), ensemble=True)
        pred = float(p[0])

        # pis shape: (1, 2, len(alpha_levels))
        lo80 = max(0.0, float(pis[0, 0, 0]))
        hi80 = float(pis[0, 1, 0])
        # 95% CI: use second alpha if available, else widen 80% CI
        if len(alpha_levels) >= 2:
            lo95 = max(0.0, float(pis[0, 0, 1]))
            hi95 = float(pis[0, 1, 1])
        else:
            lo95 = max(0.0, lo80 - abs(hi80 - lo80) * 0.5)
            hi95 = hi80 + abs(hi80 - lo80) * 0.5

        next_date = last_date + freq_delta * (step + 1)
        dates.append(next_date.strftime("%Y-%m-%d"))
        means.append(round(pred, 2))
        lo80s.append(round(lo80, 2))
        hi80s.append(round(hi80, 2))
        lo95s.append(round(lo95, 2))
        hi95s.append(round(hi95, 2))

        # Append prediction to extend the series for next iteration
        forecast_series = pd.concat([
            forecast_series,
            pd.Series([pred], index=[next_date]),
        ])

    last_actual = series.dropna()
    return ForecastResult(
        dates=dates,
        mean=means,
        ci_80_lo=lo80s,
        ci_80_hi=hi80s,
        ci_95_lo=lo95s,
        ci_95_hi=hi95s,
        method="LightGBM+MAPIE",
        last_actual_date=str(last_actual.index[-1].date()) if not last_actual.empty else "",
        last_actual_price=round(float(last_actual.iloc[-1]), 2) if not last_actual.empty else 0.0,
    )


# ---------------------------------------------------------------------------
# Chronos-Bolt-Small forecaster (optional / future use)
# ---------------------------------------------------------------------------

def _forecast_chronos(
    series: pd.Series,
    horizon: int,
) -> ForecastResult:
    """
    Chronos-Bolt-Tiny zero-shot probabilistic forecast.

    The chronos-bolt-tiny model has 9M parameters (~18 MB in bfloat16).
    The 2–4 GB dependency concern comes from PyTorch, not Chronos itself.
    Chronos-Bolt is designed for CPU inference (250× faster than original
    Chronos) and outputs quantile forecasts natively.

    Requires: pip install chronos-forecasting torch
    Pre-download model: amazon/chronos-bolt-tiny
    """
    try:
        import torch
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    except ImportError as e:
        raise ImportError(
            f"AutoGluon/PyTorch not available: {e}. "
            "Falling back to LightGBM+MAPIE."
        ) from e

    df_input = pd.DataFrame({
        "item_id": "price",
        "timestamp": series.index,
        "target": series.values,
    })
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df_input,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    )

    predictor = TimeSeriesPredictor(prediction_length=horizon, freq="W")
    predictor.fit(
        ts_df,
        hyperparameters={"Chronos": {"model_path": "amazon/chronos-bolt-tiny"}},
        verbosity=0,
    )
    preds = predictor.predict(ts_df)

    freq_delta = _infer_freq_delta(series)
    last_date  = pd.to_datetime(series.index[-1])
    dates = [
        (last_date + freq_delta * (i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    mean   = preds["mean"].values.tolist()
    lo80   = preds.get("0.1", preds["mean"] * 0.95).values.tolist()
    hi80   = preds.get("0.9", preds["mean"] * 1.05).values.tolist()
    lo95   = preds.get("0.025", preds["mean"] * 0.90).values.tolist()
    hi95   = preds.get("0.975", preds["mean"] * 1.10).values.tolist()

    last_actual = series.dropna()
    return ForecastResult(
        dates=dates,
        mean=[round(v, 2) for v in mean],
        ci_80_lo=[round(max(0, v), 2) for v in lo80],
        ci_80_hi=[round(v, 2) for v in hi80],
        ci_95_lo=[round(max(0, v), 2) for v in lo95],
        ci_95_hi=[round(v, 2) for v in hi95],
        method="Chronos-Bolt-Small",
        last_actual_date=str(last_actual.index[-1].date()) if not last_actual.empty else "",
        last_actual_price=round(float(last_actual.iloc[-1]), 2) if not last_actual.empty else 0.0,
    )


# ---------------------------------------------------------------------------
# Simple trend+seasonality fallback (when data is too sparse)
# ---------------------------------------------------------------------------

def _forecast_simple(
    series: pd.Series,
    horizon: int,
) -> ForecastResult:
    """
    Comb method: simple average of SES, Holt, and Damped Holt forecasts
    with OPTIMIZED parameters via statsmodels MLE.

    The M4 Competition's Comb benchmark used optimized parameters (via R's
    forecast package). Fixed params (old: α=0.3, β=0.1, φ=0.9) eliminate
    the method's ability to adapt to each series' autocorrelation structure.
    For ag commodity prices with regime changes, optimized α often exceeds 0.8.

    Falls back to fixed params if statsmodels unavailable.
    Used when data is too sparse for LightGBM.
    """
    s = series.dropna().astype(float)
    n = len(s)

    if n < 3:
        last = float(s.iloc[-1]) if n > 0 else 0.0
        dates_out = []
        freq_delta = pd.Timedelta(days=7)
        last_date = pd.to_datetime(s.index[-1])
        for i in range(horizon):
            dates_out.append((last_date + freq_delta * (i + 1)).strftime("%Y-%m-%d"))
        empty_list = [last] * horizon
        return ForecastResult(
            dates=dates_out, mean=empty_list,
            ci_80_lo=[last * 0.9] * horizon, ci_80_hi=[last * 1.1] * horizon,
            ci_95_lo=[last * 0.85] * horizon, ci_95_hi=[last * 1.15] * horizon,
            method="naive",
        )

    # Try statsmodels optimized ETS; fall back to manual implementation
    use_statsmodels = False
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWSmoothing
        use_statsmodels = True
    except ImportError:
        pass

    freq_delta = _infer_freq_delta(s)
    last_date  = pd.to_datetime(s.index[-1])

    if use_statsmodels and n >= 6:
        # --- Optimized ETS via statsmodels ---
        # Each model independently optimizes its parameters via MLE.
        s_idx = s.copy()
        s_idx.index = pd.to_datetime(s_idx.index)

        try:
            # SES (no trend, no seasonal)
            ses_model = HWSmoothing(s_idx, trend=None, seasonal=None).fit(optimized=True)
            f_ses_all = ses_model.forecast(horizon)
        except Exception:
            f_ses_all = pd.Series([float(s_idx.iloc[-1])] * horizon)

        try:
            # Holt (additive trend, no damping)
            holt_model = HWSmoothing(s_idx, trend="add", damped_trend=False, seasonal=None).fit(optimized=True)
            f_holt_all = holt_model.forecast(horizon)
        except Exception:
            f_holt_all = f_ses_all  # fallback

        try:
            # Damped Holt (additive trend with damping)
            dh_model = HWSmoothing(s_idx, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
            f_dh_all = dh_model.forecast(horizon)
        except Exception:
            f_dh_all = f_ses_all  # fallback

        method_label = "Comb-Optimized (SES+Holt+Damped)"
    else:
        # --- Manual fixed-param fallback ---
        alpha_ses = 0.3
        ses_level = float(s.iloc[0])
        for price in s.values[1:]:
            ses_level = alpha_ses * price + (1 - alpha_ses) * ses_level

        alpha_h, beta_h = 0.3, 0.1
        holt_level = float(s.iloc[0])
        holt_trend = 0.0
        for price in s.values[1:]:
            prev = holt_level
            holt_level = alpha_h * price + (1 - alpha_h) * (holt_level + holt_trend)
            holt_trend = beta_h * (holt_level - prev) + (1 - beta_h) * holt_trend

        alpha_d, beta_d, phi = 0.3, 0.1, 0.9
        dh_level = float(s.iloc[0])
        dh_trend = 0.0
        for price in s.values[1:]:
            prev = dh_level
            dh_level = alpha_d * price + (1 - alpha_d) * (dh_level + phi * dh_trend)
            dh_trend = beta_d * (dh_level - prev) + (1 - beta_d) * phi * dh_trend

        f_ses_all = pd.Series([ses_level] * horizon)
        f_holt_all = pd.Series([holt_level + holt_trend * i for i in range(1, horizon + 1)])
        phi_sums = [phi * (1 - phi ** i) / (1 - phi) if phi != 1.0 else i for i in range(1, horizon + 1)]
        f_dh_all = pd.Series([dh_level + dh_trend * ps for ps in phi_sums])
        method_label = "Comb-Fixed (SES+Holt+Damped)"

    # Historical volatility for uncertainty bands
    returns = s.pct_change().dropna()
    vol = float(returns.std()) if len(returns) > 1 else 0.02

    dates_out, means, lo80s, hi80s, lo95s, hi95s = [], [], [], [], [], []
    for i in range(horizon):
        # Comb: simple average of three models
        f_ses = float(f_ses_all.iloc[i]) if i < len(f_ses_all) else float(f_ses_all.iloc[-1])
        f_holt = float(f_holt_all.iloc[i]) if i < len(f_holt_all) else float(f_holt_all.iloc[-1])
        f_dh = float(f_dh_all.iloc[i]) if i < len(f_dh_all) else float(f_dh_all.iloc[-1])
        fcast = (f_ses + f_holt + f_dh) / 3.0

        sigma = abs(fcast) * vol * ((i + 1) ** 0.5)
        dates_out.append((last_date + freq_delta * (i + 1)).strftime("%Y-%m-%d"))
        means.append(round(fcast, 2))
        lo80s.append(round(max(0, fcast - 1.28 * sigma), 2))
        hi80s.append(round(fcast + 1.28 * sigma, 2))
        lo95s.append(round(max(0, fcast - 1.96 * sigma), 2))
        hi95s.append(round(fcast + 1.96 * sigma, 2))

    last_actual = series.dropna()
    return ForecastResult(
        dates=dates_out,
        mean=means,
        ci_80_lo=lo80s,
        ci_80_hi=hi80s,
        ci_95_lo=lo95s,
        ci_95_hi=hi95s,
        method=method_label,
        last_actual_date=str(last_actual.index[-1].date()) if not last_actual.empty else "",
        last_actual_price=round(float(last_actual.iloc[-1]), 2) if not last_actual.empty else 0.0,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def forecast_prices(
    price_series: pd.Series,
    horizon: int = 8,
    method: Literal["auto", "lgbm", "chronos", "simple"] = "auto",
    commodity: str = "",
) -> ForecastResult:
    """
    Forecast commodity prices with uncertainty intervals.

    Parameters
    ----------
    price_series  Pandas Series of historical prices ($/ton), date-indexed.
                  Ideally 60–90 weekly observations.
    horizon       Number of periods ahead to forecast.
    method        "auto" → try lgbm, fallback to simple.
                  "lgbm" → LightGBM + MAPIE.
                  "chronos" → Chronos-Bolt-Small (requires extra deps).
                  "simple" → Holt trend extrapolation.
    commodity     Label for display purposes.

    Returns
    -------
    ForecastResult
    """
    series = price_series.dropna().sort_index()

    if method == "chronos":
        result = _forecast_chronos(series, horizon)
    elif method == "simple":
        result = _forecast_simple(series, horizon)
    elif method == "lgbm":
        result = _forecast_lgbm_mapie(series, horizon)
    else:  # "auto"
        feat_df = make_features(series)
        # LightGBM+MAPIE enbpi needs ≥80 feature rows to reliably satisfy the
        # 1/alpha calibration requirement (alpha=0.05 → need ≥20 test samples,
        # leaving ≥60 for training + BlockBootstrap).
        # For shorter series, Holt trend with vol-based bands is preferable.
        if len(feat_df) >= 80:
            try:
                result = _forecast_lgbm_mapie(series, horizon)
            except Exception as e:
                result = _forecast_simple(series, horizon)
                result.metadata["fallback_reason"] = str(e)
        else:
            result = _forecast_simple(series, horizon)
            if len(feat_df) < 15:
                result.metadata["fallback_reason"] = f"Only {len(feat_df)} feature rows (need ≥15 for ML)"

    result.commodity = commodity
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infer_freq_delta(series: pd.Series) -> pd.Timedelta:
    """Infer the typical time step of a series."""
    idx = pd.to_datetime(series.index)
    if len(idx) < 2:
        return pd.Timedelta(days=7)
    deltas = [(idx[i+1] - idx[i]).days for i in range(min(5, len(idx)-1))]
    median_days = int(np.median(deltas))
    if median_days <= 2:
        return pd.Timedelta(days=1)
    elif median_days <= 10:
        return pd.Timedelta(days=7)
    else:
        return pd.Timedelta(days=30)


def build_price_series_from_usda(
    all_slug_data: Dict,
    ingredient_key: str,
) -> pd.Series:
    """
    Build a Pandas price Series from cached USDA slug data.
    Returns a Series indexed by date (weekly, $/ton).
    """
    from src.usda_fetcher import get_price_history
    records = get_price_history(all_slug_data, ingredient_key)
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    # Weekly resample: use mean of all reported prices that week
    weekly = df["price"].resample("W").mean().dropna()
    return weekly


def build_price_series_from_futures(
    commodity: str,
    use_live: bool = False,
) -> pd.Series:
    """
    Build a daily price Series from yfinance continuous contract data.
    Returns Series indexed by date (daily, $/ton).
    """
    from src.futures_fetcher import fetch_continuous
    df = fetch_continuous(commodity, period="6mo", use_live=use_live)
    if df is None or df.empty or "close_per_ton" not in df.columns:
        return pd.Series(dtype=float)
    return df["close_per_ton"].dropna()
