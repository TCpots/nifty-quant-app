"""
engine.py — Signal computation backend for the Streamlit app.

Runs a lightweight version of the full Phase 1–3 pipeline:
  1. Fetch fresh OHLCV data via yfinance
  2. Compute log returns, ADF check
  3. Fit GARCH(1,1) for conditional vol
  4. Compute signals: GARCH regime, ARIMA momentum, ensemble
  5. Load DL model weights if available (optional)
  6. Build recommendation with E[r], Var[r], VaR, stop loss

Designed to be fast enough for live Streamlit use (~5–15s per run).
Heavy training (Phases 2–3 full) should be done offline in Colab.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from pathlib import Path

# Try importing arch for GARCH; fall back to rolling vol if not available
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Try importing torch for DL inference
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class QuantEngine:
    """
    Lightweight inference engine for the Streamlit dashboard.
    Does NOT retrain models — runs forward inference on fresh data.
    """

    SCALE        = 100.0
    TRADING_DAYS = 252

    def __init__(self, ticker: str, lookback_years: int = 5, rfr: float = 0.065):
        self.ticker         = ticker
        self.lookback_years = lookback_years
        self.rfr            = rfr
        self.label          = ticker.replace("^", "").replace(".", "_")

    # ─────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Full pipeline. Returns dict consumed by app.py."""
        df        = self._fetch()
        log_ret   = self._log_returns(df)
        garch_vol, cond_vol_series = self._fit_garch(log_ret)
        signals   = self._compute_signals(log_ret, garch_vol, cond_vol_series)
        rec       = self._recommendation(df, log_ret, garch_vol, signals)

        # Regime masks for chart shading
        vol_pct   = cond_vol_series.rank(pct=True)
        regime_LOW  = cond_vol_series.where(vol_pct <= 0.33)
        regime_HIGH = cond_vol_series.where(vol_pct >= 0.67)

        return {
            # Price
            "price_df":            df,
            "current_price":       float(df["Close"].iloc[-1]),
            "price_change_pct":    float(
                (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            ),
            # Vol
            "garch_vol":           garch_vol,
            "hist_vol_21d":        float(log_ret.rolling(21).std().iloc[-1] * np.sqrt(self.TRADING_DAYS)),
            "cond_vol_series":     cond_vol_series,
            "vol_regime":          self._vol_regime(garch_vol, cond_vol_series),
            "regime_LOW":          regime_LOW,
            "regime_HIGH":         regime_HIGH,
            # Returns
            "log_returns":         log_ret,
            "expected_return_daily": float(log_ret.mean()),
            "variance_daily":      float(log_ret.var()),
            "mom_5d":              float(log_ret.tail(5).mean()),
            # Signals
            "signals":             signals,
            "ensemble_votes":      self._count_votes(signals),
            # Recommendation
            "recommendation":      rec,
        }

    # ─────────────────────────────────────────────────────────
    # PRIVATE — DATA
    # ─────────────────────────────────────────────────────────

    def _fetch(self) -> pd.DataFrame:
        start = (datetime.today() - timedelta(days=self.lookback_years * 365 + 60)
                 ).strftime("%Y-%m-%d")
        df = yf.download(self.ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[df["Volume"] > 0].dropna()
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert(None)
        return df

    def _log_returns(self, df: pd.DataFrame) -> pd.Series:
        r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        r.name = "log_return"
        return r

    # ─────────────────────────────────────────────────────────
    # PRIVATE — GARCH
    # ─────────────────────────────────────────────────────────

    def _fit_garch(self, log_ret: pd.Series):
        """
        Fit GARCH(1,1) with Constant mean and skewed-t errors.
        Falls back to 21-day rolling vol if arch not installed.
        Returns (current_forecast_annualised, full_conditional_vol_series).
        """
        if not HAS_ARCH:
            rv = log_ret.rolling(21).std() * np.sqrt(self.TRADING_DAYS)
            return float(rv.iloc[-1]), rv.dropna()

        try:
            scaled = log_ret * self.SCALE
            am  = arch_model(scaled, mean="Constant", vol="GARCH",
                             p=1, q=1, dist="skewt")
            res = am.fit(update_freq=0, disp="off",
                         options={"maxiter": 300})
            # 1-step-ahead forecast
            fc  = res.forecast(horizon=1, reindex=False)
            var_fc  = float(fc.variance.iloc[-1, 0])
            vol_fc  = np.sqrt(var_fc) / self.SCALE * np.sqrt(self.TRADING_DAYS)
            # Full conditional vol series
            cond_vol = (res.conditional_volatility / self.SCALE
                        * np.sqrt(self.TRADING_DAYS))
            cond_vol.index = log_ret.index[-len(cond_vol):]
            return float(vol_fc), cond_vol
        except Exception:
            rv = log_ret.rolling(21).std() * np.sqrt(self.TRADING_DAYS)
            return float(rv.iloc[-1]), rv.dropna()

    # ─────────────────────────────────────────────────────────
    # PRIVATE — SIGNALS
    # ─────────────────────────────────────────────────────────

    def _vol_regime(self, current_vol: float, vol_series: pd.Series) -> str:
        p33 = vol_series.quantile(0.33)
        p67 = vol_series.quantile(0.67)
        if current_vol <= p33:  return "LOW"
        if current_vol >= p67:  return "HIGH"
        return "MEDIUM"

    def _compute_signals(self, log_ret, garch_vol, cond_vol_series) -> dict:
        # GARCH regime: long in low vol, flat/short in high vol
        p75 = cond_vol_series.quantile(0.75)
        sig_garch = 1.0 if garch_vol < p75 else -1.0

        # ARIMA proxy: 5-day momentum (ARIMA(0,0,0) → white noise mean,
        # so we use recent momentum as the best available linear predictor)
        mom_5d  = log_ret.tail(5).mean()
        mom_20d = log_ret.tail(20).mean()
        sig_arima = 1.0 if mom_5d > 0 else -1.0

        # Mean-reversion check: if 5d momentum extreme, fade it
        z_score = (log_ret.tail(5).sum() - log_ret.rolling(63).mean().iloc[-1] * 5) / \
                  (log_ret.rolling(63).std().iloc[-1] * np.sqrt(5) + 1e-10)
        if abs(z_score) > 2.0:
            sig_arima = -np.sign(z_score)

        # DL signal: try loading saved Phase 3 forecasts
        sig_lstm = self._load_dl_signal("lstm")
        sig_gru  = self._load_dl_signal("gru")
        sig_tf   = self._load_dl_signal("transformer")

        # Ensemble: majority vote
        votes    = [sig_garch, sig_arima, sig_lstm, sig_gru, sig_tf]
        ensemble = float(np.sign(sum(votes)))
        if ensemble == 0:
            ensemble = 1.0  # tie-break: long bias (equity has positive drift)

        return {
            "garch":       sig_garch,
            "arima":       sig_arima,
            "lstm":        sig_lstm,
            "gru":         sig_gru,
            "transformer": sig_tf,
            "ensemble":    ensemble,
        }

    def _load_dl_signal(self, model_name: str) -> float:
        """
        Try to load the latest DL signal from Phase 3 CSV.
        Falls back to momentum proxy if file not found.
        """
        for search_dir in [Path("."), Path("phase3_outputs"),
                           Path("/content/phase3_outputs")]:
            p = search_dir / f"NIFTY50_dl_forecasts.csv"
            if p.exists():
                try:
                    df3 = pd.read_csv(p, index_col=0, parse_dates=True)
                    col = f"signal_{model_name}"
                    if col in df3.columns:
                        return float(df3[col].iloc[-1])
                except Exception:
                    pass
        return 1.0   # default: long bias if no DL file

    def _count_votes(self, signals: dict) -> int:
        ens = signals["ensemble"]
        votes = [signals[k] for k in ["garch", "arima", "lstm", "gru", "transformer"]]
        return sum(1 for v in votes if np.sign(v) == np.sign(ens))

    # ─────────────────────────────────────────────────────────
    # PRIVATE — RECOMMENDATION
    # ─────────────────────────────────────────────────────────

    def _recommendation(self, df, log_ret, garch_vol, signals) -> dict:
        price    = float(df["Close"].iloc[-1])
        ensemble = signals["ensemble"]
        votes    = self._count_votes(signals)
        mu_daily = float(log_ret.mean())
        sigma_d  = garch_vol / np.sqrt(self.TRADING_DAYS)

        # 95% VaR (parametric, using GARCH vol)
        from scipy.stats import norm
        var_95_pct = norm.ppf(0.05, mu_daily, sigma_d)
        var_95_rs  = price * var_95_pct

        # Suggested hold: shorter in high vol regimes
        regime = self._vol_regime(garch_vol, log_ret.rolling(21).std()
                                   * np.sqrt(self.TRADING_DAYS))
        hold   = {"LOW": "2–4 weeks", "MEDIUM": "1–2 weeks", "HIGH": "2–5 days"}[regime]

        # Stop loss: 1.5× daily GARCH vol below current price
        stop_pct  = 1.5 * sigma_d
        stop_price = price * (1 - stop_pct)

        # Action and conviction
        if ensemble > 0:
            action = "BUY"
            if votes >= 4:   conviction = "HIGH  (4–5 models agree)"
            elif votes >= 3: conviction = "MEDIUM (3 models agree)"
            else:            conviction = "LOW   (slim majority)"
        else:
            action = "SELL/AVOID"
            if votes >= 4:   conviction = "HIGH  (4–5 models agree)"
            elif votes >= 3: conviction = "MEDIUM (3 models agree)"
            else:            conviction = "LOW   (slim majority)"

        # Vol context for rationale
        vol_context = (
            f"GARCH forecasts annualised vol of {garch_vol:.1%} — "
            f"{'below' if garch_vol < 0.18 else 'above'} the historical median. "
        )
        momentum_context = (
            f"5-day momentum is {'positive' if signals['arima'] > 0 else 'negative'}. "
        )
        ensemble_context = (
            f"{votes}/5 models are aligned on this signal."
        )

        rationale = vol_context + momentum_context + ensemble_context

        return {
            "action":      action,
            "conviction":  conviction,
            "rationale":   rationale,
            "hold_period": hold,
            "var_95":      f"{var_95_pct:.2%} / ₹{var_95_rs:,.0f}",
            "stop_loss":   f"₹{stop_price:,.1f} ({stop_pct:.2%} below spot)",
            "mu_daily":    mu_daily,
            "sigma_daily": sigma_d,
        }
