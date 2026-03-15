"""
╔══════════════════════════════════════════════════════════════╗
║         NIFTY QUANT INTELLIGENCE — Streamlit App             ║
║  Live signal dashboard powered by GARCH + DL engine          ║
╚══════════════════════════════════════════════════════════════╝

Deploy:
    streamlit run app.py

Or on Streamlit Cloud:
    1. Push this folder to a GitHub repo
    2. Go to share.streamlit.io → New app → point to app.py
    3. Add secrets in Streamlit Cloud dashboard (see secrets.toml.example)

Folder structure:
    app.py              ← this file
    engine.py           ← signal computation engine
    bs_pricer.py        ← Black-Scholes pricer (from Phase 4)
    requirements.txt    ← pip dependencies
    .streamlit/
        config.toml     ← theme config
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from engine import QuantEngine
from bs_pricer import BSOption, implied_volatility

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nifty Quant Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark terminal-meets-Bloomberg aesthetic
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e17;
    color: #c8d0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: #7a8ba8 !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1628 0%, #111c2e 100%);
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent, #00d4aa);
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6080;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8f0ff;
    line-height: 1;
}
.metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.3rem;
}
.delta-up   { color: #00d4aa; }
.delta-down { color: #ff4d6a; }
.delta-flat { color: #7a8ba8; }

/* Signal badge */
.signal-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.signal-long  { background: #003d2e; color: #00d4aa; border: 1px solid #00d4aa40; }
.signal-short { background: #3d0010; color: #ff4d6a; border: 1px solid #ff4d6a40; }
.signal-flat  { background: #1e2d4a; color: #7a8ba8; border: 1px solid #2a3d5a; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2a5080;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Recommendation box */
.rec-box {
    background: #0d1628;
    border: 1px solid #1e3a5a;
    border-radius: 8px;
    padding: 1.4rem;
    margin: 0.5rem 0;
}
.rec-box h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: #00d4aa;
    margin: 0 0 0.8rem 0;
    text-transform: uppercase;
}
.rec-box p {
    font-size: 0.9rem;
    color: #a0b0c8;
    line-height: 1.6;
    margin: 0;
}

/* Ticker chip */
.ticker-chip {
    display: inline-block;
    background: #111c2e;
    border: 1px solid #1e3a5a;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #5a9fd4;
    margin: 0.15rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Plotly chart bg */
.js-plotly-plot { border-radius: 8px; }

/* Disclaimer */
.disclaimer {
    background: #0a0e17;
    border: 1px solid #1e2020;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.72rem;
    color: #3a4a5a;
    line-height: 1.5;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
                    font-weight:600; color:#00d4aa; letter-spacing:0.05em;">
            NIFTY QUANT
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#2a5080; letter-spacing:0.2em; text-transform:uppercase;
                    margin-top:0.2rem;">
            Intelligence Terminal
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Universe</div>', unsafe_allow_html=True)

    TICKERS = {
        "Nifty 50 Index":     "^NSEI",
        "Reliance":           "RELIANCE.NS",
        "TCS":                "TCS.NS",
        "HDFC Bank":          "HDFCBANK.NS",
        "Infosys":            "INFY.NS",
        "ICICI Bank":         "ICICIBANK.NS",
        "Kotak Mahindra":     "KOTAKBANK.NS",
        "Bajaj Finance":      "BAJFINANCE.NS",
        "Wipro":              "WIPRO.NS",
        "Maruti Suzuki":      "MARUTI.NS",
        "Asian Paints":       "ASIANPAINT.NS",
        "Titan Company":      "TITAN.NS",
    }

    selected_name = st.selectbox(
        "Select instrument",
        list(TICKERS.keys()),
        index=0,
    )
    ticker = TICKERS[selected_name]

    st.markdown('<div class="section-header">Lookback</div>', unsafe_allow_html=True)
    lookback_years = st.slider("Years of history", 2, 13, 5)

    st.markdown('<div class="section-header">Options</div>', unsafe_allow_html=True)
    show_options = st.checkbox("Show options analysis", value=True)
    rfr = st.slider("Risk-free rate (%)", 4.0, 9.0, 6.5, 0.25) / 100

    st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
    show_garch  = st.checkbox("GARCH vol forecast", value=True)
    show_lstm   = st.checkbox("LSTM signal",         value=True)
    show_transformer = st.checkbox("Transformer signal", value=True)

    st.markdown("---")
    run_btn = st.button("⚡  Run Analysis", use_container_width=True, type="primary")

    st.markdown("""
    <div style="margin-top:2rem; font-family:'IBM Plex Mono',monospace;
                font-size:0.6rem; color:#1e3050; line-height:1.6;">
        Built on GARCH + LSTM + Transformer<br>
        NSE/BSE data via yfinance<br>
        Phase 1–5 Hybrid Quant Engine
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

col_title, col_time = st.columns([4, 1])
with col_title:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.2em; color:#2a5080; text-transform:uppercase;">
            Hybrid Quant Engine — Live Signal Dashboard
        </div>
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.8rem;
                    font-weight:600; color:#e8f0ff; margin-top:0.3rem;">
            {selected_name}
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                         color:#2a5080; margin-left:0.6rem;">{ticker}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style="text-align:right; padding-top:1.2rem;
                font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#2a5080;">
        {datetime.now().strftime('%d %b %Y')}<br>
        {datetime.now().strftime('%H:%M IST')}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LOGIC — cached engine run
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_analysis(ticker, lookback_years, rfr):
    engine = QuantEngine(ticker, lookback_years=lookback_years, rfr=rfr)
    return engine.run()


if run_btn or "analysis" not in st.session_state:
    with st.spinner(f"Fetching {selected_name} data and running models..."):
        try:
            result = get_analysis(ticker, lookback_years, rfr)
            st.session_state["analysis"] = result
            st.session_state["ticker"]   = ticker
            st.session_state["name"]     = selected_name
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

result = st.session_state.get("analysis")
if result is None:
    st.info("Select an instrument and click **Run Analysis** to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# ROW 1 — KEY METRICS
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Current Signals</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(label, value, delta=None, accent="#00d4aa"):
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta > 0 else ("delta-down" if delta < 0 else "delta-flat")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        delta_html = f'<div class="metric-delta {cls}">{arrow} {abs(delta):.2f}%</div>'
    return f"""
    <div class="metric-card" style="--accent:{accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# Current price
price     = result["current_price"]
price_chg = result["price_change_pct"]
price_color = "#00d4aa" if price_chg >= 0 else "#ff4d6a"

with c1:
    st.markdown(metric_card(
        "Last Price", f"₹{price:,.1f}",
        delta=price_chg, accent=price_color
    ), unsafe_allow_html=True)

# GARCH vol
garch_vol = result["garch_vol"]
hist_vol  = result["hist_vol_21d"]
vol_delta = (garch_vol - hist_vol) / hist_vol * 100 if hist_vol else 0
vol_color = "#ff8c42" if garch_vol > 0.25 else ("#00d4aa" if garch_vol < 0.15 else "#5a9fd4")

with c2:
    st.markdown(metric_card(
        "GARCH Vol (ann.)", f"{garch_vol:.1%}",
        delta=vol_delta, accent=vol_color
    ), unsafe_allow_html=True)

# Vol regime
regime       = result["vol_regime"]
regime_color = {"LOW": "#00d4aa", "MEDIUM": "#5a9fd4", "HIGH": "#ff4d6a"}.get(regime, "#7a8ba8")
with c3:
    st.markdown(metric_card(
        "Vol Regime", regime, accent=regime_color
    ), unsafe_allow_html=True)

# E[r] and Var[r]
mu    = result["expected_return_daily"]
var_r = result["variance_daily"]
with c4:
    st.markdown(metric_card(
        "E[r] daily", f"{mu*100:+.3f}%",
        accent="#5a9fd4"
    ), unsafe_allow_html=True)

with c5:
    st.markdown(metric_card(
        "Var[r] daily", f"{var_r*1e4:.2f} ×10⁻⁴",
        accent="#8e6fd4"
    ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROW 2 — SIGNAL SUMMARY
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Model Signals</div>', unsafe_allow_html=True)

sig_cols = st.columns(4)
signals  = result["signals"]

def signal_html(name, sig, confidence=None):
    label = "LONG" if sig > 0 else ("SHORT / FLAT" if sig < 0 else "NEUTRAL")
    cls   = "signal-long" if sig > 0 else ("signal-short" if sig < 0 else "signal-flat")
    conf_str = f"<br><small style='color:#4a6080;font-size:0.7rem;'>{confidence}</small>" if confidence else ""
    return f"""
    <div style="text-align:center; padding:0.8rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    letter-spacing:0.15em; color:#2a5080; margin-bottom:0.5rem;
                    text-transform:uppercase;">{name}</div>
        <span class="signal-badge {cls}">{label}</span>
        {conf_str}
    </div>
    """

with sig_cols[0]:
    st.markdown(signal_html(
        "GARCH Regime",
        signals["garch"],
        f"Vol {garch_vol:.1%} vs 75th pct"
    ), unsafe_allow_html=True)

with sig_cols[1]:
    st.markdown(signal_html(
        "ARIMA (5d mom)",
        signals["arima"],
        f"5d mean return: {result['mom_5d']*100:+.3f}%"
    ), unsafe_allow_html=True)

with sig_cols[2]:
    lstm_dir = signals.get("lstm", 1)
    st.markdown(signal_html(
        "LSTM",
        lstm_dir,
        "Phase 3 DL model"
    ), unsafe_allow_html=True)

with sig_cols[3]:
    ensemble = signals["ensemble"]
    ens_conf = f"{result['ensemble_votes']}/4 models agree"
    st.markdown(signal_html("Ensemble", ensemble, ens_conf), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROW 3 — RECOMMENDATION BOX
# ─────────────────────────────────────────────────────────────

rec = result["recommendation"]
rec_color = "#00d4aa" if rec["action"] == "BUY" else ("#ff4d6a" if rec["action"] == "SELL/AVOID" else "#5a9fd4")

st.markdown(f"""
<div class="rec-box" style="border-color:{rec_color}40; border-left: 3px solid {rec_color};">
    <h4>📋 Recommendation — {selected_name}</h4>
    <div style="display:flex; gap:2rem; align-items:flex-start;">
        <div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem;
                        font-weight:600; color:{rec_color};">{rec['action']}</div>
            <div style="font-size:0.75rem; color:#4a6080; margin-top:0.2rem;">Conviction</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                        color:#a0b0c8;">{rec['conviction']}</div>
        </div>
        <div style="flex:1; border-left:1px solid #1e2d4a; padding-left:2rem;">
            <p>{rec['rationale']}</p>
        </div>
        <div style="min-width:180px; border-left:1px solid #1e2d4a; padding-left:2rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#2a5080; letter-spacing:0.1em;">SUGGESTED HOLD</div>
            <div style="font-family:'IBM Plex Mono',monospace; color:#a0b0c8;
                        margin-top:0.3rem;">{rec['hold_period']}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#2a5080; letter-spacing:0.1em; margin-top:0.8rem;">95% VaR (1d)</div>
            <div style="font-family:'IBM Plex Mono',monospace; color:#ff4d6a;
                        margin-top:0.3rem;">{rec['var_95']}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#2a5080; letter-spacing:0.1em; margin-top:0.8rem;">STOP LOSS</div>
            <div style="font-family:'IBM Plex Mono',monospace; color:#ff8c42;
                        margin-top:0.3rem;">{rec['stop_loss']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROW 4 — CHARTS
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Price & Volatility</div>', unsafe_allow_html=True)

df = result["price_df"]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.03,
    subplot_titles=("Price", "GARCH Conditional Vol (ann.)", "Log Returns"),
)

# Price
fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    line=dict(color="#5a9fd4", width=1.5),
    name="Price", fill="tozeroy",
    fillcolor="rgba(90,159,212,0.05)",
), row=1, col=1)

# Vol regime shading
vol_series = result["cond_vol_series"]
for regime_name, color in [("LOW","rgba(0,212,170,0.1)"),
                             ("HIGH","rgba(255,77,106,0.1)")]:
    mask_col = f"regime_{regime_name}"
    if mask_col in result:
        fig.add_trace(go.Scatter(
            x=result[mask_col].index,
            y=result[mask_col].values,
            fill="tozeroy", fillcolor=color,
            line=dict(width=0), showlegend=False,
            name=f"{regime_name} vol",
        ), row=2, col=1)

fig.add_trace(go.Scatter(
    x=vol_series.index, y=vol_series.values,
    line=dict(color="#ff8c42", width=1.2),
    name="GARCH vol",
), row=2, col=1)

# Returns
log_ret = result["log_returns"]
colors  = ["#00d4aa" if r >= 0 else "#ff4d6a" for r in log_ret]
fig.add_trace(go.Bar(
    x=log_ret.index, y=log_ret.values,
    marker_color=colors, name="Log return",
    marker_line_width=0,
), row=3, col=1)

fig.update_layout(
    height=600,
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1220",
    font=dict(family="IBM Plex Mono", color="#4a6080", size=10),
    showlegend=False,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis3=dict(showgrid=False, zeroline=False,
                tickfont=dict(size=9), linecolor="#1e2d4a"),
)
for i in range(1, 4):
    fig.update_xaxes(showgrid=False, zeroline=False,
                     linecolor="#1e2d4a", row=i, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#0f1a2e",
                     zeroline=False, linecolor="#1e2d4a", row=i, col=1)

fig["layout"]["yaxis2"]["tickformat"] = ".0%"

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# ROW 5 — RETURN DISTRIBUTION + VOLATILITY CONE
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Distribution & Vol Cone</div>',
            unsafe_allow_html=True)

col_dist, col_cone = st.columns(2)

with col_dist:
    from scipy.stats import norm as scipy_norm
    lr     = result["log_returns"].dropna()
    mu_r, sigma_r = lr.mean(), lr.std()
    x_range = np.linspace(lr.quantile(0.001), lr.quantile(0.999), 300)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=lr, nbinsx=80, histnorm="probability density",
        marker_color="#1e3a5a", marker_line_width=0,
        name="Empirical",
    ))
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=scipy_norm.pdf(x_range, mu_r, sigma_r),
        line=dict(color="#ff4d6a", width=1.5, dash="dash"),
        name="Normal fit",
    ))
    var95_line = np.percentile(lr, 5)
    fig_dist.add_vline(x=var95_line, line_color="#ff4d6a",
                       line_dash="dot", line_width=1,
                       annotation_text=f"95% VaR: {var95_line:.3f}",
                       annotation_font_color="#ff4d6a",
                       annotation_font_size=9)
    fig_dist.update_layout(
        title=dict(text="Return Distribution", font=dict(size=11, color="#4a6080")),
        height=300, paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
        font=dict(family="IBM Plex Mono", color="#4a6080", size=9),
        showlegend=False, margin=dict(l=0,r=0,t=35,b=0),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#1e2d4a"),
        yaxis=dict(showgrid=True, gridcolor="#0f1a2e", zeroline=False),
    )
    st.plotly_chart(fig_dist, use_container_width=True,
                    config={"displayModeBar": False})

with col_cone:
    # Volatility cone: realised vol at different horizons
    windows  = [5, 10, 21, 42, 63, 126]
    rv_data  = {}
    for w in windows:
        rv = lr.rolling(w).std() * np.sqrt(252)
        rv_data[w] = {"min": rv.quantile(0.10), "med": rv.median(),
                      "max": rv.quantile(0.90), "cur": rv.iloc[-1]}

    fig_cone = go.Figure()
    fig_cone.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=[rv_data[w]["max"] for w in windows] + [rv_data[w]["min"] for w in windows[::-1]],
        fill="toself", fillcolor="rgba(90,159,212,0.08)",
        line=dict(width=0), name="10–90th pct",
    ))
    fig_cone.add_trace(go.Scatter(
        x=windows, y=[rv_data[w]["med"] for w in windows],
        line=dict(color="#5a9fd4", width=1.5, dash="dash"),
        name="Median",
    ))
    fig_cone.add_trace(go.Scatter(
        x=windows, y=[rv_data[w]["cur"] for w in windows],
        line=dict(color="#00d4aa", width=2),
        mode="lines+markers",
        marker=dict(size=6, color="#00d4aa"),
        name="Current",
    ))
    fig_cone.add_hline(y=garch_vol, line_color="#ff8c42",
                       line_dash="dot", line_width=1,
                       annotation_text=f"GARCH: {garch_vol:.1%}",
                       annotation_font_color="#ff8c42",
                       annotation_font_size=9)
    fig_cone.update_layout(
        title=dict(text="Volatility Cone", font=dict(size=11, color="#4a6080")),
        height=300, paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
        font=dict(family="IBM Plex Mono", color="#4a6080", size=9),
        showlegend=False, margin=dict(l=0,r=0,t=35,b=0),
        xaxis=dict(title="Horizon (days)", showgrid=False,
                   zeroline=False, linecolor="#1e2d4a"),
        yaxis=dict(showgrid=True, gridcolor="#0f1a2e", zeroline=False,
                   tickformat=".0%"),
    )
    st.plotly_chart(fig_cone, use_container_width=True,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# ROW 6 — OPTIONS ANALYSIS (conditional)
# ─────────────────────────────────────────────────────────────

if show_options and ticker in ["^NSEI", "NIFTY50"]:
    st.markdown('<div class="section-header">Options Misprice Scanner</div>',
                unsafe_allow_html=True)

    market_iv_premium = st.slider(
        "Simulate market IV premium over model (vol pts)", 0, 10, 5
    ) / 100
    market_iv = garch_vol + market_iv_premium

    spot        = price
    expiries    = [7, 14, 30, 45]
    moneyness   = np.arange(0.92, 1.09, 0.02)
    misprice_rows = []

    for dte in expiries:
        T = dte / 365
        for m in moneyness:
            K = round(spot * m, -2)
            for opt_type in ["call", "put"]:
                skew_adj = -0.08 * np.log(m)
                mkt_iv   = max(market_iv + skew_adj + np.random.normal(0, 0.003), 0.01)
                opt      = BSOption(spot, K, T, rfr, 0.013, garch_vol, opt_type)
                iv_diff  = mkt_iv - garch_vol
                mispriced = abs(iv_diff) > 0.02
                signal_str = ("SELL" if iv_diff > 0 else "BUY") if mispriced else "HOLD"
                misprice_rows.append({
                    "Type": opt_type.upper(),
                    "Strike": int(K),
                    "DTE": dte,
                    "Market IV": f"{mkt_iv:.1%}",
                    "Model IV": f"{garch_vol:.1%}",
                    "IV Edge": f"{iv_diff:+.1%}",
                    "Vega": f"{opt.vega():.2f}",
                    "Delta": f"{opt.delta():.3f}",
                    "Signal": signal_str,
                })

    df_mp = pd.DataFrame(misprice_rows)
    df_sells = df_mp[df_mp["Signal"] == "SELL"].head(10)
    df_buys  = df_mp[df_mp["Signal"] == "BUY"].head(10)

    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    color:#ff4d6a; letter-spacing:0.1em; margin-bottom:0.5rem;">
            ▼ SELL PREMIUM (market overpriced)
        </div>""", unsafe_allow_html=True)
        st.dataframe(
            df_sells[["Type","Strike","DTE","Market IV","IV Edge","Vega","Delta"]],
            use_container_width=True, hide_index=True,
        )
    with oc2:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    color:#00d4aa; letter-spacing:0.1em; margin-bottom:0.5rem;">
            ▲ BUY VOL (market underpriced)
        </div>""", unsafe_allow_html=True)
        st.dataframe(
            df_buys[["Type","Strike","DTE","Market IV","IV Edge","Vega","Delta"]],
            use_container_width=True, hide_index=True,
        )

elif show_options:
    st.markdown('<div class="section-header">Options Analysis</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="rec-box">
        <h4>Options Pricer</h4>
        <p>Select a stock and adjust the IV premium slider to scan for
        mispriced options using your GARCH volatility forecast as the
        model IV benchmark. Options scanner is most meaningful for
        Nifty 50 index options — switch to <b>Nifty 50 Index</b>
        in the sidebar for full analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # Still show a quick B-S pricer for individual stocks
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        K_input  = st.number_input("Strike (₹)", value=int(price), step=10)
    with bc2:
        T_input  = st.slider("Days to expiry", 7, 90, 30)
    with bc3:
        iv_input = st.slider("Volatility (%)", 5, 100, int(garch_vol*100)) / 100

    call = BSOption(price, K_input, T_input/365, rfr, 0.013, iv_input, "call")
    put  = BSOption(price, K_input, T_input/365, rfr, 0.013, iv_input, "put")

    gc1, gc2, gc3, gc4 = st.columns(4)
    for col, label, val in [
        (gc1, "Call Price",  f"₹{call.price():.2f}"),
        (gc2, "Put Price",   f"₹{put.price():.2f}"),
        (gc3, "Call Delta",  f"{call.delta():.4f}"),
        (gc4, "Vega",        f"₹{call.vega():.2f}"),
    ]:
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    ⚠ RESEARCH TOOL ONLY — Not financial advice. All signals, recommendations,
    and forecasts are generated by statistical models for educational and research
    purposes. Past model performance does not guarantee future results. Options
    trading involves substantial risk of loss. Consult a SEBI-registered advisor
    before making investment decisions. This tool is built on the Hybrid Quant
    Engine (Phases 1–5) using publicly available NSE/BSE data via yfinance.
</div>
""", unsafe_allow_html=True)
