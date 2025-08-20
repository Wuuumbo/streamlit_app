# =============================
# File: streamlit_app.py
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Asset Pair Risk & Volatility Dashboard",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        out = data[["Close"]].rename(columns={"Close": ticker}).dropna()
        out.index = pd.to_datetime(out.index)
        return out
    except Exception:
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


def annualize_vol(returns: pd.Series, interval: str) -> float:
    # Trading periods per year by interval
    freq_map = {
        "1d": 252,
        "1h": 252 * 6.5,  # US market hours approx
        "30m": 252 * 6.5 * 2,
        "15m": 252 * 6.5 * 4,
        "1wk": 52,
        "1mo": 12,
    }
    periods = freq_map.get(interval, 252)
    return returns.std(ddof=1) * np.sqrt(periods)


def drawdown(prices: pd.Series) -> pd.DataFrame:
    cum_max = prices.cummax()
    dd = prices / cum_max - 1.0
    return pd.DataFrame({"price": prices, "cum_max": cum_max, "drawdown": dd})


def hist_var(series: pd.Series, alpha: float = 0.95) -> tuple:
    series = series.dropna()
    if series.empty:
        return np.nan, np.nan
    var = np.quantile(series, 1 - alpha)
    cvar = series[series <= var].mean() if (series <= var).any() else np.nan
    return var, cvar


def rolling_beta(x: pd.Series, y: pd.Series, window: int = 60) -> pd.Series:
    # beta of y relative to x
    def _beta(a, b):
        if np.isclose(a.var(), 0.0):
            return np.nan
        return np.cov(a, b)[0, 1] / np.var(a)

    return (
        pd.concat([x, y], axis=1).rolling(window=window).apply(lambda w: _beta(w[:, 0], w[:, 1]), raw=True)
    )

# ---------------------
# Sidebar inputs
# ---------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Choisissez n'importe quels tickers Yahoo Finance : actions, ETF, indices (^GSPC), forex (EURUSD=X), crypto (BTC-USD).")

    t1 = st.text_input("Ticker 1", "BTC-USD").strip()
    t2 = st.text_input("Ticker 2", "ETH-USD").strip()

    default_start = (datetime.utcnow() - timedelta(days=365 * 2)).date()
    start = st.date_input("Début", default_start)
    end = st.date_input("Fin", datetime.utcnow().date())

    interval = st.selectbox("Intervalle", ["1d", "1h", "30m", "15m", "1wk", "1mo"], index=0)
    logret = st.checkbox("Rendements logarithmiques", value=False)

    win_corr = st.slider("Fenêtre corrélation/volat (jours ou barres)", min_value=20, max_value=252, value=90, step=5)
    rf = st.number_input("Taux sans risque (annualisé, %)", value=0.0, step=0.25) / 100.0

    st.markdown("---")
    st.caption("Exemples : AAPL, MSFT, SPY, ^GSPC, EURUSD=X, EURGBP=X, BTC-USD, ETH-USD, GLD, CL=F (WTI)")

# ---------------------
# Data
# ---------------------
start_str, end_str = str(start), str(end)
prices1 = fetch_prices(t1, start_str, end_str, interval)
prices2 = fetch_prices(t2, start_str, end_str, interval)

if prices1.empty or prices2.empty:
    st.warning("Impossible de charger les données pour au moins un des tickers. Vérifiez les symboles et l'intervalle.")
    st.stop()

prices = prices1.join(prices2, how="inner").dropna()
prices.columns = [t1, t2]

# Normalized prices
norm = prices / prices.iloc[0] * 100.0

# Returns
rets = compute_returns(prices, log=logret)
ret1, ret2 = rets[t1], rets[t2]

# Key stats
vol1 = annualize_vol(ret1, interval)
vol2 = annualize_vol(ret2, interval)

mean1 = ret1.mean()
mean2 = ret2.mean()

# Annualized Sharpe (approx)
# Convert mean/vol by annualization consistent with interval
freq_map = {"1d": 252, "1h": 252 * 6.5, "30m": 252 * 6.5 * 2, "15m": 252 * 6.5 * 4, "1wk": 52, "1mo": 12}
periods = freq_map.get(interval, 252)
ann_mean1 = mean1 * periods
ann_mean2 = mean2 * periods

sharpe1 = (ann_mean1 - rf) / vol1 if np.isfinite(vol1) and vol1 > 0 else np.nan
sharpe2 = (ann_mean2 - rf) / vol2 if np.isfinite(vol2) and vol2 > 0 else np.nan

# Correlations
corr_full = ret1.corr(ret2)
roll_corr = rets[t1].rolling(win_corr).corr(rets[t2])

# Rolling vols
roll_vol1 = ret1.rolling(win_corr).std() * np.sqrt(periods)
roll_vol2 = ret2.rolling(win_corr).std() * np.sqrt(periods)

# Beta of t2 vs t1 on rolling window
roll_beta = rolling_beta(ret1, ret2, window=win_corr)

# VaR & CVaR (historical)
var1, cvar1 = hist_var(ret1, alpha=0.95)
var2, cvar2 = hist_var(ret2, alpha=0.95)

# Cointegration (prices)
try:
    score, pvalue, _ = coint(prices[t1].dropna(), prices[t2].dropna())
except Exception:
    pvalue = np.nan

# ---------------------
# Header & KPI
# ---------------------
st.title("📊 Asset Pair Risk & Volatility Dashboard")
left, mid, right = st.columns([2, 2, 3])
with left:
    st.metric(f"Corrélation (toutes périodes)", f"{corr_full:.3f}")
    st.metric(f"Volatilité {t1} (ann.)", f"{vol1:.2%}")
    st.metric(f"Volatilité {t2} (ann.)", f"{vol2:.2%}")
with mid:
    st.metric(f"Sharpe {t1}", f"{sharpe1:.2f}")
    st.metric(f"Sharpe {t2}", f"{sharpe2:.2f}")
    st.metric("coint. p-value", f"{pvalue:.3f}" if np.isfinite(pvalue) else "N/A")
with right:
    st.write("**VaR(95%) & CVaR(95%) (quotidien/barre)**")
    st.write(pd.DataFrame({
        "Asset": [t1, t2],
        "VaR 95%": [var1, var2],
        "CVaR 95%": [cvar1, cvar2],
    }).set_index("Asset"))

# ---------------------
# Charts
# ---------------------
st.subheader("Prix normalisés (base 100)")
fig_prices = px.line(norm, labels={"value": "Index (100=Début)", "index": "Date"})
fig_prices.update_layout(legend_title_text="Asset")
st.plotly_chart(fig_prices, use_container_width=True)

st.subheader("Corrélation glissante")
fig_corr = px.line(roll_corr.dropna(), labels={"value": "Corrélation", "index": "Date"})
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Volatilité glissante (annualisée)")
fig_rv = go.Figure()
fig_rv.add_trace(go.Scatter(x=roll_vol1.index, y=roll_vol1, name=f"Vol {t1}"))
fig_rv.add_trace(go.Scatter(x=roll_vol2.index, y=roll_vol2, name=f"Vol {t2}"))
fig_rv.update_layout(yaxis_title="Vol ann.")
st.plotly_chart(fig_rv, use_container_width=True)

st.subheader("Dispersion des rendements (nuage de points)")
scatter_df = pd.DataFrame({t1: ret1, t2: ret2}).dropna()
fig_scatter = px.scatter(scatter_df, x=t1, y=t2, trendline="ols")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Drawdowns")
colA, colB = st.columns(2)
with colA:
    dd1 = drawdown(prices[t1])
    fig_dd1 = px.area(dd1, x=dd1.index, y="drawdown", title=f"Drawdown {t1}")
    st.plotly_chart(fig_dd1, use_container_width=True)
with colB:
    dd2 = drawdown(prices[t2])
    fig_dd2 = px.area(dd2, x=dd2.index, y="drawdown", title=f"Drawdown {t2}")
    st.plotly_chart(fig_dd2, use_container_width=True)

# Spread & z-score for potential pairs trading (if cointegrated)
if np.isfinite(pvalue) and pvalue < 0.05:
    st.subheader("Spread & Z-Score (pairs trading)")
    # Hedge ratio via linear regression on prices
    x = prices[t1].values.reshape(-1, 1)
    y = prices[t2].values
    # Simple OLS: beta = cov/var
    beta = np.cov(prices[t1], prices[t2])[0, 1] / np.var(prices[t1]) if np.var(prices[t1]) > 0 else 1.0
    spread = prices[t2] - beta * prices[t1]
    zscore = (spread - spread.rolling(win_corr).mean()) / spread.rolling(win_corr).std()
    fig_spread = px.line(pd.DataFrame({"Spread": spread, "Z-Score": zscore}).dropna())
    st.plotly_chart(fig_spread, use_container_width=True)
    st.caption("p-value < 0.05 ⇒ preuve statistique de cointégration (Engle–Granger).")
else:
    st.info("Pas de cointégration statistiquement significative (p ≥ 0.05) — pairs trading non recommandé.")

# ---------------------
# Data table & export
# ---------------------
with st.expander("Voir les données (prix & rendements)"):
    st.dataframe(pd.concat({"Prices": prices, "Returns": rets}, axis=1))

csv = pd.concat({"Prices": prices, "Returns": rets}, axis=1).to_csv().encode("utf-8")
st.download_button("📥 Télécharger CSV (prix & rendements)", data=csv, file_name="pair_data.csv", mime="text/csv")

st.caption("Source: Yahoo Finance via yfinance (gratuit, sans clé API).")

# =============================
# File: requirements.txt
# =============================
# streamlit
# yfinance
# pandas
# numpy
# plotly
# scipy
# statsmodels

# =============================
# File: README.md
# =============================
# Asset Pair Risk & Volatility Dashboard

Une application Streamlit gratuite pour analyser **n'importe quelle paire d'actifs** (actions, ETF, indices, forex, crypto) :
- Prix normalisés, rendements, corrélation glissante
- Volatilités (ann.), VaR/CVaR historiques
- Sharpe (approx), bêta roulant, drawdowns
- Test de **cointégration** (Engle–Granger) et spread/z-score si p<0.05

## Utilisation (local)
1. Installer Python 3.10+
2. Créer un environnement puis :
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Déploiement gratuit (Streamlit Community Cloud)
1. Créez un dépôt GitHub avec **streamlit_app.py** et **requirements.txt**.
2. Allez sur [Streamlit Community Cloud] et "Deploy an app".
3. Sélectionnez le dépôt, le fichier `streamlit_app.py`, et déployez.
4. Vous obtiendrez une URL publique à partager aux recruteurs.

## Conseils tickers Yahoo
- Indices : `^GSPC` (S&P 500), `^NDX`, `^FCHI` (CAC 40)
- Forex : `EURUSD=X`, `EURGBP=X`
- Crypto : `BTC-USD`, `ETH-USD`
- Matières premières : `GC=F` (Or), `CL=F` (WTI)

## Notes
- Les intervalles intraday (1h/30m/15m) peuvent être limités sur longues périodes par Yahoo.
- Si un ticker ne renvoie rien, essayez un autre symbole ou passez à `1d`.
