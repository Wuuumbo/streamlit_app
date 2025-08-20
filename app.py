import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import statsmodels.api as sm
from datetime import date, timedelta
from scipy.stats import norm
from statsmodels.tsa.stattools import coint

# ============ CONFIG ============ #
st.set_page_config(
    page_title="Asset Risk & Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

TRADING_DAYS = 252

# ============ FONCTIONS ============ #
def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    # Normaliser pour avoir des colonnes simples "Close"
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.loc[:, pd.IndexSlice[:, "Close"]]
            df.columns = [c[0] for c in df.columns]
        except Exception:
            # Fallback : tenter "Close" direct
            if "Close" in df.columns:
                df = df["Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    return df.sort_index()

def compute_returns(prices, log=False):
    if log:
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()

def annualize_vol(returns):
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns, rf_annual=0.0):
    if returns.dropna().empty:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess_daily = returns.mean() - rf_daily
    vol_daily = returns.std(ddof=1)
    if vol_daily == 0 or np.isnan(vol_daily):
        return np.nan
    return float(excess_daily / vol_daily * np.sqrt(TRADING_DAYS))

def sortino_ratio(returns, rf_annual=0.0):
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    downside = returns[returns < 0]
    if downside.dropna().empty:
        return np.nan
    downside_vol = downside.std(ddof=1)
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan
    mean_excess = returns.mean() - rf_daily
    return float(mean_excess / downside_vol * np.sqrt(TRADING_DAYS))

def max_drawdown(prices):
    peak = prices.cummax()
    dd = prices / peak - 1.0
    return dd.min()

def rolling_volatility(returns, window=30):
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)

def rolling_correlation(r1, r2, window=60):
    return r1.rolling(window).corr(r2)

def beta_alpha(ret_x, ret_y):
    df = pd.concat([ret_x, ret_y], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df.iloc[:, 0])
    y = df.iloc[:, 1]
    model = sm.OLS(y, X).fit()
    beta = float(model.params.iloc[1])
    alpha = float(model.params.iloc[0])
    r2 = float(model.rsquared)
    return beta, alpha, r2

def var_es_hist(returns, q=0.95):
    r = returns.dropna().values
    if r.size == 0:
        return np.nan, np.nan
    # VaR = quantile côté pertes ; ES = moyenne des pertes au-delà du quantile
    var_q = np.quantile(r, 1 - q)
    var_loss = -float(var_q)
    es_loss = -float(r[r <= var_q].mean()) if (r <= var_q).any() else np.nan
    return var_loss, es_loss

def var_norm(returns, q=0.95):
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = norm.ppf(1 - q)  # < 0
    return -(mu + z * sigma)

# ============ SIDEBAR ============ #
st.sidebar.title("⚙️ Paramètres")

ticker_a = st.sidebar.text_input("Ticker A", "AAPL")
ticker_b = st.sidebar.text_input("Ticker B", "MSFT")

today = date.today()
start_date = st.sidebar.date_input("Date début", value=today - timedelta(days=365*2))
end_date = st.sidebar.date_input("Date fin", value=today)

log_returns = st.sidebar.checkbox("Rendements logarithmiques", False)
rf_rate = st.sidebar.number_input("Taux sans risque annualisé (%)", 0.0, step=0.25) / 100.0
win_vol = st.sidebar.slider("Fenêtre vol (jours)", 10, 120, 30, step=5)
win_corr = st.sidebar.slider("Fenêtre corr (jours)", 20, 180, 60, step=5)

st.sidebar.caption("Source des données : Yahoo! Finance via yfinance (gratuit).")

# ============ DATA ============ #
@st.cache_data(show_spinner=True)
def _load_data(a, b, start, end):
    prices_df = download_prices([a, b], str(start), str(end))
    if prices_df is None or prices_df.empty:
        return None, None
    prices_df = prices_df.ffill().dropna()
    returns_df = compute_returns(prices_df, log_returns).dropna()
    return prices_df, returns_df

prices, returns = _load_data(ticker_a.strip(), ticker_b.strip(), start_date, end_date)

st.title("📊 Asset Risk & Volatility Dashboard (2 actifs)")

if prices is None or returns is None or prices.empty or returns.empty:
    st.warning("Impossible de récupérer les données. Vérifie les tickers et la période.")
    st.stop()

# ============ KPIs ============ #
corr = returns[[ticker_a, ticker_b]].corr().iloc[0, 1]
beta, alpha, r2 = beta_alpha(returns[ticker_a], returns[ticker_b])
try:
    coint_p = float(coint(prices[ticker_a], prices[ticker_b])[1])
except Exception:
    coint_p = np.nan

def metrics_for(prices_s, rets_s):
    # CAGR annualisé (approx) basé sur 252 jours de bourse
    if len(prices_s) >= 2:
        cagr = (prices_s.iloc[-1] / prices_s.iloc[0]) ** (TRADING_DAYS / len(prices_s)) - 1
    else:
        cagr = np.nan
    var95, es95 = var_es_hist(rets_s, 0.95)
    return {
        "Dernier prix": float(prices_s.iloc[-1]),
        "CAGR": cagr,
        "Vol (ann.)": float(annualize_vol(rets_s)),
        "Sharpe": float(sharpe_ratio(rets_s, rf_rate)),
        "Sortino": float(sortino_ratio(rets_s, rf_rate)),
        "Max Drawdown": float(max_drawdown(prices_s)),
        "VaR 95% (J)": var95,
        "ES 95% (J)": es95,
        "VaR Norm 95% (J)": float(var_norm(rets_s, 0.95)),
    }

mA = metrics_for(prices[ticker_a], returns[ticker_a])
mB = metrics_for(prices[ticker_b], returns[ticker_b])

k = st.columns(4)
k[0].metric("Corrélation", f"{corr:.3f}")
k[1].metric("β (B~A)", f"{beta:.3f}")
k[2].metric("α (quotidien)", f"{alpha:.5f}")
k[3].metric("R²", f"{r2:.3f}")
st.caption(f"Cointégration (Engle–Granger) p-value = **{coint_p:.3f}**  (p < 0.05 ⇒ cointégration plausible).")

table = pd.DataFrame([mA, mB], index=[ticker_a, ticker_b])
st.subheader("Indicateurs clés")
st.dataframe(table, use_container_width=True)

# ============ GRAPHS ============ #
st.subheader("Prix indexés (base = 100)")
idx = prices / prices.iloc[0] * 100.0
fig_idx = px.line(idx, x=idx.index, y=idx.columns, labels={"value": "Indice", "variable": "Actif", "index": "Date"})
st.plotly_chart(fig_idx, use_container_width=True, theme="streamlit")

st.subheader("Volatilité glissante (annualisée)")
roll_vol = rolling_volatility(returns, window=win_vol)
fig_rv = px.line(roll_vol, x=roll_vol.index, y=roll_vol.columns, labels={"value": "Vol ann.", "variable": "Actif", "index": "Date"})
st.plotly_chart(fig_rv, use_container_width=True, theme="streamlit")

st.subheader("Corrélation glissante")
roll_corr = rolling_correlation(returns[ticker_a], returns[ticker_b], window=win_corr).to_frame("Corrélation")
fig_rc = px.line(roll_corr, x=roll_corr.index, y="Corrélation", labels={"value": "Corrélation", "index": "Date"})
st.plotly_chart(fig_rc, use_container_width=True, theme="streamlit")

st.subheader("Dispersion des rendements (B vs A) + droite de régression")
xy = returns[[ticker_a, ticker_b]].dropna().copy()
X = sm.add_constant(xy[ticker_a])
model = sm.OLS(xy[ticker_b], X).fit()
xy["pred"] = model.predict(X)
fig_sc = px.scatter(xy, x=ticker_a, y=ticker_b, opacity=0.6, labels={"x": ticker_a, "y": ticker_b})
fig_sc.add_traces(px.line(xy.sort_values(ticker_a), x=ticker_a, y="pred").data)
st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")

# ============ EXPORTS ============ #
st.subheader("Téléchargements")
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Prix (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv", mime="text/csv")
with c2:
    st.download_button("📥 Rendements (CSV)", data=returns.to_csv().encode("utf-8"), file_name="returns.csv", mime="text/csv")

st.caption("⚠️ Outil pédagogique — pas un conseil en investissement.")
