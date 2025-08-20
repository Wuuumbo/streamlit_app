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
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df

def compute_returns(prices, log=False):
    if log:
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()

def annualize_vol(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns, rf=0):
    excess = returns.mean() * TRADING_DAYS - rf
    vol = annualize_vol(returns)
    return excess / vol if vol != 0 else np.nan

def sortino_ratio(returns, rf=0):
    downside = returns[returns < 0]
    if downside.std() == 0: return np.nan
    excess = returns.mean() * TRADING_DAYS - rf
    return excess / (downside.std() * np.sqrt(TRADING_DAYS))

def max_drawdown(prices):
    roll_max = prices.cummax()
    dd = prices / roll_max - 1
    return dd.min()

def rolling_volatility(returns, window=30):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

def rolling_correlation(r1, r2, window=60):
    return r1.rolling(window).corr(r2)

def beta_alpha(ret_x, ret_y):
    df = pd.concat([ret_x, ret_y], axis=1).dropna()
    if df.empty: return np.nan, np.nan, np.nan
    X = sm.add_constant(df.iloc[:,0])
    y = df.iloc[:,1]
    model = sm.OLS(y, X).fit()
    return model.params[1], model.params[0], model.rsquared

def var_es(returns, q=0.95):
    var = -np.percentile(returns.dropna(), (1-q)*100)
    es = -returns[returns <= -var].mean()
    return var, es

def var_norm(returns, q=0.95):
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1-q)
    return -(mu + z*sigma)

# ============ SIDEBAR ============ #
st.sidebar.title("⚙️ Paramètres")

ticker_a = st.sidebar.text_input("Ticker A", "AAPL")
ticker_b = st.sidebar.text_input("Ticker B", "MSFT")

today = date.today()
start_date = st.sidebar.date_input("Date début", value=today - timedelta(days=365*2))
end_date = st.sidebar.date_input("Date fin", value=today)

log_returns = st.sidebar.checkbox("Rendements logarithmiques", False)
rf_rate = st.sidebar.number_input("Taux sans risque annualisé (%)", 0.0, step=0.25) / 100
win_vol = st.sidebar.slider("Fenêtre vol (jours)", 10, 120, 30)
win_corr = st.sidebar.slider("Fenêtre corr (jours)", 20, 180, 60)

# ============ DATA ============ #
prices = download_prices([ticker_a, ticker_b], str(start_date), str(end_date))
prices = prices.dropna()
if prices.empty:
    st.warning("⚠️ Impossible de charger les données")
    st.stop()

returns = compute_returns(prices, log_returns).dropna()

# ============ KPIs ============ #
beta, alpha, r2 = beta_alpha(returns[ticker_a], returns[ticker_b])
corr = returns[[ticker_a, ticker_b]].corr().iloc[0,1]
coint_p = coint(prices[ticker_a], prices[ticker_b])[1]

# Metrics table
def metrics(prices, rets):
    cagr = (prices.iloc[-1]/prices.iloc[0])**(TRADING_DAYS/len(prices))-1
    return {
        "Dernier prix": prices.iloc[-1],
        "CAGR": cagr,
        "Vol": annualize_vol(rets),
        "Sharpe": sharpe_ratio(rets, rf_rate),
        "Sortino": sortino_ratio(rets, rf_rate),
        "MaxDD": max_drawdown(prices),
        "VaR95": var_es(rets,0.95)[0],
        "ES95": var_es(rets,0.95)[1],
        "VaR Norm95": var_norm(rets,0.95)
    }

mA = metrics(prices[ticker_a], returns[ticker_a])
mB = metrics(prices[ticker_b], returns[ticker_b])

st.title("📊 Asset Risk & Volatility Dashboard")

col = st.columns(4)
col[0].metric("Corrélation", f"{corr:.3f}")
col[1].metric("β", f"{beta:.3f}")
col[2].metric("α", f"{alpha:.5f}")
col[3].metric("R²", f"{r2:.3f}")
st.caption(f"Cointégration p-val = {coint_p:.3f}")

df_metrics = pd.DataFrame([mA,mB], index=[ticker_a,ticker_b])
st.dataframe(df_metrics)

# ============ GRAPHS ============ #
st.subheader("Prix indexés (base 100)")
idx = prices/prices.iloc[0]*100
st.plotly_chart(px.line(idx, x=idx.index, y=idx.columns), use_container_width=True)

st.subheader("Volatilité glissante")
roll_vol = rolling_volatility(returns, win_vol)
st.plotly_chart(px.line(roll_vol, x=roll_vol.index, y=roll_vol.columns), use_container_width=True)

st.subheader("Corrélation glissante")
roll_corr = rolling_correlation(returns[ticker_a], returns[ticker_b], win_corr)
st.plotly_chart(px.line(roll_corr, x=roll_corr.index, y=roll_corr.values, labels={"y":"Corrélation"}), use_container_width=True)

st.subheader("Dispersion des rendements")
df_scatter = returns[[ticker_a, ticker_b]].dropna()
st.plotly_chart(px.scatter(df_scatter, x=ticker_a, y=ticker_b, opacity=0.5), use_container_width=True)

# ============ EXPORTS ============ #
st.subheader("Téléchargements")
st.download_button("📥 Export prix (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv")
st.download_button("📥 Export rendements (CSV)", data=returns.to_csv().encode("utf-8"), file_name="returns.csv")

st.caption("⚠️ Cet outil est pédagogique, pas un conseil en investissement.")
