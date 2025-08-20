# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.express as px
import statsmodels.api as sm

from utils import (
    download_prices,
    compute_returns,
    summary_metrics,
    rolling_volatility,
    rolling_correlation,
    beta_alpha_ols,
    cointegration_test,
    drawdown_series,
    format_pct,
)

st.set_page_config(
    page_title="Asset Risk & Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

# ======= SIDEBAR =======
st.sidebar.title("⚙️ Paramètres")

default_a = "AAPL"
default_b = "MSFT"

ticker_a = st.sidebar.text_input("Ticker A", value=default_a, help="Ex: AAPL, BTC-USD, ^GSPC, EURUSD=X")
ticker_b = st.sidebar.text_input("Ticker B", value=default_b)

today = date.today()
start_default = today - timedelta(days=365*2)

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Date début", value=start_default)
with col_d2:
    end_date = st.date_input("Date fin", value=today)

log_returns = st.sidebar.checkbox("Utiliser des rendements logarithmiques", value=False)
risk_free = st.sidebar.number_input("Taux sans risque annualisé (%)", value=0.0, step=0.25)
roll_vol_win = st.sidebar.slider("Fenêtre volatilité (jours)", 10, 120, 30, step=5)
roll_corr_win = st.sidebar.slider("Fenêtre corrélation (jours)", 20, 180, 60, step=5)
fill_missing = st.sidebar.selectbox("Données manquantes", ["ffill + dropna", "dropna"])
currency_note = st.sidebar.caption("💡 Source: Yahoo! Finance via yfinance (auto-adjust).")

st.title("📊 Asset Risk & Volatility Dashboard (2 actifs)")

# ======= DOWNLOAD PRICES =======
@st.cache_data(show_spinner=True)
def _load_data(a, b, start, end, fill):
    prices = download_prices([a, b], str(start), str(end))
    if prices is None or prices.empty:
        return None
    if fill == "ffill + dropna":
        prices = prices.ffill().dropna()
    else:
        prices = prices.dropna()
    return prices

prices = _load_data(ticker_a.strip(), ticker_b.strip(), start_date, end_date, fill_missing)

if not prices or prices is None or prices.empty:
    st.warning("Impossible de récupérer les prix. Vérifie les tickers et la période.")
    st.stop()

# ======= RETURNS =======
rets = compute_returns(prices, log=log_returns).dropna()
if rets.empty:
    st.warning("Pas assez de données de rendements après nettoyage.")
    st.stop()

# ======= TOP KPIs =======
st.subheader("Aperçu & Indicateurs clés")

rf_decimal = risk_free / 100.0
metrics_a = summary_metrics(prices[ticker_a], rets[ticker_a], rf_decimal)
metrics_b = summary_metrics(prices[ticker_b], rets[ticker_b], rf_decimal)

# Pair metrics
beta, alpha, r2 = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
corr = rets[[ticker_a, ticker_b]].corr().iloc[0,1]
coint_p = cointegration_test(prices[ticker_a], prices[ticker_b])

kpi_cols = st.columns(4)
kpi_cols[0].metric("Corrélation", f"{corr:.3f}")
kpi_cols[1].metric("β (B sur A)", f"{beta:.3f}", help=f"Régression: {ticker_b} ~ {ticker_a}")
kpi_cols[2].metric("α (journalier)", format_pct(alpha), help="Alpha de la régression journalière")
kpi_cols[3].metric("R² régression", f"{r2:.3f}")

st.caption(f"Test de cointégration (Engle-Granger) p-value = **{coint_p:.3f}** (p < 0.05 ⇒ cointégration possible)")

# ======= METRICS TABLE =======
table = pd.DataFrame({
    "Actif": [ticker_a, ticker_b],
    "Dernier prix": [metrics_a['last_price'], metrics_b['last_price']],
    "CAGR (annualisé)": [format_pct(metrics_a['cagr']), format_pct(metrics_b['cagr'])],
    "Vol (annualisée)": [format_pct(metrics_a['vol_annual']), format_pct(metrics_b['vol_annual'])],
    "Sharpe": [f"{metrics_a['sharpe']:.2f}", f"{metrics_b['sharpe']:.2f}"],
    "Sortino": [f"{metrics_a['sortino']:.2f}", f"{metrics_b['sortino']:.2f}"],
    "Max Drawdown": [format_pct(metrics_a['max_dd']), format_pct(metrics_b['max_dd'])],
    "VaR 95% (J)": [format_pct(metrics_a['var_hist_95']), format_pct(metrics_b['var_hist_95'])],
    "ES 95% (J)": [format_pct(metrics_a['es_hist_95']), format_pct(metrics_b['es_hist_95'])],
    "VaR Gauss 95% (J)": [format_pct(metrics_a['var_norm_95']), format_pct(metrics_b['var_norm_95'])],
})
st.dataframe(table, use_container_width=True)

# ======= CHARTS =======
st.subheader("Prix indexés (=100 au départ)")
indexed = prices / prices.iloc[0] * 100.0
fig_idx = px.line(indexed, x=indexed.index, y=indexed.columns, labels={"value":"Indice (base=100)", "variable":"Actif", "index":"Date"})
st.plotly_chart(fig_idx, use_container_width=True, theme="streamlit")

st.subheader("Volatilité glissante (annualisée)")
roll_vol = rolling_volatility(rets, window=roll_vol_win)
fig_rv = px.line(roll_vol, x=roll_vol.index, y=roll_vol.columns, labels={"value":"Vol annualisée", "variable":"Actif", "index":"Date"})
st.plotly_chart(fig_rv, use_container_width=True, theme="streamlit")

st.subheader("Corrélation glissante")
roll_corr = rolling_correlation(rets[ticker_a], rets[ticker_b], window=roll_corr_win).to_frame("Corrélation")
fig_rc = px.line(roll_corr, x=roll_corr.index, y="Corrélation", labels={"value":"Corrélation", "index":"Date"})
st.plotly_chart(fig_rc, use_container_width=True, theme="streamlit")

st.subheader("Dispersion des rendements (B vs A) & droite de régression")
xy = pd.DataFrame({
    ticker_a: rets[ticker_a],
    ticker_b: rets[ticker_b]
}).dropna()

# Regression line
X = sm.add_constant(xy[ticker_a])
model = sm.OLS(xy[ticker_b], X).fit()
xy["pred"] = model.predict(X)

fig_sc = px.scatter(xy, x=ticker_a, y=ticker_b, trendline=None, opacity=0.6, labels={"x":ticker_a, "y":ticker_b})
fig_sc.add_traces(px.line(xy.sort_values(ticker_a), x=ticker_a, y="pred").data)
st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")

# ======= DRAWDOWN CHART =======
st.subheader("Drawdowns cumulés")
dd_a = drawdown_series(prices[ticker_a])
dd_b = drawdown_series(prices[ticker_b])
ddf = pd.concat([dd_a.rename(ticker_a), dd_b.rename(ticker_b)], axis=1)
fig_dd = px.line(ddf, x=ddf.index, y=ddf.columns, labels={"value":"Drawdown", "variable":"Actif", "index":"Date"})
st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")

# ======= DOWNLOADS =======
st.subheader("Téléchargements")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Télécharger Prix (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv", mime="text/csv")
with col2:
    st.download_button("Télécharger Rendements (CSV)", data=rets.to_csv().encode("utf-8"), file_name="returns.csv", mime="text/csv")

st.caption("⚠️ Avertissement : Cet outil est éducatif. Pas un conseil en investissement.")
