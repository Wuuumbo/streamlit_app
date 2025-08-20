# app.py — Asset Risk, Volatility & Pair Analysis (2 actifs)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import norm, genpareto
from datetime import date, timedelta

# ---------- CONFIG ----------
st.set_page_config(page_title="Ultimate Asset Risk Dashboard", page_icon="📈", layout="wide")
st.title("📊 Ultimate Asset Risk, Volatility & Pair Dashboard")

FREQ_MAP = {
    "Journalier (1d)": ("1d", 252, "D"),
    "Hebdo (1wk)": ("1wk", 52, "W-FRI"),
    "Mensuel (1mo)": ("1mo", 12, "M"),
}
DEFAULT_FREQ = "Journalier (1d)"

# ---------- HELPERS ----------
def ann_factor(freq_key: str) -> int:
    return FREQ_MAP[freq_key][1]

def interval_code(freq_key: str) -> str:
    return FREQ_MAP[freq_key][0]

def range_kwargs(start_date, end_date):
    # Appliquer la même plage X à tous les graphiques temps
    return dict(range_x=[pd.Timestamp(start_date), pd.Timestamp(end_date)])

def pct(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{x*100:.{d}f}%"

def safe_cols(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c) in [a, b]]
    return df[cols].copy()

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame | None:
    df = yf.download(
        tickers=tickers, start=start, end=end,
        interval=interval, auto_adjust=True, progress=False,
        group_by="ticker", threads=True
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.loc[:, pd.IndexSlice[:, "Close"]]
            df.columns = [c[0] for c in df.columns]
        except Exception:
            if "Close" in df.columns:
                df = df["Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    return df.sort_index()

def post_filter(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()

def returns(prices: pd.DataFrame, log=False) -> pd.DataFrame:
    return (np.log(prices/prices.shift(1)) if log else prices.pct_change()).dropna()

def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices/prices.cummax() - 1.0

def max_drawdown(prices: pd.Series) -> float:
    return float(drawdown_series(prices).min())

def ann_mean(ret: pd.Series, k: int) -> float:
    return float(ret.mean()*k)

def ann_vol(ret: pd.Series, k: int) -> float:
    return float(ret.std(ddof=1)*np.sqrt(k))

def sharpe(ret: pd.Series, k: int, rf_annual: float) -> float:
    if ret.dropna().empty: return np.nan
    rf_p = (1+rf_annual)**(1/k)-1
    ex = ret.mean()-rf_p
    sig = ret.std(ddof=1)
    return float(ex/sig*np.sqrt(k)) if sig>0 else np.nan

def sortino(ret: pd.Series, k: int, rf_annual: float) -> float:
    rf_p = (1+rf_annual)**(1/k)-1
    dn = ret[ret<0]
    if dn.dropna().empty: return np.nan
    sigd = dn.std(ddof=1)
    return float((ret.mean()-rf_p)/sigd*np.sqrt(k)) if sigd>0 else np.nan

def cornish_fisher_var(ret: pd.Series, q=0.95):
    x = ret.dropna()
    if x.empty: return np.nan
    mu, sig = x.mean(), x.std(ddof=1)
    S, K = x.skew(), x.kurt()  # excess kurtosis
    z = norm.ppf(1-q)
    zcf = z + (1/6)*(z**2-1)*S + (1/24)*(z**3-3*z)*K - (1/36)*(2*z**3-5*z)*S**2
    return float(-(mu + zcf*sig))

def hist_var_es(ret: pd.Series, q=0.95):
    r = ret.dropna().values
    if r.size==0: return np.nan, np.nan
    qv = np.quantile(r, 1-q)
    var_loss = -float(qv)
    es_loss = -float(r[r<=qv].mean()) if (r<=qv).any() else np.nan
    return var_loss, es_loss

def ou_half_life(spread: pd.Series) -> float:
    x = (spread - spread.mean()).dropna()
    if len(x)<30: return np.nan
    z1 = x.shift(1).dropna()
    zt = x.loc[z1.index]
    rho, *_ = np.linalg.lstsq(z1.values.reshape(-1,1), zt.values, rcond=None)
    rho = float(rho[0])
    if rho<=0 or rho>=1: return np.nan
    return float(-np.log(2)/np.log(rho))

def beta_alpha_ols(x: pd.Series, y: pd.Series):
    df = pd.concat([x,y], axis=1).dropna()
    if df.empty: return np.nan, np.nan, np.nan
    X = sm.add_constant(df.iloc[:,0])
    m = sm.OLS(df.iloc[:,1], X).fit()
    return float(m.params[1]), float(m.params[0]), float(m.rsquared)

def rolling_beta(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    xy = pd.concat([x,y], axis=1).dropna()
    cov = xy[x.name].rolling(w).cov(xy[y.name])
    var = xy[x.name].rolling(w).var()
    return (cov/var).rename("β glissant")

def hurst_rs(series: pd.Series) -> float:
    x = series.dropna().values
    if len(x)<256: return np.nan
    # log-log R/S over scales
    scales = np.unique(np.geomspace(8, len(x)//4, num=8, dtype=int))
    rs, n = [], []
    for s in scales:
        if s<2: continue
        chunks = len(x)//s
        Y = x[:chunks*s].reshape(chunks, s)
        Z = Y - Y.mean(axis=1, keepdims=True)
        R = (np.cumsum(Z, axis=1).max(axis=1) - np.cumsum(Z, axis=1).min(axis=1))
        S = Z.std(axis=1, ddof=1)
        RS = np.where(S>0, R/S, np.nan)
        rs.append(np.nanmean(RS))
        n.append(s)
    ln_n = np.log(n); ln_rs = np.log(rs)
    H = np.polyfit(ln_n, ln_rs, 1)[0]
    return float(H)

def ewma_mu_cov(rets: pd.DataFrame, lam: float):
    r = rets.dropna()
    if r.empty: return r.mean(), r.cov()
    w = np.array([lam**k for k in range(len(r))])[::-1]; w /= w.sum()
    mu = pd.Series((r.values*w[:,None]).sum(axis=0), index=r.columns)
    S = np.zeros((r.shape[1], r.shape[1]))
    m = r - mu
    for t in range(len(m)):
        v = m.iloc[t].values.reshape(-1,1)
        S = lam*S + (1-lam)*(v@v.T)
    return mu, pd.DataFrame(S, index=r.columns, columns=r.columns)

def min_var_w(cov: pd.DataFrame) -> pd.Series:
    inv = np.linalg.pinv(cov.values)
    ones = np.ones((cov.shape[0],1))
    w = inv@ones; w = w/(ones.T@inv@ones)
    return pd.Series(w.flatten(), index=cov.columns)

def tangency_w(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.Series:
    mu_ex = mu.values - rf
    inv = np.linalg.pinv(cov.values)
    w = inv@mu_ex
    w = w/(np.ones(len(mu))@w)
    return pd.Series(w, index=mu.index)

def port_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float):
    mu_p = float(w.values@mu.values)
    sig_p = float(np.sqrt(w.values@cov.values@w.values))
    S = (mu_p-rf)/sig_p if sig_p>0 else np.nan
    return mu_p, sig_p, S

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Paramètres")
    c = st.columns(2)
    with c[0]: ticker_a = st.text_input("Ticker A", "AAPL")
    with c[1]: ticker_b = st.text_input("Ticker B", "MSFT")

    freq_key = st.selectbox("Fréquence", list(FREQ_MAP.keys()), index=list(FREQ_MAP.keys()).index(DEFAULT_FREQ))
    k = ann_factor(freq_key)
    interval = interval_code(freq_key)

    today = date.today()
    d = st.columns(2)
    with d[0]: start_date = st.date_input("Date début", value=today - timedelta(days=365*2))
    with d[1]: end_date   = st.date_input("Date fin", value=today)

    log_ret = st.checkbox("Rendements logarithmiques", False)
    rf_annual = st.number_input("Taux sans risque annualisé (%)", 0.0, step=0.25)/100.0

    roll_vol_w = st.slider(f"Fenêtre vol ({freq_key})", 10, 200, 30, step=5)
    roll_corr_w = st.slider(f"Fenêtre corr ({freq_key})", 10, 260, 60, step=5)
    roll_beta_w = st.slider(f"Fenêtre β ({freq_key})", 10, 260, 60, step=5)

    with st.expander("Paramètres avancés"):
        lam = st.slider("λ EWMA (RiskMetrics)", 0.80, 0.99, 0.94, 0.01)
        long_only = st.checkbox("Contraindre portefeuilles long-only", True)
        enable_heavy = st.checkbox("Activer modèles lourds (GARCH, régimes)", False)

    st.caption("✅ Tous les graphiques sont cadrés sur la plage de dates sélectionnée.")

# ---------- DATA ----------
raw = fetch_prices([ticker_a, ticker_b], str(start_date), str(end_date), interval)
if raw is None or raw.empty:
    st.warning("Données indisponibles. Vérifie tickers/période/fréquence.")
    st.stop()

prices = post_filter(safe_cols(raw.ffill().dropna(), ticker_a, ticker_b), start_date, end_date)
if prices.empty:
    st.warning("Aucune donnée après filtrage strict. Élargis la fenêtre.")
    st.stop()

rets = returns(prices, log_ret)
if rets.empty:
    st.warning("Rendements insuffisants après nettoyage.")
    st.stop()

# Estimation mu/cov (EWMA ou classique), annualisées
mu_est, cov_est = ewma_mu_cov(rets, lam)
mu_ann, cov_ann = mu_est*k, cov_est*k

# ---------- TABS ----------
tab_over, tab_roll, tab_port, tab_pair, tab_risk, tab_diag, tab_data = st.tabs(
    ["Overview", "Rolling", "Portfolio", "Pair Trading", "Advanced Risk", "Diagnostics", "Data"]
)

# ---------- OVERVIEW ----------
with tab_over:
    corr = rets[[ticker_a, ticker_b]].corr().iloc[0,1]
    beta, alpha, r2 = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
    try:
        coint_p = float(coint(prices[ticker_a], prices[ticker_b])[1])
    except Exception:
        coint_p = np.nan

    kpi = st.columns(5)
    kpi[0].metric("Corrélation", f"{corr:.3f}")
    kpi[1].metric("β (B~A)", f"{beta:.3f}")
    kpi[2].metric("α (par période)", f"{alpha:.5f}")
    kpi[3].metric("R² OLS", f"{r2:.3f}")
    kpi[4].metric("p-val cointégration", f"{coint_p:.3f}")

    def metrics_block(P: pd.Series, R: pd.Series):
        var95, es95 = hist_var_es(R, 0.95)
        return {
            "Dernier prix": float(P.iloc[-1]),
            "CAGR": (P.iloc[-1]/P.iloc[0])**(k/len(P)) - 1 if len(P)>=2 else np.nan,
            "Vol (ann.)": ann_vol(R, k),
            "Sharpe": sharpe(R, k, rf_annual),
            "Sortino": sortino(R, k, rf_annual),
            "Max DD": max_drawdown(P),
            "VaR95 (hist/period)": var95,
            "ES95 (hist/period)": es95,
            "VaR95 (Cornish-Fisher)": cornish_fisher_var(R),
            "Skew": float(R.skew()),
            "Kurtosis (excess)": float(R.kurt()),
        }

    mA = metrics_block(prices[ticker_a], rets[ticker_a])
    mB = metrics_block(prices[ticker_b], rets[ticker_b])
    tbl = pd.DataFrame([mA, mB], index=[ticker_a, ticker_b]).copy()
    for col in ["CAGR", "Vol (ann.)", "Max DD", "VaR95 (hist/period)", "ES95 (hist/period)", "VaR95 (Cornish-Fisher)"]:
        tbl[col] = tbl[col].apply(lambda x: pct(x))
    st.subheader("Indicateurs clés")
    st.dataframe(tbl, use_container_width=True)

    # Prix indexés
    st.subheader("Prix indexés (base=100)")
    idx = prices/prices.iloc[0]*100
    fig_idx = px.line(idx, x=idx.index, y=idx.columns, labels={"value":"Indice","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_idx, use_container_width=True, theme="streamlit")

    # Scatter rendements + droite OLS
    st.subheader("Dispersion des rendements (B vs A) + OLS")
    XY = rets[[ticker_a, ticker_b]].dropna().copy()
    X = sm.add_constant(XY[ticker_a]); mod = sm.OLS(XY[ticker_b], X).fit()
    XY["pred"] = mod.predict(X)
    fig_sc = px.scatter(XY, x=ticker_a, y=ticker_b, opacity=0.6)
    fig_sc.add_traces(px.line(XY.sort_values(ticker_a), x=ticker_a, y="pred").data)
    st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")

    # Drawdowns
    st.subheader("Drawdowns cumulés")
    dd = pd.concat([drawdown_series(prices[ticker_a]).rename(ticker_a),
                    drawdown_series(prices[ticker_b]).rename(ticker_b)], axis=1)
    fig_dd = px.line(dd, x=dd.index, y=dd.columns, labels={"value":"Drawdown","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")

# ---------- ROLLING ----------
with tab_roll:
    st.subheader("Volatilité glissante (annualisée)")
    rv = rets.rolling(roll_vol_w).std(ddof=1)*np.sqrt(k)
    fig_rv = px.line(rv, x=rv.index, y=rv.columns, labels={"value":"Vol ann.","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_rv, use_container_width=True, theme="streamlit")

    st.subheader("Corrélation glissante")
    rc = rets[ticker_a].rolling(roll_corr_w).corr(rets[ticker_b]).to_frame("Corrélation")
    fig_rc = px.line(rc, x=rc.index, y="Corrélation", **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_rc, use_container_width=True, theme="streamlit")

    st.subheader("β glissant (B sur A)")
    rb = rolling_beta(rets[ticker_a], rets[ticker_b], roll_beta_w).dropna()
    fig_rb = px.line(rb, x=rb.index, y=rb.values, labels={"y":"β glissant"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_rb, use_container_width=True, theme="streamlit")

    st.subheader("Sharpe glissant (annualisé)")
    sharpe_roll = (rets.rolling(roll_vol_w).mean() / rets.rolling(roll_vol_w).std(ddof=1))*np.sqrt(k)
    fig_sr = px.line(sharpe_roll, x=sharpe_roll.index, y=sharpe_roll.columns, labels={"value":"Sharpe","variable":"Actif"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_sr, use_container_width=True, theme="streamlit")

# ---------- PORTFOLIO ----------
with tab_port:
    st.subheader("Hypothèses annuelles (EWMA si coché)")
    c0,c1,c2 = st.columns(3)
    c0.metric(f"μ̂ {ticker_a}", pct(mu_ann[ticker_a]))
    c1.metric(f"μ̂ {ticker_b}", pct(mu_ann[ticker_b]))
    c2.metric("Corrélation", f"{rets[[ticker_a,ticker_b]].corr().iloc[0,1]:.3f}")
    st.markdown("**Covariance (annuelle)**")
    st.dataframe(cov_ann.style.format("{:.6f}"), use_container_width=True)

    rf = rf_annual
    w_min = min_var_w(cov_ann); w_tan = tangency_w(mu_ann, cov_ann, rf)
    vols = pd.Series(np.sqrt(np.diag(cov_ann.values)), index=cov_ann.columns)
    w_rp = (1/vols)/(1/vols).sum()
    b_ba, _, _ = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
    w_beta = pd.Series({ticker_a:1.0, ticker_b:-b_ba}); w_beta = w_beta/w_beta.abs().sum()
    if long_only:
        w_min = w_min.clip(lower=0); w_min = w_min/w_min.sum()
        w_tan = w_tan.clip(lower=0); w_tan = w_tan/w_tan.sum()
        w_rp  = w_rp.clip(lower=0);  w_rp  = w_rp / w_rp.sum()
    rows=[]
    for name,w in [("Equal-Weight", pd.Series([0.5,0.5], index=[ticker_a,ticker_b])),
                   ("Min-Variance", w_min),
                   ("Tangency", w_tan),
                   ("Risk-Parity", w_rp),
                   ("β-neutral", w_beta)]:
        mu_p, sig_p, S = port_stats(w, mu_ann, cov_ann, rf)
        rows.append({"Portfolio":name, f"w_{ticker_a}":w[ticker_a], f"w_{ticker_b}":w[ticker_b], "μ (ann.)":mu_p, "σ (ann.)":sig_p, "Sharpe":S})
    port_tbl = pd.DataFrame(rows)
    st.subheader("Poids & performances (annuelles)")
    st.dataframe(port_tbl.style.format({f"w_{ticker_a}":"{:.2%}", f"w_{ticker_b}":"{:.2%}", "μ (ann.)":"{:.2%}", "σ (ann.)":"{:.2%}", "Sharpe":"{:.2f}"}), use_container_width=True)

    # Frontière efficiente 2 actifs
    w = np.linspace(0,1,201)
    mu_a, mu_b = mu_ann[ticker_a], mu_ann[ticker_b]
    var_a, var_b = cov_ann.loc[ticker_a,ticker_a], cov_ann.loc[ticker_b,ticker_b]
    cov_ab = cov_ann.loc[ticker_a,ticker_b]
    mu_pf = w*mu_a+(1-w)*mu_b
    sig_pf = np.sqrt((w**2)*var_a+((1-w)**2)*var_b+2*w*(1-w)*cov_ab)
    fig_front = px.line(x=sig_pf, y=mu_pf, labels={"x":"σ (ann.)","y":"μ (ann.)"}, title="Efficient Frontier (2 actifs)")
    mu_min, sig_min, _ = port_stats(w_min, mu_ann, cov_ann, rf)
    mu_tan, sig_tan, _ = port_stats(w_tan, mu_ann, cov_ann, rf)
    fig_front.add_scatter(x=[sig_min], y=[mu_min], mode="markers", name="Min-Var")
    fig_front.add_scatter(x=[sig_tan], y=[mu_tan], mode="markers", name="Tangency")
    st.plotly_chart(fig_front, use_container_width=True, theme="streamlit")

    # Backtest cumulé des portefeuilles (périodique)
    st.subheader("Backtest cumulé (base=1)")
    weights = {"Equal":pd.Series([0.5,0.5], index=[ticker_a,ticker_b]),
               "MinVar":w_min, "Tangency":w_tan, "RiskParity":w_rp, "BetaNeutral":w_beta}
    cum = pd.DataFrame(index=rets.index)
    for name,w in weights.items():
        r = (rets*w).sum(axis=1)
        cum[name] = (1+r).cumprod()
    fig_cum = px.line(cum, x=cum.index, y=cum.columns, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_cum, use_container_width=True, theme="streamlit")

# ---------- PAIR TRADING ----------
with tab_pair:
    st.subheader("Cointegration & Spread (OLS sur prix)")
    X = sm.add_constant(prices[ticker_a]); model = sm.OLS(prices[ticker_b], X).fit()
    hedge = float(model.params[1])
    spread = prices[ticker_b] - hedge*prices[ticker_a]
    z = (spread - spread.rolling(60).mean())/spread.rolling(60).std(ddof=1)
    hl = ou_half_life(spread)

    kpi = st.columns(4)
    kpi[0].metric("Hedge ratio b", f"{hedge:.3f}")
    kpi[1].metric("p-val cointégration", f"{coint(prices[ticker_a], prices[ticker_b])[1]:.3f}")
    kpi[2].metric("Half-life OU", f"{hl:.1f}" if not pd.isna(hl) else "—")
    kpi[3].metric("Z-score actuel", f"{z.iloc[-1]:.2f}")

    fig_sp = px.line(spread, labels={"value":"Spread","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_sp, use_container_width=True, theme="streamlit")

    st.subheader("Z-score du spread (fenêtre 60)")
    Z = z.to_frame("Z").dropna()
    fig_z = px.line(Z, x=Z.index, y="Z", **range_kwargs(start_date,end_date))
    fig_z.add_hline(y=0, line_dash="dot"); fig_z.add_hline(y=2, line_dash="dash", annotation_text="+2")
    fig_z.add_hline(y=-2, line_dash="dash", annotation_text="-2")
    st.plotly_chart(fig_z, use_container_width=True, theme="streamlit")
    st.caption("⚠️ Signal pédagogique : |Z|>2 ⇒ écart extrême; reversion attendue (non-conseil).")

# ---------- ADVANCED RISK ----------
with tab_risk:
    st.subheader("EVT (POT-GPD) sur pertes (VaR/ES extrêmes)")
    cols = st.columns(2)
    for i,(label, r) in enumerate([(ticker_a, -rets[ticker_a]), (ticker_b, -rets[ticker_b])]):
        r = r.dropna()
        if len(r)<200:
            cols[i].warning(f"{label}: série trop courte pour EVT.")
            continue
        u = np.quantile(r, 0.95)  # seuil 95% des pertes
        excess = r[r>u] - u
        try:
            # Fit GPD aux excès
            c, loc, scale = genpareto.fit(excess.values, floc=0)
            p_tail = (excess.size)/len(r)  # prob. d'excès
            # Cible p=99% global
            p = 0.99
            # VaR_p(u) = u + σ/ξ * [ ( ( (1-p) / (1-u_prob) )^(-ξ) - 1 ) ]  (pour pertes)
            # Ici 1-u_prob ≈ p_tail
            if c != 0:
                var_p = u + scale/c * (( ( (1-p)/p_tail )**(-c) - 1 ))
            else:
                var_p = u + scale*np.log(p_tail/(1-p))
            # ES approx GPD
            es_p = (var_p + (scale - c*u)/(1-c)) if c<1 else np.nan
            cols[i].metric(f"EVT VaR99 {label}", pct(var_p))
            cols[i].metric(f"EVT ES99 {label}", pct(es_p))
        except Exception:
            cols[i].warning(f"{label}: échec ajustement GPD.")

    st.subheader("CoVaR (Quantile Regression, q=5%)")
    try:
        # B | A en détresse (A à son quantile 5%)
        qa = rets[ticker_a].quantile(0.05)
        qb = rets[ticker_b].quantile(0.05)
        # CoVaR_{B|A}: quantile 5% de B conditionnel à A
        qmod = QuantReg(rets[ticker_b], sm.add_constant(rets[ticker_a])).fit(q=0.05)
        covar_b_a = float(qmod.predict([1, qa])[0])
        # Delta CoVaR: différence vs condition médiane d'A
        qmod_med = QuantReg(rets[ticker_b], sm.add_constant(rets[ticker_a])).fit(q=0.5)
        covar_med = float(qmod_med.predict([1, rets[ticker_a].median()])[0])
        d_covar_b_a = covar_b_a - covar_med

        # CoVaR_{A|B}
        qmod2 = QuantReg(rets[ticker_a], sm.add_constant(rets[ticker_b])).fit(q=0.05)
        covar_a_b = float(qmod2.predict([1, qb])[0])
        qmod2_med = QuantReg(rets[ticker_a], sm.add_constant(rets[ticker_b])).fit(q=0.5)
        d_covar_a_b = covar_a_b - float(qmod2_med.predict([1, rets[ticker_b].median()])[0])

        c = st.columns(4)
        c[0].metric(f"CoVaR5% {ticker_b}|{ticker_a}", pct(-covar_b_a))
        c[1].metric("ΔCoVaR B|A", pct(-d_covar_b_a))
        c[2].metric(f"CoVaR5% {ticker_a}|{ticker_b}", pct(-covar_a_b))
        c[3].metric("ΔCoVaR A|B", pct(-d_covar_a_b))
    except Exception:
        st.warning("Quantile Regression échouée (échantillon trop court ou colinéarité).")

    st.subheader("GARCH(1,1) cond. vol (optionnel)")
    if enable_heavy:
        try:
            from arch import arch_model  # optionnel
            cols = st.columns(2)
            for i,(label, r) in enumerate([(ticker_a, rets[ticker_a]), (ticker_b, rets[ticker_b])]):
                r = r.dropna()
                if len(r)<250:
                    cols[i].warning(f"{label}: série trop courte pour GARCH.")
                    continue
                am = arch_model(r*100, mean="zero", vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off")
                cond_vol = res.conditional_volatility/100*np.sqrt(ann_factor(freq_key))
                fig_g = px.line(cond_vol, labels={"value":"Vol cond. (ann.)","index":"Date"}, **range_kwargs(start_date,end_date))
                cols[i].plotly_chart(fig_g, use_container_width=True, theme="streamlit")
        except Exception:
            st.info("Librairie 'arch' non disponible — ajoute-la à requirements pour activer GARCH.")

    st.subheader("Régimes markoviens (variance switching) — 2 régimes")
    if enable_heavy:
        try:
            # Sur |ret| pour capturer la variance (régimes calme/turbu)
            y = (rets.mean(axis=1)).dropna()
            mod = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
            res = mod.fit(disp=False)
            pr = res.smoothed_marginal_probabilities[1]  # prob régime 1 (haute variance)
            fig_m = px.line(pr, labels={"value":"Prob. régime haute variance","index":"Date"}, **range_kwargs(start_date,end_date))
            st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")
        except Exception:
            st.info("Régimes markoviens indisponibles (échec optimisation). Essaie une période plus longue.")

    st.subheader("Hurst (R/S) — persistance vs mean-reversion")
    H_a = hurst_rs(prices[ticker_a].pct_change().dropna())
    H_b = hurst_rs(prices[ticker_b].pct_change().dropna())
    c = st.columns(2)
    c[0].metric(f"Hurst {ticker_a}", f"{H_a:.2f}" if not pd.isna(H_a) else "—",
                help=">0.5: persistant; <0.5: anti-persistant (mean-reverting)")
    c[1].metric(f"Hurst {ticker_b}", f"{H_b:.2f}" if not pd.isna(H_b) else "—")

# ---------- DIAGNOSTICS ----------
with tab_diag:
    ra, rb = rets[ticker_a].dropna(), rets[ticker_b].dropna()

    def adf_p(x): 
        try: return float(adfuller(x)[1])
        except: return np.nan
    def ljung_p(x, l=10):
        try: return float(acorr_ljungbox(x, lags=[l], return_df=True)["lb_pvalue"].iloc[0])
        except: return np.nan
    def arch_p(x, l=10):
        try: return float(het_arch(x, nlags=l)[1])
        except: return np.nan
    def jb_p(x):
        try: return float(jarque_bera(x)[1])
        except: return np.nan
    def granger_min(x_from, y_to, L=5):
        try:
            df = pd.concat([y_to, x_from], axis=1).dropna()
            if len(df)<50: return np.nan
            res = grangercausalitytests(df, maxlag=L, verbose=False)
            p = [res[i][0]["ssr_ftest"][1] for i in range(1, L+1)]
            return float(np.nanmin(p))
        except: return np.nan

    grid = pd.DataFrame({
        "Test": ["ADF", "Ljung-Box(10)", "ARCH(10)", "Jarque-Bera", "Granger A→B", "Granger B→A"],
        ticker_a: [adf_p(ra), ljung_p(ra), arch_p(ra), jb_p(ra), granger_min(ra, rb), np.nan],
        ticker_b: [adf_p(rb), ljung_p(rb), arch_p(rb), jb_p(rb), np.nan, granger_min(rb, ra)],
    })
    st.dataframe(grid, use_container_width=True)

    st.subheader("Histogrammes des rendements")
    hc = st.columns(2)
    hc[0].plotly_chart(px.histogram(ra, nbins=60, labels={"value":f"Rendements {ticker_a}"}), use_container_width=True, theme="streamlit")
    hc[1].plotly_chart(px.histogram(rb, nbins=60, labels={"value":f"Rendements {ticker_b}"}), use_container_width=True, theme="streamlit")

# ---------- DATA ----------
with tab_data:
    st.subheader("Paramètres")
    st.write({
        "Tickers": [ticker_a, ticker_b],
        "Fréquence": freq_key,
        "Plage": {"start": str(start_date), "end": str(end_date)},
        "Log returns": log_ret, "RF (ann.)": rf_annual,
        "Fenêtres": {"vol": roll_vol_w, "corr": roll_corr_w, "beta": roll_beta_w},
        "EWMA λ": lam, "Long-only": long_only, "Modèles lourds": enable_heavy
    })

    st.subheader("Prix (filtrés)")
    st.dataframe(prices, use_container_width=True)
    st.download_button("📥 PRIX (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv")

    st.subheader("Rendements")
    st.dataframe(rets, use_container_width=True)
    st.download_button("📥 RENDEMENTS (CSV)", data=rets.to_csv().encode("utf-8"), file_name="returns.csv")



