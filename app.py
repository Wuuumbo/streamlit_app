# app.py — Ultimate Asset Risk, Volatility & Pair Dashboard (2 assets)
# Un seul fichier — prêt pour Streamlit Cloud/Replit.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import norm, genpareto
from datetime import date, timedelta

# ------------- CONFIG -------------
st.set_page_config(page_title="Ultimate Asset Risk Dashboard", page_icon="📈", layout="wide")
st.title("📊 Ultimate Asset Risk, Volatility & Pair Dashboard — 2 actifs")

FREQ_MAP = {
    "Journalier (1d)": ("1d", 252, "D"),      # interval_yf, annualization, pandas rule (after resample)
    "Hebdo (1wk)":     ("1wk", 52,  "W-FRI"),
    "Mensuel (1mo)":   ("1mo", 12,  "M"),
}
DEFAULT_FREQ = "Journalier (1d)"

# ------------- HELPERS -------------
def ann_factor(freq_key: str) -> int:
    return FREQ_MAP[freq_key][1]

def resample_rule(freq_key: str) -> str:
    return FREQ_MAP[freq_key][2]

def range_kwargs(start_date, end_date):
    # uniformiser l’abscisse sur toute l’app
    return dict(range_x=[pd.Timestamp(start_date), pd.Timestamp(end_date)])

def pct(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{x*100:.{d}f}%"

def safe_two_cols(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c) in [a, b]]
    return df[cols].copy()

def strict_resample(prices_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample en fin de période (last) pour stabiliser la lecture finance."""
    if rule == "D":  # déjà journalier
        return prices_daily.copy()
    return prices_daily.resample(rule).last()

def returns(prices: pd.DataFrame, log=False) -> pd.DataFrame:
    return (np.log(prices/prices.shift(1)) if log else prices.pct_change()).dropna()

def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices/prices.cummax() - 1.0

def max_drawdown(prices: pd.Series) -> float:
    return float(drawdown_series(prices).min())

def ann_vol(ret: pd.Series, k: int) -> float:
    return float(ret.std(ddof=1)*np.sqrt(k))

def sharpe(ret: pd.Series, k: int, rf_annual: float) -> float:
    if ret.dropna().empty: return np.nan
    rf_p = (1+rf_annual)**(1/k) - 1
    sig = ret.std(ddof=1)
    return float(((ret.mean()-rf_p)/sig)*np.sqrt(k)) if sig>0 else np.nan

def sortino(ret: pd.Series, k: int, rf_annual: float) -> float:
    rf_p = (1+rf_annual)**(1/k) - 1
    dn = ret[ret<0]
    if dn.dropna().empty: return np.nan
    sigd = dn.std(ddof=1)
    return float(((ret.mean()-rf_p)/sigd)*np.sqrt(k)) if sigd>0 else np.nan

def cornish_fisher_var(ret: pd.Series, q=0.95):
    x = ret.dropna()
    if x.empty: return np.nan
    mu, sig = x.mean(), x.std(ddof=1)
    S, K = x.skew(), x.kurt()  # excess kurt
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

def ou_half_life(spread: pd.Series) -> float:
    x = (spread - spread.mean()).dropna()
    if len(x)<30: return np.nan
    x1 = x.shift(1).dropna()
    xt = x.loc[x1.index]
    rho, *_ = np.linalg.lstsq(x1.values.reshape(-1,1), xt.values, rcond=None)
    rho = float(rho[0])
    if rho<=0 or rho>=1: return np.nan
    return float(-np.log(2)/np.log(rho))

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

def tail_corr(a: pd.Series, b: pd.Series, q=0.05, upper=False):
    """Corrélation conditionnelle dans la queue : les deux < q (ou > 1-q)."""
    if upper:
        mask = (a >= a.quantile(1-q)) & (b >= b.quantile(1-q))
    else:
        mask = (a <= a.quantile(q)) & (b <= b.quantile(q))
    sub = pd.concat([a[mask], b[mask]], axis=1).dropna()
    return float(sub.corr().iloc[0,1]) if len(sub)>5 else np.nan

def xcorr_at_lags(a: pd.Series, b: pd.Series, lags=5):
    """Cross-corr lead/lag : corr(b_t, a_{t-k}) et corr(a_t, b_{t-k}) pour k=1..lags."""
    res = []
    for k in range(1, lags+1):
        res.append({
            "lag": k,
            "corr A→B": pd.concat([b, a.shift(k)], axis=1).dropna().corr().iloc[0,1],
            "corr B→A": pd.concat([a, b.shift(k)], axis=1).dropna().corr().iloc[0,1],
        })
    return pd.DataFrame(res)

# ------------- SIDEBAR -------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    c = st.columns(2)
    with c[0]: ticker_a = st.text_input("Ticker A", "AAPL")
    with c[1]: ticker_b = st.text_input("Ticker B", "MSFT")

    freq_key = st.selectbox("Fréquence", list(FREQ_MAP.keys()), index=list(FREQ_MAP.keys()).index(DEFAULT_FREQ))
    k = ann_factor(freq_key)
    rule = resample_rule(freq_key)

    today = date.today()
    d = st.columns(2)
    with d[0]: start_date = st.date_input("Date début", value=today - timedelta(days=365*3))
    with d[1]: end_date   = st.date_input("Date fin", value=today)

    log_ret = st.checkbox("Rendements logarithmiques", False)
    rf_annual = st.number_input("Taux sans risque annualisé (%)", 0.0, step=0.25)/100.0

    roll_vol_w  = st.slider(f"Fenêtre vol ({freq_key})", 10, 200, 30, step=5)
    roll_corr_w = st.slider(f"Fenêtre corr ({freq_key})", 10, 260, 60, step=5)
    roll_beta_w = st.slider(f"Fenêtre β ({freq_key})",   10, 260, 60, step=5)

    with st.expander("Paramètres avancés"):
        lam = st.slider("λ EWMA (RiskMetrics)", 0.80, 0.99, 0.94, 0.01)
        long_only = st.checkbox("Contraindre portefeuilles long-only", True)
        enable_heavy = st.checkbox("Activer modèles lourds (GARCH, régimes)", False)
        lags_xcorr = st.slider("Lags cross-corr (1..20)", 3, 20, 5)

    st.caption("✅ Tous les graphiques sont cadrés sur la plage de dates sélectionnée. Les données sont **récupérées en journalier** sur la plage (avec buffer technique), puis **resamplées** selon la fréquence choisie.")

# ------------- DATA FETCH (STRICT DATE HANDLING) -------------
@st.cache_data(show_spinner=True)
def fetch_daily_close(tickers, start, end):
    """Télécharge en journalier pour couvrir toute la plage. Retour: Close columns."""
    df = yf.download(tickers=tickers, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, group_by="ticker", threads=True)
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

# buffer pour fiabiliser les fenêtres glissantes et resampling
max_win = max(roll_vol_w, roll_corr_w, roll_beta_w)
pre_buffer_days = int(max_win*3)  # buffer généreux
fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=pre_buffer_days)
fetch_end   = pd.Timestamp(end_date)   + pd.Timedelta(days=2)  # +2 pour inclure la dernière clôture

raw_daily = fetch_daily_close([ticker_a, ticker_b], str(fetch_start.date()), str(fetch_end.date()))
if raw_daily is None or raw_daily.empty:
    st.warning("Données indisponibles (Yahoo Finance). Vérifie tickers/période.")
    st.stop()

# Vérifier que les 2 colonnes existent
missing_cols = [t for t in [ticker_a, ticker_b] if t not in raw_daily.columns]
if missing_cols:
    st.error(f"Les tickers suivants n'ont pas été trouvés dans les données: {missing_cols}. "
             f"Colonnes reçues: {list(raw_daily.columns)}")
    st.stop()

# Alignement, ffill et strict filter
raw_daily = safe_two_cols(raw_daily.ffill(), ticker_a, ticker_b).dropna(how="all")
prices_resampled = strict_resample(raw_daily, rule).ffill()
prices = prices_resampled.loc[(prices_resampled.index>=pd.Timestamp(start_date)) &
                              (prices_resampled.index<=pd.Timestamp(end_date))].copy()

if prices.empty:
    st.warning("Aucune donnée après resampling/filtre strict. Élargis la période ou change la fréquence.")
    st.stop()

rets = returns(prices, log_ret)
if rets.empty:
    st.warning("Rendements insuffisants après nettoyage.")
    st.stop()

# Audit de complétude : couverture attendue vs observée (utilise jours ouvrés pour le journalier)
def expected_points(start, end, rule):
    freq = "B" if rule == "D" else rule  # "B" = business days
    rng = pd.date_range(start=start, end=end, freq=freq)
    return len(rng)

expected = expected_points(start_date, end_date, rule)
coverage = len(prices)/expected if expected>0 else np.nan
st.sidebar.markdown(f"**Audit data** : points observés = {len(prices)} / attendus ≈ {expected} → couverture ≈ **{coverage:.0%}**")
if coverage < 0.7:
    st.sidebar.warning("Couverture < 70% : possible manque de data (jours fériés/actif récent). Interprétez avec prudence.")

# ------------- TABS -------------
tab_over, tab_roll, tab_port, tab_pair, tab_risk, tab_diag, tab_dep, tab_data = st.tabs(
    ["Overview", "Rolling", "Portfolio", "Pair Trading", "Advanced Risk", "Diagnostics", "Dépendances de queue & Lags", "Data"]
)

# ========== OVERVIEW ==========
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

    st.caption("**Interprétation** — Corrélation: comouvement global; β: sensibilité de B aux variations de A; α: performance idiosyncratique; R²: qualité d’ajustement; cointégration: relation d’équilibre de long terme (p<0,05).")

    # Table indicateurs
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
    for col in ["CAGR","Vol (ann.)","Max DD","VaR95 (hist/period)","ES95 (hist/period)","VaR95 (Cornish-Fisher)"]:
        tbl[col] = tbl[col].apply(pct)
    st.subheader("Indicateurs clés (annualisés)")
    st.dataframe(tbl, use_container_width=True)
    st.caption("**Mode d’emploi** — Utilise CAGR/Sharpe/Sortino pour classer le couple rendement/risque; Max DD pour le risque extrême; VaR/ES pour pertes attendues en queue; Skew/Kurtosis pour la forme de distribution (asymétrie, queues épaisses).")

    # Prix indexés
    st.subheader("Prix indexés (base=100)")
    idx = prices/prices.iloc[0]*100
    fig_idx = px.line(idx, x=idx.index, y=idx.columns, labels={"value":"Indice","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_idx, use_container_width=True, theme="streamlit")
    st.caption("**Pourquoi** — Met les actifs sur une échelle comparable; **Comment** — Regarde qui surperforme (s’éloigne >100) sur la fenêtre choisie.")

    # Scatter + droite OLS (✅ correctifs)
    st.subheader("Dispersion des rendements (B vs A) + OLS")
    XY = rets[[ticker_a, ticker_b]].dropna().copy()
    X = sm.add_constant(XY[ticker_a])
    y = XY[ticker_b]
    mod = sm.OLS(y, X).fit()
    XY["pred"] = mod.predict(X)
    fig_sc = px.scatter(XY, x=ticker_a, y=ticker_b, opacity=0.6)
    fig_sc.add_traces(px.line(XY.sort_values(ticker_a), x=ticker_a, y="pred").data)
    st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")
    st.caption("**Lecture** — La pente correspond à β; les points loin de la droite sont des anomalies/idiosyncratiques; utile pour **hedging** et **pairs**.")

    # Drawdowns
    st.subheader("Drawdowns cumulés")
    dd = pd.concat([drawdown_series(prices[ticker_a]).rename(ticker_a),
                    drawdown_series(prices[ticker_b]).rename(ticker_b)], axis=1)
    fig_dd = px.line(dd, x=dd.index, y=dd.columns, labels={"value":"Drawdown","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")
    st.caption("**À quoi ça sert** — Visualise les pertes par rapport au pic; utile pour jauger l’**expérience investisseur** en période difficile.")

# ========== ROLLING ==========
with tab_roll:
    st.subheader("Volatilité glissante (annualisée)")
    rv = rets.rolling(roll_vol_w).std(ddof=1)*np.sqrt(k)
    fig_rv = px.line(rv, x=rv.index, y=rv.columns, labels={"value":"Vol ann.","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_rv, use_container_width=True, theme="streamlit")
    st.caption("**Pourquoi** — Suivi du **risque dynamique**; observe les régimes calmes vs turbulents; compare A vs B à date donnée.")

    st.subheader("Corrélation glissante")
    rc = rets[ticker_a].rolling(roll_corr_w).corr(rets[ticker_b]).to_frame("Corrélation")
    fig_rc = px.line(rc, x=rc.index, y="Corrélation", **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_rc, use_container_width=True, theme="streamlit")
    st.caption("**Usage** — Si la corrélation plonge durablement, diversification accrue; si elle monte vers 1, bénéfices de diversification s’érodent.")

    st.subheader("β glissant (B sur A)")
    rb = rolling_beta(rets[ticker_a], rets[ticker_b], roll_beta_w).dropna()
    # ✅ Série → DataFrame (évite ValueError Plotly)
    rb_df = rb.reset_index()
    rb_df.columns = ["Date", "β glissant"]
    fig_rb = px.line(rb_df, x="Date", y="β glissant", **range_kwargs(start_date, end_date))
    st.plotly_chart(fig_rb, use_container_width=True, theme="streamlit")
    st.caption("**Interprétation** — β dynamique utile pour **hedging tactique** : ajuste les tailles long/short en fonction du β courant.")

    st.subheader("Sharpe glissant (annualisé)")
    sharpe_roll = (rets.rolling(roll_vol_w).mean() / rets.rolling(roll_vol_w).std(ddof=1))*np.sqrt(k)
    fig_sr = px.line(sharpe_roll, x=sharpe_roll.index, y=sharpe_roll.columns, labels={"value":"Sharpe","variable":"Actif","index":"Date"}, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_sr, use_container_width=True, theme="streamlit")
    st.caption("**Comment l’utiliser** — Suivre la **qualité** du rendement ajusté du risque dans le temps; compare A/B pour choisir l’actif dominant.")

# ========== PORTFOLIO ==========
with tab_port:
    st.subheader("Hypothèses (annualisées) — μ̂ & Σ̂ (EWMA si coché)")
    mu_est, cov_est = ewma_mu_cov(rets, lam)
    mu_ann, cov_ann = mu_est*k, cov_est*k
    c0,c1,c2 = st.columns(3)
    c0.metric(f"μ̂ {ticker_a}", pct(mu_ann[ticker_a]))
    c1.metric(f"μ̂ {ticker_b}", pct(mu_ann[ticker_b]))
    c2.metric("Corrélation", f"{rets[[ticker_a,ticker_b]].corr().iloc[0,1]:.3f}")
    st.dataframe(cov_ann.style.format("{:.6f}"), use_container_width=True)
    st.caption("**Note** — μ̂, Σ̂ estimés sur la fenêtre sélectionnée ; EWMA (λ) met plus de poids au récent (RiskMetrics).")

    rf = rf_annual
    w_min = min_var_w(cov_ann)
    w_tan = tangency_w(mu_ann, cov_ann, rf)
    vols = pd.Series(np.sqrt(np.diag(cov_ann.values)), index=cov_ann.columns)
    w_rp = (1/vols)/(1/vols).sum()
    b_ba, _, _ = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
    w_beta = pd.Series({ticker_a:1.0, ticker_b:-b_ba}); w_beta = w_beta/w_beta.abs().sum()
    if long_only:
        for W in [w_min, w_tan, w_rp]:
            W[W<0]=0; s=W.sum(); 
            if s>0: W/=s

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
    st.caption("**Lecture** — Compare les architectures de portefeuille. **Tangency** maximise Sharpe; **Min-Var** minimise le risque; **Risk-Parity** égalise les contributions de risque; **β-neutral** vise neutralité directionnelle entre A & B.")

    # Frontière efficiente (2 actifs)
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
    st.caption("**Utilisation** — Choisis le point sur la frontière selon ton budget de risque (σ) et ton objectif de rendement (μ).")

    # Backtest cumulé périodique
    st.subheader("Backtest cumulé (base=1)")
    weights = {"Equal":pd.Series([0.5,0.5], index=[ticker_a,ticker_b]),
               "MinVar":w_min, "Tangency":w_tan, "RiskParity":w_rp, "BetaNeutral":w_beta}
    cum = pd.DataFrame(index=rets.index)
    for name,w in weights.items():
        r = (rets*w).sum(axis=1)
        cum[name] = (1+r).cumprod()
    fig_cum = px.line(cum, x=cum.index, y=cum.columns, **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_cum, use_container_width=True, theme="streamlit")
    st.caption("**Attention** — Backtest simple, rééquilibrage implicite à chaque période; pas de coûts de transaction, pas de frictions.")

# ========== PAIR TRADING ==========
with tab_pair:
    st.subheader("Cointegration & Spread (OLS sur prix)")
    X = sm.add_constant(prices[ticker_a]); model = sm.OLS(prices[ticker_b], X).fit()
    hedge = float(model.params[1])
    spread = prices[ticker_b] - hedge*prices[ticker_a]
    z = (spread - spread.rolling(60).mean())/spread.rolling(60).std(ddof=1)
    hl = ou_half_life(spread)

    kpi = st.columns(4)
    kpi[0].metric("Hedge ratio b", f"{hedge:.3f}")
    try:
        kpi[1].metric("p-val cointégration", f"{coint(prices[ticker_a], prices[ticker_b])[1]:.3f}")
    except Exception:
        kpi[1].metric("p-val cointégration", "—")
    kpi[2].metric("Half-life OU", f"{hl:.1f}" if not pd.isna(hl) else "—")
    kpi[3].metric("Z-score actuel", f"{z.iloc[-1]:.2f}")

    # ✅ Série → DataFrame avant plotting
    sp_df = spread.reset_index()
    sp_df.columns = ["Date", "Spread"]
    fig_sp = px.line(sp_df, x="Date", y="Spread", **range_kwargs(start_date,end_date))
    st.plotly_chart(fig_sp, use_container_width=True, theme="streamlit")
    st.caption("**Idée** — Spread = B − b·A; si cointégrés, le spread est mean-reverting. **Trades** (éducatif) : |Z|>2 ⇒ contrarien; sortie vers Z≈0.")

    st.subheader("Z-score du spread (fenêtre 60)")
    Z = z.rename("Z").dropna().reset_index()
    Z.columns = ["Date", "Z"]
    fig_z = px.line(Z, x="Date", y="Z", **range_kwargs(start_date,end_date))
    fig_z.add_hline(y=0, line_dash="dot"); fig_z.add_hline(y=2, line_dash="dash", annotation_text="+2")
    fig_z.add_hline(y=-2, line_dash="dash", annotation_text="-2")
    st.plotly_chart(fig_z, use_container_width=True, theme="streamlit")
    st.caption("**Lecture** — Zones ±2σ = anomalies; surveille la durée passée en extrême et la vitesse de reversion (half-life).")

# ========== ADVANCED RISK ==========
with tab_risk:
    st.subheader("EVT (POT–GPD) sur pertes (VaR/ES extrêmes)")
    cols = st.columns(2)
    for i,(label, r) in enumerate([(ticker_a, -rets[ticker_a]), (ticker_b, -rets[ticker_b])]):
        r = r.dropna()
        if len(r)<200:
            cols[i].warning(f"{label}: série trop courte pour EVT.")
            continue
        u = np.quantile(r, 0.95)         # seuil 95% des pertes
        excess = r[r>u] - u
        try:
            c, loc, scale = genpareto.fit(excess.values, floc=0)
            p_tail = (excess.size)/len(r) # prob. d’excès
            p = 0.99
            if c != 0:
                var_p = u + scale/c * (((1-p)/p_tail)**(-c) - 1)
            else:
                var_p = u + scale*np.log(p_tail/(1-p))
            es_p = (var_p + (scale - c*u)/(1-c)) if c<1 else np.nan
            cols[i].metric(f"EVT VaR99 {label}", pct(var_p))
            cols[i].metric(f"EVT ES99  {label}", pct(es_p))
        except Exception:
            cols[i].warning(f"{label}: échec ajustement GPD.")
    st.caption("**Pourquoi** — EVT modélise les **queues** extrêmes mieux que la normale; pertinent pour le **risk-management** en stress.")

    st.subheader("CoVaR (Régression quantile, q=5%)")
    try:
        qa = rets[ticker_a].quantile(0.05)
        qb = rets[ticker_b].quantile(0.05)
        qmod = QuantReg(rets[ticker_b], sm.add_constant(rets[ticker_a])).fit(q=0.05)
        covar_b_a = float(qmod.predict([1, qa])[0])
        qmod_med = QuantReg(rets[ticker_b], sm.add_constant(rets[ticker_a])).fit(q=0.5)
        d_covar_b_a = covar_b_a - float(qmod_med.predict([1, rets[ticker_a].median()])[0])

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
    st.caption("**Interprétation** — CoVaR mesure le **risque conditionnel**: pertes de B **données** une détresse d’A (quantile bas).")

    st.subheader("Régimes markoviens (variance switching) — 2 états")
    if enable_heavy:
        try:
            y = (rets.mean(axis=1)).dropna()
            mod = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
            res = mod.fit(disp=False)
            pr = res.smoothed_marginal_probabilities[1]  # régime de variance élevée
            pr_df = pr.reset_index()
            pr_df.columns = ["Date", "ProbHighVar"]
            fig_m = px.line(pr_df, x="Date", y="ProbHighVar", **range_kwargs(start_date,end_date))
            st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")
            st.caption("**Usage** — Détecte régimes calmes vs turbulents; utile pour **switching** de levier/exposition.")
        except Exception:
            st.info("Régimes markoviens indisponibles (optimisation). Essaie une période plus longue.")
    else:
        st.info("Active “Modèles lourds” pour estimer les régimes markoviens.")

# ========== DIAGNOSTICS ==========
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
    st.subheader("Tests statistiques (p-values)")
    st.dataframe(grid, use_container_width=True)
    st.caption("**Rappels** — ADF: stationnarité; Ljung-Box: autocorr; ARCH: hétéroscédasticité; JB: normalité; Granger: causalité temporelle. Seuils usuels p<0,05.")

    st.subheader("Histogrammes des rendements")
    hc = st.columns(2)
    hc[0].plotly_chart(px.histogram(ra, nbins=60, labels={"value":f"Rendements {ticker_a}"}), use_container_width=True, theme="streamlit")
    hc[1].plotly_chart(px.histogram(rb, nbins=60, labels={"value":f"Rendements {ticker_b}"}), use_container_width=True, theme="streamlit")
    st.caption("**Astuce** — Queue lourde et asymétrie visibles à l’œil; valide/infirme l’hypothèse gaussienne.")

# ========== DÉPENDANCES DE QUEUE & LAGS ==========
with tab_dep:
    st.subheader("Corrélations de queue (conditionnelles)")
    lc = tail_corr(rets[ticker_a], rets[ticker_b], q=0.05, upper=False)
    uc = tail_corr(rets[ticker_a], rets[ticker_b], q=0.05, upper=True)
    cc = st.columns(2)
    cc[0].metric("Corrél. queue basse (5%)", f"{lc:.3f}" if not pd.isna(lc) else "—")
    cc[1].metric("Corrél. queue haute (95%)", f"{uc:.3f}" if not pd.isna(uc) else "—")
    st.caption("**Idée** — La dépendance en queue **basse** (co-krash) est cruciale pour la gestion des risques; la queue **haute** aide à comprendre les rallies synchrones.")

    st.subheader("Cross-correlation lead/lag")
    xct = xcorr_at_lags(rets[ticker_a], rets[ticker_b], lags=lags_xcorr)
    st.dataframe(xct.style.format({"corr A→B":"{:.3f}", "corr B→A":"{:.3f}"}), use_container_width=True)
    st.caption("**Lecture** — Corr A→B au lag k = corr(B_t, A_{t-k}). Si élevée, A **précède** B (signal prédictif potentiel). Attention aux faux positifs.")

# ========== DATA ==========
with tab_data:
    st.subheader("Paramètres")
    st.write({
        "Tickers": [ticker_a, ticker_b],
        "Fréquence": freq_key,
        "Plage": {"start": str(start_date), "end": str(end_date)},
        "Log returns": log_ret, "RF (ann.)": rf_annual,
        "Fenêtres": {"vol": roll_vol_w, "corr": roll_corr_w, "beta": roll_beta_w},
        "EWMA λ": lam, "Long-only": long_only, "Modèles lourds": enable_heavy,
        "Buffer jours (fetch)": pre_buffer_days,
        "Couverture observée": f"{coverage:.0%}"
    })

    st.subheader("Prix (filtrés sur la plage)")
    st.dataframe(prices, use_container_width=True)
    st.download_button("📥 PRIX (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv")

    st.subheader("Rendements (selon fréquence)")
    st.dataframe(rets, use_container_width=True)
    st.download_button("📥 RENDEMENTS (CSV)", data=rets.to_csv().encode("utf-8"), file_name="returns.csv")

st.caption("⚠️ Outil d'analyse avancée à visée pédagogique. Non destiné au conseil en investissement.")
