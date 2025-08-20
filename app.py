# app.py — Asset Risk & Volatility Dashboard (2 actifs)
# Compatible Streamlit Cloud / Replit — un seul fichier.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import norm
from datetime import date, timedelta

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Asset Risk & Volatility Dashboard", page_icon="📈", layout="wide")
st.title("📊 Asset Risk & Volatility Dashboard — 2 actifs")

FREQ_MAP = {"Journalier (1d)": ("1d", 252, "D"),
            "Hebdo (1wk)": ("1wk", 52, "W-FRI"),
            "Mensuel (1mo)": ("1mo", 12, "M")}
DEFAULT_FREQ = "Journalier (1d)"

# -------------------- UTILS --------------------
def ann_factor(freq_key: str) -> int:
    return FREQ_MAP[freq_key][1]

def interval_code(freq_key: str) -> str:
    return FREQ_MAP[freq_key][0]

def resample_rule(freq_key: str) -> str:
    return FREQ_MAP[freq_key][2]

def format_pct(x, digits=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"

def ensure_two_cols(df: pd.DataFrame, a: str, b: str):
    cols = [c for c in df.columns if str(c) in [a, b]]
    return df[[a, b]].copy() if set([a,b]).issubset(df.columns) else df[cols].copy()

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame | None:
    """Télécharge via yfinance (auto_adjust), convertit en colonnes simples Close, trie et retourne."""
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        return None
    # Normaliser en colonnes Close simples
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.loc[:, pd.IndexSlice[:, "Close"]]
            df.columns = [c[0] for c in df.columns]
        except Exception:
            if "Close" in df.columns:
                df = df["Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    df = df.sort_index()
    return df

def post_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filtre strictement la plage demandée après téléchargement."""
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()

def compute_returns(prices: pd.DataFrame, log=False) -> pd.DataFrame:
    if log:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()

def ewma_stats(returns: pd.DataFrame, lam: float):
    """RiskMetrics: moyenne & covariance EWMA (vectorisée)"""
    r = returns.dropna().values
    if r.size == 0:
        return returns.mean(), returns.cov()
    # Moyenne EWMA
    w = np.array([lam**k for k in range(len(r))])[::-1]
    w = w / w.sum()
    mu = pd.Series((r * w[:,None]).sum(axis=0), index=returns.columns)
    # Covariance EWMA
    x = returns - mu
    S = np.zeros((returns.shape[1], returns.shape[1]))
    for t in range(len(x)):
        v = x.iloc[t].values.reshape(-1,1)
        S = lam * S + (1-lam) * (v @ v.T)
    cov = pd.DataFrame(S, index=returns.columns, columns=returns.columns)
    return mu, cov

def annualize_mean(ret: pd.Series, k: int) -> float:
    return float(ret.mean() * k)

def annualize_vol(ret: pd.Series, k: int) -> float:
    return float(ret.std(ddof=1) * np.sqrt(k))

def sharpe_ratio(ret: pd.Series, k: int, rf_annual: float) -> float:
    if ret.dropna().empty:
        return np.nan
    rf_periodic = (1 + rf_annual) ** (1 / k) - 1
    excess = ret.mean() - rf_periodic
    vol = ret.std(ddof=1)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return float(excess / vol * np.sqrt(k))

def sortino_ratio(ret: pd.Series, k: int, rf_annual: float) -> float:
    rf_periodic = (1 + rf_annual) ** (1 / k) - 1
    downside = ret[ret < 0]
    if downside.dropna().empty:
        return np.nan
    dd_vol = downside.std(ddof=1)
    if dd_vol == 0 or pd.isna(dd_vol):
        return np.nan
    mean_excess = ret.mean() - rf_periodic
    return float(mean_excess / dd_vol * np.sqrt(k))

def max_drawdown(prices: pd.Series) -> float:
    return float((prices / prices.cummax() - 1).min())

def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1.0

def beta_alpha_ols(x: pd.Series, y: pd.Series):
    """y ~ a + b x"""
    df = pd.concat([x, y], axis=1).dropna()
    if df.empty: return np.nan, np.nan, np.nan
    X = sm.add_constant(df.iloc[:,0])
    model = sm.OLS(df.iloc[:,1], X).fit()
    return float(model.params[1]), float(model.params[0]), float(model.rsquared)

def rolling_beta(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """β_t = Cov_t(x,y)/Var_t(x) (équiv. OLS sans intercept)"""
    xy = pd.concat([x, y], axis=1).dropna()
    cov = xy[x.name].rolling(window).cov(xy[y.name])
    var = xy[x.name].rolling(window).var()
    return (cov / var).rename("Rolling β")

def var_es_hist(ret: pd.Series, q=0.95):
    r = ret.dropna().values
    if r.size == 0: return np.nan, np.nan
    qv = np.quantile(r, 1 - q)
    var_loss = -float(qv)
    es_loss = -float(r[r <= qv].mean()) if (r <= qv).any() else np.nan
    return var_loss, es_loss

def cornish_fisher_var(ret: pd.Series, q=0.95, k: int = 252):
    """VaR Cornish–Fisher annualisée à partir de moments (skew, kurt)"""
    x = ret.dropna()
    mu, sig = x.mean(), x.std(ddof=1)
    S = float(x.skew())
    K = float(x.kurt())  # excès de kurtosis (pandas = Fisher)
    z = norm.ppf(1 - q)
    z_cf = z + (1/6)*(z**2-1)*S + (1/24)*(z**3-3*z)*K - (1/36)*(2*z**3-5*z)*S**2
    var_daily = -(mu + z_cf * sig)
    return float(var_daily)

def cagr_from_prices(prices: pd.Series, k: int) -> float:
    if len(prices) < 2: return np.nan
    T = len(prices) / k
    total = float(prices.iloc[-1] / prices.iloc[0])
    return total**(1/T) - 1 if T > 0 else np.nan

def half_life_ou(series: pd.Series) -> float:
    """Half-life de mean reversion via AR(1): X_t = ρ X_{t-1} + ε_t ; HL = -ln(2)/ln(ρ)"""
    x = series.dropna()
    if len(x) < 30: return np.nan
    y = x.shift(1).dropna()
    z = x.loc[y.index]
    rho, _, _, _ = np.linalg.lstsq(y.values.reshape(-1,1), z.values, rcond=None)
    rho = float(rho[0])
    if rho <= 0 or rho >= 1: return np.nan
    return float(-np.log(2) / np.log(rho))

def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    inv = np.linalg.pinv(cov.values)
    ones = np.ones((cov.shape[0], 1))
    w = inv @ ones
    w = w / (ones.T @ inv @ ones)
    return pd.Series(w.flatten(), index=cov.columns)

def tangency_weights(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.Series:
    ones = np.ones(len(mu))
    mu_excess = mu.values - rf
    inv = np.linalg.pinv(cov.values)
    w = inv @ mu_excess
    w = w / (ones @ w)  # somme=1
    return pd.Series(w, index=mu.index)

def long_only_projection(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0)
    s = w.sum()
    return w / s if s > 0 else w

def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, k: int, rf: float):
    mu_p = float(np.dot(w.values, mu.values))
    vol_p = float(np.sqrt(w.values @ cov.values @ w.values))
    sharpe = (mu_p - rf) / vol_p if vol_p > 0 else np.nan
    return mu_p, vol_p, sharpe

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    col_t = st.columns(2)
    with col_t[0]:
        ticker_a = st.text_input("Ticker A", "AAPL")
    with col_t[1]:
        ticker_b = st.text_input("Ticker B", "MSFT")

    freq_key = st.selectbox("Fréquence", list(FREQ_MAP.keys()), index=list(FREQ_MAP.keys()).index(DEFAULT_FREQ))
    k = ann_factor(freq_key)
    interval = interval_code(freq_key)

    today = date.today()
    dcol = st.columns(2)
    with dcol[0]:
        start_date = st.date_input("Date début", value=today - timedelta(days=365*2))
    with dcol[1]:
        end_date = st.date_input("Date fin", value=today)

    log_returns = st.checkbox("Rendements logarithmiques", False)
    rf_annual = st.number_input("Taux sans risque annualisé (%)", value=0.0, step=0.25) / 100.0

    roll_vol_win = st.slider(f"Fenêtre vol ({freq_key})", 10, 200, 30, step=5)
    roll_corr_win = st.slider(f"Fenêtre corr ({freq_key})", 10, 260, 60, step=5)
    roll_beta_win = st.slider(f"Fenêtre β ({freq_key})", 10, 260, 60, step=5)

    with st.expander("Paramètres avancés (portfolio & risques)"):
        lam = st.slider("λ EWMA (RiskMetrics)", 0.80, 0.99, 0.94, 0.01)
        long_only = st.checkbox("Contraindre portefeuilles long-only", True)
        use_ewma = st.checkbox("Utiliser moyennes/covariances EWMA", True)

    st.caption("Source: Yahoo Finance via yfinance — Les graphiques reflètent **strictement** la plage de dates sélectionnée.")

# -------------------- DATA --------------------
raw = fetch_prices([ticker_a, ticker_b], str(start_date), str(end_date), interval)
if raw is None or raw.empty:
    st.warning("Impossible de télécharger les prix. Vérifie les tickers, la plage et la fréquence.")
    st.stop()

# Filtre strict de la plage
prices = post_filter(ensure_two_cols(raw, ticker_a, ticker_b).ffill().dropna(), start_date, end_date)
if prices.empty:
    st.warning("Aucune donnée après filtrage. Essaie d'élargir la période.")
    st.stop()

# Rendements
rets = compute_returns(prices, log_returns).dropna()
if rets.empty:
    st.warning("Pas assez de points de rendements après nettoyage.")
    st.stop()

# Choix stats (moyenne/cov)
if use_ewma:
    mu_est, cov_est = ewma_stats(rets, lam)
else:
    mu_est, cov_est = rets.mean(), rets.cov()

mu_ann = mu_est * k  # annualisation simple
cov_ann = cov_est * k

# -------------------- OVERVIEW --------------------
tab_over, tab_roll, tab_port, tab_pair, tab_diag, tab_data = st.tabs(
    ["Overview", "Rolling & Corr", "Portfolio", "Pair Trading", "Diagnostics", "Data"]
)

with tab_over:
    # KPIs base
    corr = rets[[ticker_a, ticker_b]].corr().iloc[0,1]
    beta, alpha, r2 = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
    try:
        coint_res = coint(prices[ticker_a], prices[ticker_b])
        coint_p = float(coint_res[1])
    except Exception:
        coint_p = np.nan

    c1 = st.columns(4)
    c1[0].metric("Corrélation", f"{corr:.3f}")
    c1[1].metric("β (B sur A)", f"{beta:.3f}", help=f"Régression {ticker_b} ~ {ticker_a}")
    c1[2].metric("α (par période)", f"{alpha:.5f}")
    c1[3].metric("R²", f"{r2:.3f}")
    st.caption(f"Cointégration (Engle–Granger) p-value = **{coint_p:.3f}** (p < 0.05 ⇒ cointégration plausible).")

    # Table indicateurs par actif
    def metrics_block(P: pd.Series, R: pd.Series):
        var95, es95 = var_es_hist(R, 0.95)
        return {
            "Dernier prix": float(P.iloc[-1]),
            f"CAGR (~{k}/an)": cagr_from_prices(P, k),
            "Vol (ann.)": annualize_vol(R, k),
            "Sharpe": sharpe_ratio(R, k, rf_annual),
            "Sortino": sortino_ratio(R, k, rf_annual),
            "Max DD": max_drawdown(P),
            "VaR95 (hist/period)": var95,
            "ES95 (hist/period)": es95,
            "VaR95 (Cornish-Fisher)": cornish_fisher_var(R, 0.95, k),
            "Skew": float(R.dropna().skew()),
            "Kurtosis (excess)": float(R.dropna().kurt())
        }

    mA = metrics_block(prices[ticker_a], rets[ticker_a])
    mB = metrics_block(prices[ticker_b], rets[ticker_b])
    table = pd.DataFrame([mA, mB], index=[ticker_a, ticker_b])
    # format pour l'affichage
    fmt_cols_pct = ["CAGR (~{}/an)".format(k), "Vol (ann.)", "Max DD"]
    for c in table.columns:
        if "VaR95" in c or "ES95" in c or c in fmt_cols_pct:
            table[c] = table[c].apply(lambda x: format_pct(x))
    st.subheader("Indicateurs clés")
    st.dataframe(table, use_container_width=True)

    # Prix indexés
    st.subheader("Prix indexés (base=100)")
    idx = prices / prices.iloc[0] * 100
    fig_idx = px.line(idx, x=idx.index, y=idx.columns, labels={"value":"Indice", "variable":"Actif", "index":"Date"})
    st.plotly_chart(fig_idx, use_container_width=True, theme="streamlit")

    # Dispersion & droite OLS
    st.subheader("Dispersion des rendements (B vs A) + droite OLS")
    XY = rets[[ticker_a, ticker_b]].dropna().copy()
    X = sm.add_constant(XY[ticker_a])
    model = sm.OLS(XY[ticker_b], X).fit()
    XY["pred"] = model.predict(X)
    fig_sc = px.scatter(XY, x=ticker_a, y=ticker_b, opacity=0.6)
    fig_sc.add_traces(px.line(XY.sort_values(ticker_a), x=ticker_a, y="pred").data)
    st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")

    # Drawdowns
    st.subheader("Drawdowns cumulés")
    dd = pd.concat([drawdown_series(prices[ticker_a]).rename(ticker_a),
                    drawdown_series(prices[ticker_b]).rename(ticker_b)], axis=1)
    fig_dd = px.line(dd, x=dd.index, y=dd.columns, labels={"value":"Drawdown", "variable":"Actif", "index":"Date"})
    st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")

with tab_roll:
    st.subheader("Volatilité glissante (annualisée)")
    roll_vol = rets.rolling(roll_vol_win).std(ddof=1) * np.sqrt(k)
    st.plotly_chart(px.line(roll_vol, x=roll_vol.index, y=roll_vol.columns,
                            labels={"value":"Vol ann.", "variable":"Actif", "index":"Date"}),
                    use_container_width=True, theme="streamlit")

    st.subheader("Corrélation glissante")
    rc = rets[ticker_a].rolling(roll_corr_win).corr(rets[ticker_b]).to_frame("Corrélation")
    st.plotly_chart(px.line(rc, x=rc.index, y="Corrélation"),
                    use_container_width=True, theme="streamlit")

    st.subheader("β glissant (B sur A)")
    rb = rolling_beta(rets[ticker_a], rets[ticker_b], roll_beta_win).dropna()
    st.plotly_chart(px.line(rb, x=rb.index, y=rb.values, labels={"y":"β glissant"}), use_container_width=True, theme="streamlit")

with tab_port:
    st.subheader("Hypothèses d'entrée (annuelles)")
    c0, c1, c2 = st.columns(3)
    c0.metric(f"μ̂ {ticker_a}", format_pct(mu_ann[ticker_a]))
    c1.metric(f"μ̂ {ticker_b}", format_pct(mu_ann[ticker_b]))
    c2.metric("Corrélation", f"{rets[[ticker_a, ticker_b]].corr().iloc[0,1]:.3f}")

    st.markdown("**Covariance (annuelle)**")
    st.dataframe(cov_ann.style.format("{:.6f}"), use_container_width=True)

    # Portefeuilles
    mu_vec = mu_ann.copy()
    cov = cov_ann.copy()
    rf = rf_annual

    # Min-variance
    w_min = min_variance_weights(cov)
    if long_only: w_min = long_only_projection(w_min)

    # Tangency (max Sharpe)
    w_tan = tangency_weights(mu_vec, cov, rf)
    if long_only: w_tan = long_only_projection(w_tan)

    # Risk-parity (pour 2 actifs: w ∝ 1/σ)
    vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.columns)
    w_rp = (1/vols) / (1/vols).sum()
    if long_only: w_rp = long_only_projection(w_rp)

    # β-neutral (y ~ a + b x → w=[1, -b], normalisé à somme |w|=1)
    b_beta, _, _ = beta_alpha_ols(rets[ticker_a], rets[ticker_b])
    w_beta = pd.Series({ticker_a: 1.0, ticker_b: -b_beta})
    w_beta = w_beta / w_beta.abs().sum()

    # Stats portefeuilles
    rows = []
    for name, w in [("Equal-Weight", pd.Series([0.5,0.5], index=[ticker_a, ticker_b])),
                    ("Min-Variance", w_min),
                    ("Tangency", w_tan),
                    ("Risk-Parity", w_rp),
                    ("β-neutral", w_beta)]:
        mu_p, vol_p, S_p = portfolio_stats(w, mu_vec, cov, k, rf)
        rows.append({"Portfolio":name,
                     "w_"+ticker_a: w[ticker_a], "w_"+ticker_b: w[ticker_b],
                     "μ (ann.)": mu_p, "σ (ann.)": vol_p, "Sharpe": S_p})
    port_table = pd.DataFrame(rows)
    st.subheader("Portefeuilles — poids & performances (annuelles)")
    st.dataframe(port_table.style.format({f"w_{ticker_a}":"{:.2%}", f"w_{ticker_b}":"{:.2%}",
                                          "μ (ann.)":"{:.2%}", "σ (ann.)":"{:.2%}", "Sharpe":"{:.2f}"}),
                 use_container_width=True)

    # Efficient frontier (2 actifs)
    w_grid = np.linspace(0,1,201)
    mu_a, mu_b = mu_vec[ticker_a], mu_vec[ticker_b]
    var_a, var_b = cov.loc[ticker_a, ticker_a], cov.loc[ticker_b, ticker_b]
    cov_ab = cov.loc[ticker_a, ticker_b]
    mu_pf = w_grid*mu_a + (1-w_grid)*mu_b
    var_pf = (w_grid**2)*var_a + ((1-w_grid)**2)*var_b + 2*w_grid*(1-w_grid)*cov_ab
    sig_pf = np.sqrt(var_pf)
    fig_front = px.line(x=sig_pf, y=mu_pf, labels={"x":"σ (ann.)", "y":"μ (ann.)"}, title="Efficient Frontier (2 actifs)")
    # Points spéciaux
    mu_min, sig_min, _ = portfolio_stats(w_min, mu_vec, cov, k, rf)
    mu_tan, sig_tan, _ = portfolio_stats(w_tan, mu_vec, cov, k, rf)
    fig_front.add_scatter(x=[sig_min], y=[mu_min], mode="markers", name="Min-Var")
    fig_front.add_scatter(x=[sig_tan], y=[mu_tan], mode="markers", name="Tangency")
    st.plotly_chart(fig_front, use_container_width=True, theme="streamlit")

    # Backtest cumulé des portefeuilles (périodique)
    port_weights = {
        "Equal": pd.Series([0.5,0.5], index=[ticker_a, ticker_b]),
        "MinVar": w_min, "Tangency": w_tan, "RiskParity": w_rp, "BetaNeutral": w_beta
    }
    cum = pd.DataFrame(index=rets.index)
    for name, w in port_weights.items():
        r = (rets * w).sum(axis=1)
        cum[name] = (1 + r).cumprod()
    st.subheader("Backtest — cumul (base=1)")
    st.plotly_chart(px.line(cum, x=cum.index, y=cum.columns), use_container_width=True, theme="streamlit")

with tab_pair:
    st.subheader("Cointegration & Spread (OLS sur prix)")
    # hedge ratio b: P_b ~ a + b P_a → spread = P_b - b P_a
    X = sm.add_constant(prices[ticker_a])
    model = sm.OLS(prices[ticker_b], X).fit()
    hedge_b = float(model.params[1])
    spread = prices[ticker_b] - hedge_b * prices[ticker_a]
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std(ddof=1)
    hl = half_life_ou(spread - spread.mean())

    c2 = st.columns(4)
    c2[0].metric("Hedge ratio (b)", f"{hedge_b:.3f}")
    c2[1].metric("p-value cointégration", f"{coint(prices[ticker_a], prices[ticker_b])[1]:.3f}")
    c2[2].metric("Half-life (périodes)", f"{hl:.1f}" if not pd.isna(hl) else "—")
    c2[3].metric("Z-score actuel", f"{z.iloc[-1]:.2f}")

    st.plotly_chart(px.line(spread, labels={"value":"Spread", "index":"Date"}), use_container_width=True, theme="streamlit")
    st.subheader("Z-score du spread (fenêtre 60)")
    z_plot = z.to_frame("Z").dropna()
    fig_z = px.line(z_plot, x=z_plot.index, y="Z")
    fig_z.add_hline(y=0, line_dash="dot")
    fig_z.add_hline(y=2, line_dash="dash", annotation_text="+2")
    fig_z.add_hline(y=-2, line_dash="dash", annotation_text="-2")
    st.plotly_chart(fig_z, use_container_width=True, theme="streamlit")

    st.caption("Idée (éducative) : |Z|>2 ⇒ écart extrême potentiel; reversion attendue vers 0 (pas un conseil).")

with tab_diag:
    st.subheader("Tests de base (rendements)")
    ra, rb = rets[ticker_a].dropna(), rets[ticker_b].dropna()

    # ADF (stationnarité)
    def adf_p(x):
        try:
            return float(adfuller(x.dropna())[1])
        except Exception:
            return np.nan
    p_adf_a, p_adf_b = adf_p(ra), adf_p(rb)

    # Ljung-Box (auto-corrélation)
    def ljung_p(x, lags=10):
        try:
            return float(acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            return np.nan
    p_lb_a, p_lb_b = ljung_p(ra), ljung_p(rb)

    # ARCH (hétéroscédasticité)
    def arch_p(x, lags=10):
        try:
            return float(het_arch(x.dropna(), nlags=lags)[1])
        except Exception:
            return np.nan
    p_arch_a, p_arch_b = arch_p(ra), arch_p(rb)

    # Jarque–Bera (normalité)
    def jb_p(x):
        try:
            return float(jarque_bera(x.dropna())[1])
        except Exception:
            return np.nan
    p_jb_a, p_jb_b = jb_p(ra), jb_p(rb)

    # Granger causality (min p-value sur lags)
    def granger_min_p(x_from, y_to, maxlag=5):
        try:
            df = pd.concat([y_to, x_from], axis=1).dropna()
            if len(df) < 50: return np.nan
            res = grangercausalitytests(df, maxlag=maxlag, verbose=False)
            pvals = [res[L][0]["ssr_ftest"][1] for L in range(1, maxlag+1)]
            return float(np.nanmin(pvals))
        except Exception:
            return np.nan

    p_a_to_b = granger_min_p(ra, rb, 5)
    p_b_to_a = granger_min_p(rb, ra, 5)

    grid = pd.DataFrame({
        "Test": ["ADF", "Ljung-Box(10)", "ARCH(10)", "Jarque–Bera", "Granger A→B", "Granger B→A"],
        ticker_a: [p_adf_a, p_lb_a, p_arch_a, p_jb_a, p_a_to_b, np.nan],
        ticker_b: [p_adf_b, p_lb_b, p_arch_b, p_jb_b, np.nan, p_b_to_a],
    })
    st.dataframe(grid, use_container_width=True)

    st.subheader("Histogrammes")
    hc = st.columns(2)
    hc[0].plotly_chart(px.histogram(ra, nbins=60, labels={"value":f"Rendements {ticker_a}"}), use_container_width=True)
    hc[1].plotly_chart(px.histogram(rb, nbins=60, labels={"value":f"Rendements {ticker_b}"}), use_container_width=True)

with tab_data:
    st.subheader("Paramètres actuels")
    st.write({
        "Ticker A": ticker_a, "Ticker B": ticker_b,
        "Fréquence": freq_key, "λ EWMA": lam, "Long-only": long_only,
        "Log returns": log_returns, "Taux sans risque (ann.)": rf_annual,
        "Fenêtres": {"vol": roll_vol_win, "corr": roll_corr_win, "beta": roll_beta_win},
        "Plage": {"start": str(start_date), "end": str(end_date)}
    })

    st.subheader("Prix (filtrés sur la plage)")
    st.dataframe(prices, use_container_width=True)
    st.download_button("📥 Export PRIX (CSV)", data=prices.to_csv().encode("utf-8"), file_name="prices.csv")

    st.subheader("Rendements")
    st.dataframe(rets, use_container_width=True)
    st.download_button("📥 Export RENDEMENTS (CSV)", data=rets.to_csv().encode("utf-8"), file_name="returns.csv")

# -------------------- DISCLAIMER --------------------
st.caption("⚠️ Outil pédagogique avancé. Ce n’est pas un conseil en investissement.")
