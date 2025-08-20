# =============================
# File: streamlit_app.py (v3 — plotting fix + hardening)
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title="Asset Pair Risk & Volatility Dashboard",
    layout="wide",
)

# =============================
# Utilities — robust data layer
# =============================

INTRADAY_CHUNK_DAYS = {
    "15m": 59,   # Yahoo limite ~60 jours pour <=30m
    "30m": 59,
    "1h": 729,   # ~2 ans pour 1h (approx)
}

ANNUALIZATION = {"1d":252, "1h":252*6.5, "30m":252*6.5*2, "15m":252*6.5*4, "1wk":52, "1mo":12}


def _to_utc_naive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        df = df.copy()
        df.index = idx.tz_convert('UTC').tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def fetch_prices_full_range(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetches *all* data across [start,end] with chunking for intraday intervals.
    Ensures no silent truncation by Yahoo. Returns a one-column DF named after the ticker.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)  # include end date fully

    # Choose chunking strategy
    chunks = []
    if interval in INTRADAY_CHUNK_DAYS:
        step = relativedelta(days=INTRADAY_CHUNK_DAYS[interval])
        t0 = start_dt
        while t0 < end_dt:
            t1 = min(t0 + step, end_dt)
            df = yf.download(
                ticker,
                start=t0,
                end=t1,
                interval=interval,
                auto_adjust=True,
                prepost=False,
                progress=False,
            )
            df = _to_utc_naive(df)
            if not df.empty:
                chunks.append(df)
            t0 = t1
        data = pd.concat(chunks).sort_index().drop_duplicates()
    else:
        data = yf.download(
            ticker, start=start_dt, end=end_dt, interval=interval,
            auto_adjust=True, prepost=False, progress=False
        )
        data = _to_utc_naive(data)

    if data.empty:
        return pd.DataFrame()

    out = data[["Close"]].rename(columns={"Close": ticker}).dropna()

    # Trim strictly to [start_dt, end_dt)
    out = out.loc[(out.index >= start_dt) & (out.index < end_dt)]
    return out


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    rets = np.log(prices/prices.shift(1)) if log else prices.pct_change()
    return rets.dropna(how='all')


def annualize_vol(returns: pd.Series, interval: str) -> float:
    periods = ANNUALIZATION.get(interval, 252)
    return returns.std(ddof=1) * np.sqrt(periods)


def drawdown(prices: pd.Series) -> pd.DataFrame:
    cum_max = prices.cummax()
    dd = prices/cum_max - 1.0
    return pd.DataFrame({"price":prices, "cum_max":cum_max, "drawdown":dd})


def hist_var(series: pd.Series, alpha: float = 0.95) -> tuple:
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan
    var = np.quantile(s, 1-alpha)
    cvar = s[s <= var].mean() if (s <= var).any() else np.nan
    return var, cvar


def pearson_ci(r: float, n: int, level: float = 0.95):
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1/np.sqrt(n-3)
    zcrit = stats.norm.ppf(0.5 + level/2)
    lo, hi = z - zcrit*se, z + zcrit*se
    return np.tanh(lo), np.tanh(hi)


def tail_comovement(x: pd.Series, y: pd.Series, q: float = 0.05):
    """Probability of joint extreme events: P(X<=q & Y<=q) and P(X>=1-q & Y>=1-q)."""
    xq_lo, xq_hi = x.quantile(q), x.quantile(1-q)
    yq_lo, yq_hi = y.quantile(q), y.quantile(1-q)
    lo = ((x <= xq_lo) & (y <= yq_lo)).mean()
    hi = ((x >= xq_hi) & (y >= yq_hi)).mean()
    return lo, hi


def rolling_beta(x: pd.Series, y: pd.Series, window: int = 60) -> pd.Series:
    def _beta(a, b):
        va = np.var(a)
        if not np.isfinite(va) or np.isclose(va, 0.0):
            return np.nan
        return np.cov(a, b)[0,1] / va
    out = pd.concat([x,y], axis=1).rolling(window).apply(lambda w: _beta(w[:,0], w[:,1]), raw=True)
    # Return as Series with same index as inputs
    out = out.iloc[:,0]
    out.name = "beta"
    return out

# -------- Plot helpers to avoid Plotly Express pitfalls --------

def line_from_series(s: pd.Series, name: str = None, yaxis_title: str = None) -> go.Figure:
    s = s.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name or (s.name or "value")))
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig


def line_from_df(df: pd.DataFrame, ycols: list, yaxis_title: str = None) -> go.Figure:
    fig = go.Figure()
    for col in ycols:
        series = df[col].dropna()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=col))
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig

# ---------------------
# Sidebar — inputs
# ---------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Tickers **Yahoo Finance** : actions, ETF, indices `^GSPC`, forex `EURUSD=X`, crypto `BTC-USD`. 

**Remarque couverture** : pour les intervalles intraday, Yahoo impose des bornes d'historique — ici, la collecte est **segmentée** pour couvrir **toute** la plage.")

    t1 = st.text_input("Ticker 1", "BTC-USD").strip()
    t2 = st.text_input("Ticker 2", "ETH-USD").strip()

    default_start = (datetime.utcnow() - relativedelta(years=2)).date()
    start = st.date_input("Début (inclus)", default_start)
    end = st.date_input("Fin (inclus)", datetime.utcnow().date())

    interval = st.selectbox("Intervalle", ["1d","1h","30m","15m","1wk","1mo"], index=0)
    logret = st.checkbox("Rendements logarithmiques", value=False)
    win_corr = st.slider("Fenêtre (barres) : corrélation/volatilité/bêta", 20, 252, 90, 5)
    rf = st.number_input("Taux sans risque annualisé (%)", value=0.0, step=0.25)/100.0

    st.markdown("---")
    st.caption("Exemples : AAPL, MSFT, SPY, ^FCHI, EURUSD=X, BTC-USD, ETH-USD, GC=F, CL=F")

# ---------------------
# Data — full-range fetch & diagnostics
# ---------------------
start_str, end_str = str(start), str(end)
prices1 = fetch_prices_full_range(t1, start_str, end_str, interval)
prices2 = fetch_prices_full_range(t2, start_str, end_str, interval)

if prices1.empty or prices2.empty:
    st.error("❌ Impossible de charger les données pour au moins un ticker. Vérifiez symboles/intervalle.")
    st.stop()

# Align on common timeline
prices = prices1.join(prices2, how='inner').dropna()
prices.columns = [t1, t2]

# Coverage diagnostics
req_start = pd.to_datetime(start_str)
req_end = pd.to_datetime(end_str) + pd.Timedelta(days=1)
loaded_start, loaded_end = prices.index.min(), prices.index.max()

coverage_info = {
    "Requête de": [req_start.strftime('%Y-%m-%d')],
    "à": [ (pd.to_datetime(end_str)).strftime('%Y-%m-%d') ],
    "Données de": [loaded_start.strftime('%Y-%m-%d %H:%M')],
    "à (dernière barre)": [loaded_end.strftime('%Y-%m-%d %H:%M')],
}

with st.expander("🔎 Diagnostic couverture des données"):
    st.write(pd.DataFrame(coverage_info))
    if loaded_start > req_start + pd.Timedelta(hours=1) or loaded_end < req_end - pd.Timedelta(hours=1):
        st.warning("La plage récupérée est plus courte que demandé. Les limites viennent de la disponibilité Yahoo pour cet intervalle/ticker. La collecte **segmentée** utilisée ici maximise néanmoins la couverture.")
    else:
        st.success("Couverture conforme à la plage demandée.")

# ---------------------
# Returns & risk metrics
# ---------------------
rets = compute_returns(prices, log=logret)
ret1, ret2 = rets[t1], rets[t2]

periods = ANNUALIZATION.get(interval, 252)
vol1 = annualize_vol(ret1, interval)
vol2 = annualize_vol(ret2, interval)
ann_mean1 = ret1.mean() * periods
ann_mean2 = ret2.mean() * periods
sharpe1 = (ann_mean1 - rf)/vol1 if vol1>0 else np.nan
sharpe2 = (ann_mean2 - rf)/vol2 if vol2>0 else np.nan

corr_full = ret1.corr(ret2)
# Pearson significance
nobs = rets.dropna().shape[0]
if np.isfinite(corr_full) and nobs>3:
    tstat = corr_full*np.sqrt((nobs-2)/(1-corr_full**2))
    pval_corr = 2*(1 - stats.t.cdf(abs(tstat), df=nobs-2))
    ci_lo, ci_hi = pearson_ci(corr_full, nobs, 0.95)
else:
    pval_corr, ci_lo, ci_hi = np.nan, np.nan, np.nan

roll_corr = rets[t1].rolling(win_corr).corr(rets[t2])
roll_vol1 = ret1.rolling(win_corr).std()*np.sqrt(periods)
roll_vol2 = ret2.rolling(win_corr).std()*np.sqrt(periods)
roll_b = rolling_beta(ret1, ret2, window=win_corr)  # beta of t2 relative to t1

var1, cvar1 = hist_var(ret1, 0.95)
var2, cvar2 = hist_var(ret2, 0.95)

# Cointegration & spread diagnostics
try:
    c_score, c_pvalue, _ = coint(prices[t1].dropna(), prices[t2].dropna())
    # Hedge ratio via OLS-style cov/var
    beta_hat = np.cov(prices[t1], prices[t2])[0,1]/np.var(prices[t1]) if np.var(prices[t1])>0 else 1.0
    spread = prices[t2] - beta_hat*prices[t1]
    adf_stat, adf_pvalue, *_ = adfuller(spread.dropna(), maxlag=1, autolag='AIC')
except Exception:
    c_pvalue, beta_hat, spread, adf_pvalue = np.nan, np.nan, pd.Series(dtype=float), np.nan

lo_tail, hi_tail = tail_comovement(ret1, ret2, q=0.05)

# ---------------------
# Header KPIs + Explanations
# ---------------------
st.title("📊 Asset Pair Risk & Volatility Dashboard")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Corrélation (plein échantillon)", f"{corr_full:.3f}")
with c2:
    st.metric(f"Vol {t1} (ann.)", f"{vol1:.2%}")
    st.metric(f"Vol {t2} (ann.)", f"{vol2:.2%}")
with c3:
    st.metric(f"Sharpe {t1}", f"{sharpe1:.2f}")
    st.metric(f"Sharpe {t2}", f"{sharpe2:.2f}")
with c4:
    st.metric("Cointegration p-value", f"{c_pvalue:.3f}" if np.isfinite(c_pvalue) else "N/A")
    st.metric("ADF p-value (spread)", f"{adf_pvalue:.3f}" if np.isfinite(adf_pvalue) else "N/A")

with st.expander("ℹ️ Comment lire ces KPIs"):
    st.markdown(
        f"""
        - **Corrélation** : co-variation des rendements. *p-value* ≈ {pval_corr:.3g} ; IC95% ≈ [{ci_lo:.2f}, {ci_hi:.2f}].
        - **Volatilité annualisée** : écart-type × √(périodes/an). Comparez à intervalle identique.
        - **Sharpe** : rendement excédentaire par unité de risque (indicatif).
        - **Cointégration / ADF** : relation d'équilibre de long terme (spread stationnaire) ⇒ pairs trading possible si stable.
        """
    )

# ---------------------
# Charts + Technical guidance
# ---------------------
st.subheader("Prix normalisés (base 100 au début de la fenêtre)")
norm = prices/prices.iloc[0]*100.0
fig_prices = line_from_df(norm, [t1, t2], yaxis_title="Index 100=Début")
st.plotly_chart(fig_prices, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : comparer les trajectoires de prix en neutralisant l'échelle absolue.\
    **Lecture** : écart croissant ⇒ sur/sous-performance relative.\
    **Tip** : si des barres semblent manquer, vérifiez **Diagnostic couverture**.
    """)

st.subheader("Corrélation glissante (fenêtre = {} barres)".format(win_corr))
fig_corr = line_from_series(roll_corr, name="Corrélation", yaxis_title="Corrélation")
st.plotly_chart(fig_corr, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : suivre la **stabilité** de la corrélation.\
    **Lecture** : >0.7 = forte co-mouvance ; <0 = décorrélation. Grande fenêtre = plus lisse, moins réactif.
    """)

st.subheader("Volatilité glissante (annualisée)")
fig_rv = go.Figure()
fig_rv.add_trace(go.Scatter(x=roll_vol1.index, y=roll_vol1.values, name=f"Vol {t1}"))
fig_rv.add_trace(go.Scatter(x=roll_vol2.index, y=roll_vol2.values, name=f"Vol {t2}"))
fig_rv.update_layout(yaxis_title="Vol ann.")
st.plotly_chart(fig_rv, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : profiler le **risque** dans le temps. Pics = stress; plateaux = accalmie. Utilisez pour calibrer l'expo.
    """)

st.subheader("Bêta glissant ({} vs {})".format(t2, t1))
fig_rb = line_from_series(roll_b, name="β glissant", yaxis_title="β")
st.plotly_chart(fig_rb, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : sensibilité du rendement de {t2} au rendement de {t1}.\
    **Lecture** : β>1 = plus risqué que la référence; β<1 = moins sensible. Négatif = mouvement opposé.
    """)

st.subheader("Dispersion des rendements (nuage + droite OLS)")
scatter_df = pd.DataFrame({t1:ret1, t2:ret2}).dropna()
fig_scatter = px.scatter(scatter_df.reset_index(), x=t1, y=t2, trendline="ols")
st.plotly_chart(fig_scatter, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : relation linéaire instantanée.\
    **Lecture** : nuage serré + pente = relation robuste; dispersion = relation faible.
    """)

st.subheader("Drawdowns (pertes relatives au pic)")
colA, colB = st.columns(2)
with colA:
    dd1 = drawdown(prices[t1])
    fig_dd1 = line_from_series(dd1["drawdown"], name=f"DD {t1}", yaxis_title="Drawdown")
    st.plotly_chart(fig_dd1, use_container_width=True)
with colB:
    dd2 = drawdown(prices[t2])
    fig_dd2 = line_from_series(dd2["drawdown"], name=f"DD {t2}", yaxis_title="Drawdown")
    st.plotly_chart(fig_dd2, use_container_width=True)
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **Objectif** : quantifier l'**amplitude** et la **durée** des baisses.\
    **Usage** : utile pour stop-loss, sizing, tolérance au risque.
    """)

st.subheader("Risque de queue & pertes extrêmes")
var_df = pd.DataFrame({
    "Asset":[t1, t2],
    "VaR 95%": [hist_var(ret1, 0.95)[0], hist_var(ret2, 0.95)[0]],
    "CVaR 95%": [hist_var(ret1, 0.95)[1], hist_var(ret2, 0.95)[1]],
}).set_index("Asset")
st.dataframe(var_df)

lo_p, hi_p = lo_tail, hi_tail
st.caption(f"Co-mouvements extrêmes (q=5%) — choc conjoint baissier: {lo_p:.2%} | choc conjoint haussier: {hi_p:.2%}")
with st.expander("📘 Comment l'utiliser"):
    st.markdown("""
    **VaR(95%)** : perte seuil non dépassée ~95% du temps (historique).\
    **CVaR(95%)** : perte **moyenne** conditionnelle au-delà de la VaR.\
    **Co-mouvements extrêmes** : probabilité de chocs *simultanés* (queues). Utile pour stress-tests et couverture.
    """)

# Spread & z-score (pairs trading) si cointégration
if np.isfinite(c_pvalue) and c_pvalue < 0.05 and spread.size>0:
    st.subheader("Spread (y − β·x) & Z-Score")
    z = (spread - spread.rolling(win_corr).mean())/spread.rolling(win_corr).std()
    fig_spread = line_from_df(pd.DataFrame({"Spread":spread, "Z-Score":z}), ["Spread","Z-Score"], yaxis_title="Valeur")
    st.plotly_chart(fig_spread, use_container_width=True)
    with st.expander("📘 Comment l'utiliser"):
        st.markdown("""
        **Objectif** : spread stationnaire si cointégration.\
        **Signal** : z-score ±2/±3 = zones de sur/sous-valorisation relatives ⇒ entrées/sorties potentielles (avec gestion du risque).
        """)
else:
    st.info("Pas de cointégration statistiquement significative (p ≥ 0.05) — pairs trading non recommandé.")

# ---------------------
# Data table & export
# ---------------------
with st.expander("🧾 Voir/Exporter les données (prix & rendements)"):
    table = pd.concat({"Prices":prices, "Returns":rets}, axis=1)
    st.dataframe(table)
    csv = table.to_csv().encode('utf-8')
    st.download_button("📥 Télécharger CSV", data=csv, file_name="pair_data.csv", mime="text/csv")

st.caption("Sources: Yahoo Finance via yfinance (gratuit, sans clé API). Les données intraday sont récupérées par **segments** pour respecter les limites Yahoo et couvrir la plage demandée. Les graphiques utilisent Plotly Graph Objects pour éviter les erreurs de mapping x/y.")

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
# python-dateutil

# =============================
# File: README.md
# =============================
# Asset Pair Risk & Volatility Dashboard (v3)

**Correctifs majeurs** :
- Remplacement de `plotly.express.line` (avec `data_frame` + `x`/`y` conflictuels) par des **helpers** `graph_objects` (séries/DF) ⇒ plus d'erreur `ValueError: Cannot accept list of column references...`.
- Correction d'une coquille `.drop_duplicates()` et harmonisation des noms (`roll_b`).
- Durcissement des cas vides/NaN avant tracé.

## Déploiement (Streamlit Community Cloud)
1. Repo GitHub : `streamlit_app.py` + `requirements.txt`.
2. "Deploy an app" → fichier d'entrée `streamlit_app.py`.
3. Partagez l'URL publique.

## Conseils
- Si un graphe reste vide, vérifiez la **fenêtre** (win_corr) vs. longueur d'historique et l'onglet **Diagnostic couverture**.
- Pour crypto 24/7, ajustez si besoin l'annualisation (paramètre `ANNUALIZATION`).
