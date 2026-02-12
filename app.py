import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha Pro | Finance & Arbitrage",
    page_icon="💰",
    layout="wide"
)

# --- DESIGN SYSTEM ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; border-top: 4px solid #00d4ff; }
    .stAlert { border-radius: 12px; }
    .pro-header { color: #00d4ff; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES FINANCIÈRES ---
TICGN = 16.37  # €/MWh
CTA = 4.50     # Contribution Tarifaire d'Acheminement (estimée)
VAT = 1.20     # TVA 20%

# --- MOTEUR DE DONNÉES ---

@st.cache_data(ttl=3600)
def get_pro_market_data(ticker="TTF=F"):
    """Récupère 2 ans de données de clôture ajustées pour le TTF"""
    end = datetime.now()
    start = end - timedelta(days=730)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty: return pd.DataFrame()
        # Clean MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'][[ticker]].copy()
            df.columns = ['Price']
        else:
            df = data[['Close']].copy()
            df.columns = ['Price']
        return df.dropna()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_pro_weather(lat=48.85, lon=2.35):
    """Récupère l'historique météo (Archive API)"""
    end = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_mean&timezone=Europe%2FParis"
    try:
        res = requests.get(url, timeout=10).json()
        df = pd.DataFrame(res['daily'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={"temperature_2m_mean": "Temp"}).set_index('time')
        return df
    except:
        return pd.DataFrame()

# --- MOTEUR QUANTITATIF ---

def calculate_var(returns, confidence_level=0.95):
    """Calcule la Value-at-Risk historique"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def monte_carlo_simulation(last_price, vol, days=30, simulations=100):
    """Simulation de trajectoires de prix (Geometric Brownian Motion)"""
    dt = 1/252
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price
    for t in range(1, days):
        rand = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp((0 - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rand)
    return price_paths

# --- INTERFACE ---

st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=80)
st.sidebar.title("Volt-Alpha Pro v4.0")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Titulaire du Master 2 Finance et Banque de la TSM*")
st.sidebar.divider()

# Inputs de Trading
st.sidebar.subheader("⚙️ Paramètres de Trading")
conf_level = st.sidebar.select_slider("Niveau de Confiance VaR", options=[0.90, 0.95, 0.99], value=0.95)
horizon = st.sidebar.slider("Horizon Simulation (Jours)", 7, 90, 30)

st.sidebar.subheader("🔌 Sources Institutionnelles")
st.sidebar.markdown("""
- [ICE TTF Gas Futures](https://www.theice.com/products/27996665/Dutch-TTF-Gas-Futures)
- [Open-Meteo ERA5 Data](https://open-meteo.com/)
- [Commission de Régulation de l'Énergie](https://www.cre.fr/)
""")

# --- LOGIQUE PRINCIPALE ---

with st.spinner("Analyse des vecteurs de rentabilité..."):
    prices = get_pro_market_data()
    weather = get_pro_weather()

    if not prices.empty and not weather.empty:
        # Alignement des données
        df = pd.merge(prices, weather, left_index=True, right_index=True, how='inner')
        df['Returns'] = df['Price'].pct_change()
        df['DJU'] = np.maximum(0, 18 - df['Temp'])
        df = df.dropna()

        # Stats Avancées
        current_price = df['Price'].iloc[-1]
        volatility = df['Returns'].std() * np.sqrt(252)
        var_val = calculate_var(df['Returns'].dropna(), conf_level)
        correlation = df['Price'].corr(df['DJU'])

        # --- DASHBOARD ---
        st.markdown(f"<h1 class='pro-header'>Terminal d'Arbitrage Énergie : France</h1>", unsafe_allow_html=True)
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Prix Spot TTF", f"{current_price:.2f} €", f"{df['Returns'].iloc[-1]*100:.2f}%")
        with k2:
            st.metric("Volatilité Ann.", f"{volatility*100:.1f}%", help="Écart-type des rendements log-normaux")
        with k3:
            st.metric(f"VaR {int(conf_level*100)}% (1j)", f"{var_val*current_price:.2f} €", delta_color="inverse")
        with k4:
            st.metric("Corrélation Gaz/DJU", f"{correlation:.2f}", delta="Forte" if correlation > 0.6 else "Stable")

        # Tabs d'Analyse
        t1, t2, t3 = st.tabs(["📈 Graphique de Convergence", "🎲 Simulation Monte Carlo", "🔬 Analyse des Risques"])

        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Price'], name="Prix Gaz (Axe G)", line=dict(color='#00d4ff', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['DJU'], name="Rigueur DJU (Axe D)", line=dict(color='#ff4b4b', width=1, dash='dot'), yaxis="y2"))
            
            fig.update_layout(
                template="plotly_dark", height=500,
                yaxis=dict(title="Prix TTF (€/MWh)", titlefont=dict(color="#00d4ff"), tickfont=dict(color="#00d4ff")),
                yaxis2=dict(title="DJU (°C base 18)", titlefont=dict(color="#ff4b4b"), tickfont=dict(color="#ff4b4b"), overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.subheader(f"Projection Monte Carlo ({horizon} jours)")
            sim_data = monte_carlo_simulation(current_price, volatility, days=horizon, simulations=50)
            
            fig_sim = go.Figure()
            for s in range(50):
                fig_sim.add_trace(go.Scatter(y=sim_data[:, s], line=dict(width=0.5, color='rgba(0, 212, 255, 0.2)'), showlegend=False))
            
            # Médiane de la simulation
            fig_sim.add_trace(go.Scatter(y=np.median(sim_data, axis=1), line=dict(color='#ff4b4b', width=3), name="Trajectoire Médiane"))
            
            fig_sim.update_layout(template="plotly_dark", height=450, title="Probabilités de trajectoires de prix")
            st.plotly_chart(fig_sim, use_container_width=True)
            st.info("Cette simulation projette l'évolution du prix en fonction de la volatilité actuelle, en supposant une absence de chocs climatiques imprévus.")

        with t3:
            c_a, c_b = st.columns(2)
            with c_a:
                st.subheader("Distribution des Rendements")
                fig_dist = px.histogram(df, x="Returns", nbins=50, template="plotly_dark", color_discrete_sequence=['#00d4ff'])
                fig_dist.add_vline(x=var_val, line_dash="dash", line_color="red", annotation_text=f"VaR {int(conf_level*100)}%")
                st.plotly_chart(fig_dist, use_container_width=True)
            with c_b:
                st.subheader("Corrélation Mobile (Rolling 60j)")
                roll_corr = df['Price'].rolling(60).corr(df['DJU'])
                st.plotly_chart(px.line(roll_corr, template="plotly_dark", color_discrete_sequence=['#ff4b4b']), use_container_width=True)

        # Rapport Final
        st.divider()
        st.subheader("📝 Note de Synthèse Quantitative")
        st.markdown(f"""
        L'analyse du marché français via le hub **PEG/TTF** et la station météo de **Paris** indique :
        - Un risque de baisse (VaR) de **{abs(var_val*current_price):.2f} €** par MWh sur une journée.
        - Une volatilité annuelle de **{volatility*100:.1f}%**, typique d'un marché en phase de restockage.
        - Le levier climatique (DJU) explique environ **{correlation**2*100:.1f}%** de la variance totale du prix sur 2 ans.
        
        *Stratégie conseillée :* Accumulation si le spread Intraday/Futures s'écarte de plus de 2 $\sigma$ de la moyenne mobile 30j.
        """)

    else:
        st.error("Données de marché indisponibles. Vérifiez la connexion aux serveurs Yahoo Finance.")

st.divider()
st.caption("Volt-Alpha Pro | Propriété de Florentin Gaugry - Expertise Finance de Marché.")
