import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha | Monitor d'Arbitrage Énergétique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    .source-info { font-size: 0.75rem; color: #888; margin-top: -10px; }
    .stAlert { background-color: #1e2130; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE DONNÉES ROBUSTE ---

def fetch_real_commodity(ticker_list):
    """Teste une liste de tickers et renvoie la première cotation réelle trouvée"""
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]
        
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="7d", interval="1d", progress=False)
            if df.empty:
                continue
            
            # Gestion MultiIndex et extraction
            if isinstance(df.columns, pd.MultiIndex):
                close_series = df['Close'][ticker]
            else:
                close_series = df['Close']
                
            valid_series = close_series.dropna()
            if not valid_series.empty:
                return float(valid_series.iloc[-1]), valid_series.index[-1].strftime('%d/%m/%Y'), ticker
        except:
            continue
    return None, None, None

def get_smard_data():
    """Récupère les prix Spot via SMARD en remontant le temps si nécessaire"""
    now = int(time.time() * 1000)
    # On teste les 4 derniers paquets hebdomadaires
    for i in range(4):
        timestamp = now - (i * 7 * 24 * 3600 * 1000)
        url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{timestamp}.json"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'series' in data and len(data['series']) > 0:
                    df = pd.DataFrame(data['series'], columns=['Timestamp', 'Price'])
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                    return df.set_index('Timestamp')
        except:
            continue
    return pd.DataFrame()

def get_weather_real(lat=48.8566, lon=2.3522):
    """Prévisions Open-Meteo"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=windspeed_100m,shortwave_radiation&forecast_days=1"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except: return pd.DataFrame()

# --- LOGIQUE MÉTIER ---

st.sidebar.title("⚡ Volt-Alpha v1.9")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

market_zone = st.sidebar.selectbox("Zone de Marché", ["France (FR)", "Allemagne (DE)"])
st.sidebar.info("Moteur de secours multi-flux activé pour garantir l'affichage des données réelles.")

# --- EXECUTION ---

with st.spinner('Extraction des données réelles (Multi-flux)...'):
    # Gaz TTF (Ticker principal stable)
    gas_price, gas_date, gas_ticker = fetch_real_commodity(["TTF=F"])
    
    # Carbone EUA (On teste plusieurs tickers car CFI.L est souvent vide)
    co2_price, co2_date, co2_ticker = fetch_real_commodity(["EUA=F", "KEUA.L", "CFI.L"])
    
    # Power Prices (Flux SMARD dynamique)
    power_df = get_smard_data()
    weather_df = get_weather_real()

# Calcul du coût marginal
if gas_price and co2_price:
    marginal_cost = (gas_price / 0.55) + (0.37 * co2_price)
else:
    marginal_cost = 0

# --- AFFICHAGE ---

st.title(f"Monitor de Corrélation & Arbitrage - {market_zone}")

# Ligne 1 : Metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    if not power_df.empty:
        current_spot = power_df['Price'].iloc[-1]
        st.metric("Prix Spot Réel (DE/LU)", f"{current_spot:.2f} €")
        st.caption(f"Dernier flux : {power_df.index[-1].strftime('%H:%M')}")
    else:
        st.error("Flux Spot SMARD indisponible")

with c2:
    if gas_price:
        st.metric("Gaz TTF (FUT)", f"{gas_price:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {gas_date} ({gas_ticker})</p>", unsafe_allow_html=True)
    else:
        st.error("Flux Gaz indisponible")

with c3:
    if co2_price:
        st.metric("Carbone EUA", f"{co2_price:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {co2_date} ({co2_ticker})</p>", unsafe_allow_html=True)
    else:
        st.error("Flux CO2 indisponible")

with c4:
    if marginal_cost > 0:
        st.metric("Break-even CCGT", f"{marginal_cost:.2f} €")
        st.caption("Modèle Merit Order")
    else:
        st.error("Calcul impossible")

# Ligne 2 : Graphiques
t1, t2 = st.tabs(["📈 Analyse de Convergence", "🌦️ Météo Fondamentale"])

with t1:
    if not power_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=power_df.index, y=power_df['Price'], name="Prix Spot", line=dict(color='#00d4ff', width=2)))
        if marginal_cost > 0:
            fig.add_hline(y=marginal_cost, line_dash="dash", line_color="red", annotation_text="Coût Marginal CCGT")
        fig.update_layout(template="plotly_dark", title="Spread Temps Réel vs Fondamentaux", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("En attente de synchronisation avec l'API SMARD...")

with t2:
    if not weather_df.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(px.line(weather_df, x='time', y='windspeed_100m', title="Vent (m/s)", template="plotly_dark"), use_container_width=True)
        with col_m2:
            st.plotly_chart(px.area(weather_df, x='time', y='shortwave_radiation', title="Solaire (W/m²)", template="plotly_dark"), use_container_width=True)

st.divider()
st.markdown("""
**Expertise TSM :** Ce dashboard utilise désormais un mécanisme de redondance. Si le ticker principal du Carbone (`CFI.L`) échoue, le système bascule sur les contrats Futures (`EUA=F`) ou l'ETF (`KEUA.L`) pour garantir la continuité du calcul du Spark Spread.
""")
