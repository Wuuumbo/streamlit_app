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
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE DONNÉES VERSION 2.1 (PRO-DATA) ---

def fetch_commodity_v2(tickers):
    """Extraction résiliente pour les commodities financières"""
    if isinstance(tickers, str): tickers = [tickers]
    
    for ticker in tickers:
        try:
            # Récupération de l'historique récent
            data = yf.Ticker(ticker).history(period="1mo")
            if data.empty: continue
            
            # Extraction de la dernière valeur non nulle
            # On utilise 'Close' de manière explicite
            valid_data = data['Close'].dropna()
            if not valid_data.empty:
                val = float(valid_data.iloc[-1])
                date_str = valid_data.index[-1].strftime('%d/%m/%Y')
                return val, date_str, ticker
        except Exception:
            continue
    return None, None, None

def get_smard_data_v2():
    """Récupère les prix Spot via l'index de fichiers de SMARD (Plus robuste)"""
    # 410 = Day Ahead, DE = Region
    index_url = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    try:
        # 1. On récupère d'abord l'index des timestamps disponibles
        index_res = requests.get(index_url, timeout=5)
        if index_res.status_code != 200: return pd.DataFrame()
        
        timestamps = index_res.json().get('timestamps', [])
        if not timestamps: return pd.DataFrame()
        
        # 2. On prend le timestamp le plus récent
        last_ts = timestamps[-1]
        data_url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{last_ts}.json"
        
        data_res = requests.get(data_url, timeout=5)
        if data_res.status_code == 200:
            json_data = data_res.json()
            if 'series' in json_data:
                df = pd.DataFrame(json_data['series'], columns=['Timestamp', 'Price'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                return df.set_index('Timestamp')
    except Exception:
        pass
    return pd.DataFrame()

def get_weather_v2():
    """Prévisions météo réelles"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=48.8566&longitude=2.3522&hourly=windspeed_100m,shortwave_radiation&forecast_days=1"
    try:
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except: return pd.DataFrame()

# --- LOGIQUE MÉTIER ---

st.sidebar.title("⚡ Volt-Alpha v2.1")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

market_zone = st.sidebar.selectbox("Zone de Marché", ["France (FR)", "Allemagne (DE)"])
st.sidebar.info("V2.1 : Indexation dynamique SMARD et Scan Carbone étendu activés.")

# --- EXECUTION ---

with st.spinner('Synchronisation des flux de marché réels...'):
    # Gaz TTF
    gas_val, gas_date, gas_t = fetch_commodity_v2(["TTF=F", "TG.F"])
    
    # Carbone EUA (Extension de la liste des tickers pour garantir un résultat)
    # EUA=F (ICE), CO2.L (WisdomTree), KEUA (ETF), KRBN (ETF Global)
    co2_val, co2_date, co2_t = fetch_commodity_v2(["EUA=F", "CO2.L", "KEUA", "CFI.L"])
    
    # Power Spot
    power_df = get_smard_data_v2()
    weather_df = get_weather_v2()

# Calcul du Coût Marginal (Spark Spread)
if gas_val and co2_val:
    marginal_cost = (gas_val / 0.55) + (0.37 * co2_val)
else:
    marginal_cost = None

# --- AFFICHAGE ---

st.title(f"Monitor de Corrélation & Arbitrage - {market_zone}")

# Barre de statut en Sidebar
st.sidebar.subheader("Statut des Flux")
def status_indicator(val, name):
    color = "#28a745" if val else "#dc3545"
    st.sidebar.markdown(f"<div class='status-box' style='background-color: {color}22; border-left: 4px solid {color};'>{name}: {'OK' if val else 'HS'}</div>", unsafe_allow_html=True)

status_indicator(gas_val, "Flux Gaz (TTF)")
status_indicator(co2_val, "Flux Carbone (EUA)")
status_indicator(not power_df.empty, "Flux Power (Spot)")

# Ligne 1 : Metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    if not power_df.empty:
        curr_p = power_df['Price'].iloc[-1]
        st.metric("Prix Spot Réel (DE/LU)", f"{curr_p:.2f} €")
        st.caption(f"Dernière cotation: {power_df.index[-1].strftime('%H:%M')}")
    else:
        st.error("Flux Spot indisponible")

with c2:
    if gas_val:
        st.metric("Gaz TTF (Live)", f"{gas_val:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {gas_date} ({gas_t})</p>", unsafe_allow_html=True)
    else:
        st.error("Donnée Gaz manquante")

with c3:
    if co2_val:
        st.metric("Carbone EUA (Live)", f"{co2_val:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {co2_date} ({co2_t})</p>", unsafe_allow_html=True)
    else:
        st.error("Donnée CO2 manquante")

with c4:
    if marginal_cost:
        st.metric("Break-even CCGT", f"{marginal_cost:.2f} €")
        st.caption("Modèle Merit Order")
    else:
        st.error("Calcul impossible")

# Ligne 2 : Graphiques
t1, t2 = st.tabs(["📊 Dynamique du Spread", "🌦️ Météo & Fondamentaux"])

with t1:
    if not power_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=power_df.index, y=power_df['Price'], name="Prix Spot DE/LU", line=dict(color='#00d4ff', width=2)))
        if marginal_cost:
            fig.add_hline(y=marginal_cost, line_dash="dash", line_color="red", annotation_text="Break-even CCGT")
        fig.update_layout(template="plotly_dark", title="Analyse de convergence Spot vs Coût Marginal", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Le graphique s'affichera dès que le flux SMARD sera synchronisé.")

with t2:
    if not weather_df.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(px.line(weather_df, x='time', y='windspeed_100m', title="Vent (m/s)", template="plotly_dark", color_discrete_sequence=['#5af2a5']), use_container_width=True)
        with col_m2:
            st.plotly_chart(px.area(weather_df, x='time', y='shortwave_radiation', title="Solaire (W/m²)", template="plotly_dark", color_discrete_sequence=['#f9d71c']), use_container_width=True)

st.divider()
st.markdown("""
**Expertise TSM :** La version 2.1 résout le problème des données manquantes en interrogeant directement l'index des fichiers SMARD. 
Pour le carbone, l'utilisation de `CO2.L` assure une cotation même quand les contrats Futures ICE sont difficiles d'accès sur Yahoo Finance.
""")
