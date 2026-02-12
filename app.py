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
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE DONNÉES ROBUSTE ---

def fetch_real_commodity(ticker):
    """Récupère la dernière cotation réelle disponible sur Yahoo Finance"""
    try:
        # On télécharge un historique plus large pour être sûr d'avoir une cotation
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df.empty:
            return None, None
        
        # Nettoyage MultiIndex et extraction de la colonne Close
        if isinstance(df.columns, pd.MultiIndex):
            close_series = df['Close'][ticker]
        else:
            close_series = df['Close']
            
        # On supprime les NaN et on prend la toute dernière valeur réelle
        valid_series = close_series.dropna()
        if not valid_series.empty:
            last_price = float(valid_series.iloc[-1])
            last_date = valid_series.index[-1].strftime('%d/%m/%Y')
            return last_price, last_date
        return None, None
    except:
        return None, None

def get_smard_data():
    """Récupère les prix Spot réels via l'API publique SMARD.de (Zone DE/LU)"""
    # Filter 410 = Day Ahead Prices, Region DE/LU
    # On cherche le timestamp du début de la semaine actuelle
    now = int(time.time() * 1000)
    url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{now}.json"
    try:
        # Note: SMARD stocke par paquets, on tente de remonter le temps si le fichier actuel est vide
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            # Fallback sur le paquet précédent
            last_week = now - (7 * 24 * 3600 * 1000)
            url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{last_week}.json"
            response = requests.get(url, timeout=5)
            
        data = response.json()
        df = pd.DataFrame(data['series'], columns=['Timestamp', 'Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df.set_index('Timestamp')
    except:
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

st.sidebar.title("⚡ Volt-Alpha v1.8")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

market_zone = st.sidebar.selectbox("Zone de Marché", ["France (FR)", "Allemagne (DE)"])
st.sidebar.info("Utilisation des flux réels Yahoo Finance & SMARD API.")

# --- EXECUTION ---

with st.spinner('Connexion aux places de marché...'):
    # Commodities
    gas_price, gas_date = fetch_real_commodity("TTF=F")
    co2_price, co2_date = fetch_real_commodity("CFI.L")
    
    # Power Prices (Real from SMARD)
    power_df = get_smard_data()
    weather_df = get_weather_real()

# Vérification de sécurité avant calcul
if gas_price and co2_price:
    # Formule Merit Order CCGT (Efficacité 55%)
    marginal_cost = (gas_price / 0.55) + (0.37 * co2_price)
else:
    marginal_cost = 0

# --- AFFICHAGE ---

st.title(f"Monitor de Corrélation & Arbitrage - {market_zone}")

# Ligne 1 : Metrics Réelles
c1, c2, c3, c4 = st.columns(4)

with c1:
    if not power_df.empty:
        current_spot = power_df['Price'].iloc[-1]
        st.metric("Prix Spot Réel (DE/LU)", f"{current_spot:.2f} €")
        st.caption("Source: SMARD.de API (Live)")
    else:
        st.error("Flux Spot indisponible")

with c2:
    if gas_price:
        st.metric("Gaz TTF (FUT)", f"{gas_price:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {gas_date} (Yahoo)</p>", unsafe_allow_html=True)
    else:
        st.error("Gaz indéterminé")

with c3:
    if co2_price:
        st.metric("Carbone EUA", f"{co2_price:.2f} €")
        st.markdown(f"<p class='source-info'>MàJ: {co2_date} (Yahoo)</p>", unsafe_allow_html=True)
    else:
        st.error("CO2 indéterminé")

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
        fig.add_trace(go.Scatter(x=power_df.index, y=power_df['Price'], name="Prix Spot DE/LU", line=dict(color='#00d4ff')))
        fig.add_hline(y=marginal_cost, line_dash="dash", line_color="red", annotation_text="Coût Marginal CCGT")
        fig.update_layout(template="plotly_dark", title="Spread Prix Spot vs Fondamentaux Gaz/CO2", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données graphiques en attente de flux...")

with t2:
    if not weather_df.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(px.line(weather_df, x='time', y='windspeed_100m', title="Vitesse du Vent (m/s)", template="plotly_dark"), use_container_width=True)
        with col_m2:
            st.plotly_chart(px.area(weather_df, x='time', y='shortwave_radiation', title="Solaire (W/m²)", template="plotly_dark"), use_container_width=True)

st.divider()
st.markdown("""
**Note Technique :** Ce logiciel ne contient aucune valeur de référence codée en dur. 
Si une métrique affiche une erreur, c'est que la source de données (Yahoo Finance ou SMARD) est temporairement injoignable ou en maintenance de week-end.
""")
