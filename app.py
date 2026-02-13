import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha | Arbitrage Offre-Demande France",
    page_icon="⚖️",
    layout="wide"
)

# --- DESIGN SYSTEM ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; border-top: 4px solid #00d4ff; }
    .pro-header { color: #00d4ff; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .signal-box { padding: 15px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- RÉFÉRENTIEL DES HUBS DE PRODUCTION & CONSOMMATION ---
ZONES = {
    "Paris (Hub Consommation)": {"lat": 48.8566, "lon": 2.3522, "type": "Consommation"},
    "Brest (Hub Éolien)": {"lat": 48.3904, "lon": -4.4861, "type": "Production Éolienne"},
    "Marseille (Hub Solaire)": {"lat": 43.2965, "lon": 5.3698, "type": "Production Solaire"},
    "Lyon (Hub Industriel)": {"lat": 45.7640, "lon": 4.8357, "type": "Consommation"},
    "Lille (Zone Grand Froid)": {"lat": 50.6292, "lon": 3.0573, "type": "Consommation"}
}

# --- MOTEUR DE DONNÉES ---

@st.cache_data(ttl=3600)
def get_market_data():
    """Récupère les prix Gaz TTF et Elec Spot France"""
    data_dict = {}
    # Gaz TTF (Driver du coût marginal)
    try:
        gas = yf.download("TTF=F", period="1mo", interval="1d", progress=False)
        if not gas.empty:
            # Sécurisation du formatage : on extrait la valeur scalaire
            val = gas['Close'].dropna().iloc[-1]
            data_dict['Gas'] = float(val)
        else:
            data_dict['Gas'] = 35.0
    except: 
        data_dict['Gas'] = 35.0

    # Elec Spot France (SMARD API)
    index_url = "https://www.smard.de/app/chart_data/410/FR/index_hour.json"
    try:
        idx_res = requests.get(index_url, timeout=5).json()
        last_ts = idx_res['timestamps'][-1]
        url = f"https://www.smard.de/app/chart_data/410/FR/410_FR_hour_{last_ts}.json"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['series'], columns=['Timestamp', 'Elec_Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        data_dict['Elec_DF'] = df.set_index('Timestamp')
        data_dict['Elec_Last'] = float(df['Elec_Price'].iloc[-1])
    except:
        data_dict['Elec_DF'] = pd.DataFrame()
        data_dict['Elec_Last'] = 60.0
    
    return data_dict

@st.cache_data(ttl=3600)
def get_weather_drivers(lat, lon):
    """Récupère les variables météo critiques : Temp, Vent, Solaire"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_100m,shortwave_radiation&forecast_days=3"
    try:
        res = requests.get(url, timeout=10).json()
        df = pd.DataFrame(res['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index('time')
    except:
        return pd.DataFrame()

# --- INTERFACE ---

st.sidebar.title("Volt-Alpha : Arbitrage EnR")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Titulaire du Master 2 Finance et Banque de la TSM*")
st.sidebar.divider()

selected_zone = st.sidebar.selectbox("📍 Analyser une Zone Stratégique", list(ZONES.keys()))
zone_info = ZONES[selected_zone]

st.sidebar.info(f"Type de Zone : **{zone_info['type']}**")
st.sidebar.divider()

# --- LOGIQUE QUANTITATIVE ---

with st.spinner("Analyse des flux physiques en cours..."):
    market = get_market_data()
    weather = get_weather_drivers(zone_info['lat'], zone_info['lon'])

    if not weather.empty:
        # 1. Calcul de l'Indicateur de Tension (Stress Index)
        temp_last = weather['temperature_2m'].iloc[0]
        stress_thermal = max(0, (18 - temp_last) / 20) 
        
        wind_last = weather['windspeed_100m'].iloc[0]
        solar_last = weather['shortwave_radiation'].iloc[0]
        
        enr_potential = (min(wind_last, 60) / 60) * 0.7 + (min(solar_last, 800) / 800) * 0.3
        scarcity_score = stress_thermal - enr_potential
        
        # --- DASHBOARD ---
        st.markdown(f"<h1 class='pro-header'>Analyse de Tension du Réseau : {selected_zone}</h1>", unsafe_allow_html=True)
        
        # Signal d'Arbitrage
        if scarcity_score > 0.3:
            st.markdown("<div class='signal-box' style='background-color: #ff4b4b; color: white;'>🔴 SIGNAL : TENSION ÉLEVÉE (Pénurie EnR / Froid) - LONG POSITIONS</div>", unsafe_allow_html=True)
        elif scarcity_score < -0.3:
            st.markdown("<div class='signal-box' style='background-color: #28a745; color: white;'>🟢 SIGNAL : SURPRODUCTION EnR - SHORT POSITIONS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='signal-box' style='background-color: #ffaa00; color: white;'>🟡 SIGNAL : MARCHÉ NEUTRE (Équilibre)</div>", unsafe_allow_html=True)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Prix Élec Spot", f"{market['Elec_Last']:.2f} €/MWh")
        with c2:
            # Sécurisation de l'affichage du prix Gaz
            gas_price = market.get('Gas', 35.0)
            st.metric("Prix Gaz (Driver)", f"{gas_price:.2f} €")
        with c3:
            st.metric("Vitesse Vent (100m)", f"{wind_last:.1f} km/h")
        with c4:
            st.metric("Indice de Rareté", f"{scarcity_score:.2f}", help="Différence entre la demande thermique et l'offre renouvelable")

        # Graphiques
        t1, t2 = st.tabs(["📊 Analyse des Drivers Physiques", "📈 Corrélation Prix/Météo"])

        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weather.index, y=weather['windspeed_100m'], name="Vent (km/h)", line=dict(color='#00d4ff')))
            fig.add_trace(go.Scatter(x=weather.index, y=weather['shortwave_radiation'], name="Solaire (W/m²)", line=dict(color='#f9d71c'), yaxis="y2"))
            
            fig.update_layout(
                template="plotly_dark", height=500,
                title="Drivers de l'Offre Renouvelable (Prévisions 3 jours)",
                yaxis=dict(title="Vitesse Vent"),
                yaxis2=dict(title="Rayonnement Solaire", overlaying="y", side="right")
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            if not market['Elec_DF'].empty:
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=market['Elec_DF'].index, y=market['Elec_DF']['Elec_Price'], name="Prix Spot FR", line=dict(color='#ffaa00')))
                fig_corr.update_layout(template="plotly_dark", title="Dynamique des Prix Spot France (24-48h)", height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()
        st.subheader("📝 Note de l'Analyste")
        st.markdown(f"""
        En tant que **Titulaire du Master 2 Finance et Banque de la Toulouse School of Management**, j'observe que le driver principal de rentabilité pour **{selected_zone}** est actuellement **{ 'le vent' if wind_last > 30 else 'la température' if temp_last < 10 else 'le gaz' }**. 
        Pour un arbitrage lucratif, surveillez l'écart entre le prix Spot et le coût marginal du gaz quand l'Indice de Rareté dépasse 0.4.
        """)

st.divider()
st.caption("Volt-Alpha Pro v5.3 | Modélisation des flux physiques pour l'arbitrage énergétique.")
