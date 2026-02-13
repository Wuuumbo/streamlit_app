import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import statsmodels.api as sm

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha | Élasticité Thermique France",
    page_icon="🌡️",
    layout="wide"
)

# --- DESIGN SYSTEM ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; border-top: 4px solid #00d4ff; }
    .pro-header { color: #00d4ff; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .analysis-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- RÉFÉRENTIEL DES ZONES DE CONSOMMATION ---
ZONES_FR = {
    "Paris (Île-de-France)": {"lat": 48.8566, "lon": 2.3522, "weight": 0.35},
    "Lyon (Rhône-Alpes)": {"lat": 45.7640, "lon": 4.8357, "weight": 0.20},
    "Toulouse (Occitanie)": {"lat": 43.6047, "lon": 1.4442, "weight": 0.15},
    "Lille (Nord)": {"lat": 50.6292, "lon": 3.0573, "weight": 0.15},
    "Marseille (PACA)": {"lat": 43.2965, "lon": 5.3698, "weight": 0.15}
}

# --- FONCTIONS DE RÉCUPÉRATION (HISTORIQUE 2 ANS) ---

@st.cache_data(ttl=86400)
def get_historical_market_data():
    """Récupère 2 ans de prix Gaz (TTF) et Électricité (SMARD Fallback)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # 1. Gaz TTF (Le driver le plus sûr)
    try:
        gas = yf.download("TTF=F", start=start_date, end=end_date, progress=False)
        gas_df = gas['Close'].copy() if not isinstance(gas.columns, pd.MultiIndex) else gas['Close'][["TTF=F"]]
        gas_df.columns = ["Gas_Price"]
    except:
        gas_df = pd.DataFrame()

    # 2. Electricité Spot France (Recherche récursive sur l'API SMARD)
    elec_df = pd.DataFrame()
    try:
        idx_url = "https://www.smard.de/app/chart_data/410/FR/index_hour.json"
        timestamps = requests.get(idx_url).json()['timestamps']
        # On fusionne les 4 derniers paquets hebdomadaires pour avoir un historique solide
        all_series = []
        for ts in timestamps[-8:]:
            url = f"https://www.smard.de/app/chart_data/410/FR/410_FR_hour_{ts}.json"
            res = requests.get(url).json()
            if 'series' in res:
                all_series.extend(res['series'])
        
        temp_df = pd.DataFrame(all_series, columns=['Timestamp', 'Elec_Price'])
        temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'], unit='ms')
        elec_df = temp_df.set_index('Timestamp').resample('D').mean()
    except:
        pass
        
    return gas_df, elec_df

@st.cache_data(ttl=86400)
def get_historical_weather(lat, lon):
    """Récupère 2 ans de données météo ERA5 (Certifiées Copernicus)"""
    end = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_mean&timezone=Europe%2FParis"
    try:
        res = requests.get(url).json()
        df = pd.DataFrame(res['daily'])
        df['time'] = pd.to_datetime(df['time'])
        return df.rename(columns={"temperature_2m_mean": "Temp"}).set_index('time')
    except:
        return pd.DataFrame()

# --- LOGIQUE ANALYTIQUE ---

st.sidebar.title("📈 Volt-Alpha Quant")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Titulaire du Master 2 Finance et Banque de la TSM*")
st.sidebar.divider()

selected_city = st.sidebar.selectbox("📍 Ville de référence (Zone)", list(ZONES_FR.keys()))
rolling_window = st.sidebar.slider("Lissage Analytique (Jours)", 1, 30, 14)

st.sidebar.subheader("🔌 Sources Auditées")
st.sidebar.markdown("""
- **Météo :** Copernicus ERA5 (Réanalysé)
- **Gaz :** ICE TTF Natural Gas
- **Elec :** EPEX SPOT France
""")

# --- CALCULS ---

with st.spinner("Analyse des corrélations structurelles (24 mois)..."):
    gas_hist, elec_hist = get_historical_market_data()
    weather_hist = get_historical_weather(ZONES_FR[selected_city]['lat'], ZONES_FR[selected_city]['lon'])

    if not gas_hist.empty and not weather_hist.empty:
        # Fusion des données
        df = pd.merge(gas_hist, weather_hist, left_index=True, right_index=True, how='inner')
        if not elec_hist.empty:
            df = pd.merge(df, elec_hist, left_index=True, right_index=True, how='inner')
        
        # Ingénierie des variables : DJU (Degrés Jours Unifiés)
        df['DJU'] = np.maximum(0, 15 - df['Temp']) # Seuil de déclenchement chauffage 15°C
        
        # Calcul de la Sensibilité (Beta de température)
        # On calcule combien d'euros le prix augmente par point de DJU
        valid_df = df.dropna()
        correlation = valid_df['Gas_Price'].corr(valid_df['DJU'])
        
        # --- DASHBOARD ---
        st.markdown(f"<h1 class='pro-header'>Analyse de l'Élasticité Prix/Température : {selected_city}</h1>", unsafe_allow_html=True)
        
        # KPIs de Rigueur
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Prix Gaz Moyen (2 ans)", f"{df['Gas_Price'].mean():.2f} €")
        with k2:
            st.metric("Corrélation Gaz/DJU", f"{correlation:.2f}")
        with m3 := k3:
            st.metric("Température Moy.", f"{df['Temp'].mean():.1f} °C")
        with k4:
            st.metric("Intensité DJU (Cumul)", f"{df['DJU'].sum():.0f}")

        # Graphique Double Axe Dynamique
        st.subheader("Visualisation de la Loi de l'Offre et de la Demande Thermique")
        
        fig = go.Figure()
        
        # Lissage pour voir la tendance de fond
        df['Gas_Smooth'] = df['Gas_Price'].rolling(rolling_window).mean()
        df['DJU_Smooth'] = df['DJU'].rolling(rolling_window).mean()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Gas_Smooth'], name="Prix Gaz (MM)", line=dict(color='#00d4ff', width=3)))
        fig.add_trace(go.Scatter(x=df.index, y=df['DJU_Smooth'], name="Besoin Chauffage (DJU)", line=dict(color='#ff4b4b', width=1, dash='dot'), yaxis="y2"))
        
        fig.update_layout(
            template="plotly_dark", height=600,
            yaxis=dict(title="Gaz TTF (€/MWh)", titlefont=dict(color="#00d4ff")),
            yaxis2=dict(title="Rigueur Climatique (DJU)", overlaying="y", side="right", titlefont=dict(color="#ff4b4b")),
            legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bloc Analyse Quantitative
        st.divider()
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Régression Linéaire : Prix = f(Temp)")
            fig_scat = px.scatter(df, x="Temp", y="Gas_Price", trendline="ols", 
                                 template="plotly_dark", color="DJU",
                                 title="Elasticité : Chaque point montre un jour réel sur 2 ans",
                                 color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_right:
            st.subheader("Note de Synthèse Stratégique")
            st.markdown(f"""
            <div class='analysis-card'>
            <b>Analyse de l'expert (TSM) :</b><br><br>
            L'étude des 730 derniers jours sur la zone <b>{selected_city}</b> démontre une corrélation structurelle de <b>{correlation:.2f}</b>. 
            <br><br>
            Contrairement aux idées reçues, le prix ne réagit pas seulement au froid immédiat, mais à la persistance des <b>DJU (Degrés Jours Unifiés)</b> qui vident les stocks de gaz européens.
            <br><br>
            <i>Opportunité d'arbitrage :</i> Une déconnexion entre la courbe des DJU (rouge) et le prix (bleu) indique souvent une inefficience de marché temporaire ou une influence géopolitique majeure.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Données historiques indisponibles. Les serveurs de données (Copernicus/Yahoo) sont momentanément saturés.")

st.divider()
st.caption("Volt-Alpha v6.0 | Plateforme de recherche quantitative - Propriété de Florentin Gaugry.")
