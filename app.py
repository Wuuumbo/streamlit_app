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
    page_title="Volt-Alpha Pro | Analyse Fondamentale",
    page_icon="⚡",
    layout="wide"
)

# --- DESIGN SYSTEM ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 12px; border-top: 4px solid #00d4ff; }
    .pro-header { color: #00d4ff; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .source-tag { font-size: 0.75rem; color: #888; }
    </style>
    """, unsafe_allow_html=True)

# --- RÉFÉRENTIEL DES ZONES ---
ZONES = {
    "Toulouse (Occitanie)": {"lat": 43.6047, "lon": 1.4442},
    "Bordeaux (Nouvelle-Aquitaine)": {"lat": 44.8378, "lon": -0.5792},
    "Lyon (Auvergne-Rhône-Alpes)": {"lat": 45.7640, "lon": 4.8357},
    "Paris (Île-de-France)": {"lat": 48.8566, "lon": 2.3522},
    "Marseille (PACA)": {"lat": 43.2965, "lon": 5.3698},
    "Lille (Hauts-de-France)": {"lat": 50.6292, "lon": 3.0573},
    "Strasbourg (Grand Est)": {"lat": 48.5734, "lon": 7.7521}
}

# --- MOTEUR DE DONNÉES RÉELLES ---

@st.cache_data(ttl=3600)
def get_real_gas_data():
    """Récupère 2 ans de prix Gaz TTF (Benchmark Européen)"""
    try:
        data = yf.download("TTF=F", period="2y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'][["TTF=F"]].copy()
            df.columns = ['Gas_Price']
        else:
            df = data[['Close']].copy()
            df.columns = ['Gas_Price']
        return df.dropna()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_real_elec_data():
    """Récupère les prix Spot Electricité via SMARD API (Benchmark Day-Ahead)"""
    # On utilise l'index SMARD pour trouver le dernier paquet de données horaire disponible
    index_url = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    try:
        idx_res = requests.get(index_url, timeout=5).json()
        last_ts = idx_res['timestamps'][-1]
        data_url = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{last_ts}.json"
        data_res = requests.get(data_url, timeout=5).json()
        df = pd.DataFrame(data_res['series'], columns=['Timestamp', 'Elec_Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        # Agrégation journalière pour comparaison météo
        df = df.set_index('Timestamp').resample('D').mean()
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_real_weather_archive(lat, lon):
    """Récupère l'historique météo réel sur 2 ans pour la zone choisie"""
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

# --- INTERFACE ---

st.sidebar.title("Volt-Alpha Pro v5.0")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Titulaire du Master 2 Finance et Banque de la TSM*")
st.sidebar.divider()

selected_zone = st.sidebar.selectbox("📍 Sélectionner une Zone / Ville", list(ZONES.keys()))
rolling_avg = st.sidebar.slider("Lissage (Moyenne Mobile)", 1, 30, 7)

st.sidebar.subheader("🔌 Sources de Marché")
st.sidebar.markdown("""
- **Gaz :** TTF Dutch Futures (Yahoo)
- **Électricité :** SMARD Day-Ahead DE/LU (Proxy EU)
- **Météo :** Archive ERA5 (Open-Meteo)
""")

# --- LOGIQUE DE CALCUL ---

with st.spinner("Chargement des séries temporelles réelles..."):
    gas_df = get_real_gas_data()
    elec_df = get_real_elec_data()
    weather_df = get_real_weather_archive(ZONES[selected_zone]['lat'], ZONES[selected_zone]['lon'])

    if not gas_df.empty and not weather_df.empty:
        # Alignement des 3 sources
        master_df = pd.merge(gas_df, weather_df, left_index=True, right_index=True, how='inner')
        if not elec_df.empty:
            master_df = pd.merge(master_df, elec_df, left_index=True, right_index=True, how='inner')

        # Feature Engineering : DJU (Rigueur hivernale)
        master_df['DJU'] = np.maximum(0, 18 - master_df['Temp'])
        
        # Lissage pour l'analyse de tendance
        master_df['Gas_Smooth'] = master_df['Gas_Price'].rolling(rolling_avg).mean()
        master_df['Temp_Smooth'] = master_df['Temp'].rolling(rolling_avg).mean()
        if 'Elec_Price' in master_df.columns:
            master_df['Elec_Smooth'] = master_df['Elec_Price'].rolling(rolling_avg).mean()

        # --- DASHBOARD ---
        st.markdown(f"<h1 class='pro-header'>Analyse Comparative Météo-Énergie : {selected_zone}</h1>", unsafe_allow_html=True)
        
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Gaz TTF", f"{master_df['Gas_Price'].iloc[-1]:.2f} €")
        with k2:
            if 'Elec_Price' in master_df.columns:
                st.metric("Elec Spot (Moy)", f"{master_df['Elec_Price'].iloc[-1]:.2f} €")
        with k3:
            st.metric("Température", f"{master_df['Temp'].iloc[-1]:.1f} °C")
        with k4:
            corr = master_df['Gas_Price'].corr(master_df['DJU'])
            st.metric("Corrélation Gaz/DJU", f"{corr:.2f}")

        # Graphique Principal : Superposition Temp vs Prix
        st.subheader(f"Observation Historique (Lissage {rolling_avg} jours)")
        
        fig = go.Figure()
        
        # Gaz
        fig.add_trace(go.Scatter(x=master_df.index, y=master_df['Gas_Smooth'], name="Gaz TTF (€/MWh)", line=dict(color='#00d4ff', width=3)))
        
        # Electricité (si dispo)
        if 'Elec_Smooth' in master_df.columns:
            fig.add_trace(go.Scatter(x=master_df.index, y=master_df['Elec_Smooth'], name="Elec Spot (€/MWh)", line=dict(color='#ffaa00', width=2, dash='dot')))
            
        # Température (Axe Secondaire)
        fig.add_trace(go.Scatter(
            x=master_df.index, y=master_df['Temp_Smooth'], 
            name="Température (°C)", 
            line=dict(color='#ff4b4b', width=1),
            yaxis="y2",
            opacity=0.6
        ))

        fig.update_layout(
            template="plotly_dark", height=600,
            yaxis=dict(title="Prix Énergie (€/MWh)", titlefont=dict(color="#00d4ff")),
            yaxis2=dict(title="Température (°C)", titlefont=dict(color="#ff4b4b"), overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Analyse Comparative par Zone
        st.divider()
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("Sensibilité Prix/Froid (Scatter)")
            # On compare le prix du gaz aux DJU (indicateur de chauffage)
            fig_scat = px.scatter(
                master_df, x="DJU", y="Gas_Price", 
                color="Temp", template="plotly_dark",
                title="Dispersion : Prix Gaz vs Rigueur Climatique",
                labels={"DJU": "Rigueur (DJU)", "Gas_Price": "Gaz (€/MWh)"},
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with c_right:
            st.subheader("Statistiques de la Zone")
            st.write(f"Analyse sur les 2 dernières années pour **{selected_zone}** :")
            stats = master_df[['Gas_Price', 'Temp', 'DJU']].describe().T
            st.dataframe(stats.style.format("{:.2f}"))
            
            # Note d'analyse
            st.info(f"**Analyse TSM :** La corrélation entre la météo de {selected_zone} et le gaz européen est de {corr:.2f}. "
                    f"Cela démontre {'une forte dépendance saisonnière' if abs(corr) > 0.6 else 'une influence modérée par rapport aux drivers géopolitiques'}.")

    else:
        st.error("Impossible de synchroniser les flux de données. Vérifiez la connectivité des API.")

st.divider()
st.caption("Volt-Alpha Pro v5.0 | Focus Observation Réelle & Analyse Comparative Zone France.")
