import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha | Analyse Gaz & Météo Historique",
    page_icon="🔥",
    layout="wide"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .report-text { font-size: 0.9rem; color: #ccc; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- COORDINATES DES RÉGIONS ---
REGIONS = {
    "Paris (Île-de-France)": {"lat": 48.8566, "lon": 2.3522},
    "Lille (Hauts-de-France)": {"lat": 50.6292, "lon": 3.0573},
    "Lyon (Auvergne-Rhône-Alpes)": {"lat": 45.7640, "lon": 4.8357},
    "Marseille (Provence-Alpes-Côte d'Azur)": {"lat": 43.2965, "lon": 5.3698},
    "Strasbourg (Grand Est)": {"lat": 48.5734, "lon": 7.7521},
    "Bordeaux (Nouvelle-Aquitaine)": {"lat": 44.8378, "lon": -0.5792},
    "Toulouse (Occitanie)": {"lat": 43.6047, "lon": 1.4442}
}

# --- FONCTIONS DE RÉCUPÉRATION ---

@st.cache_data
def get_historical_gas(start_date, end_date):
    """Récupère les prix du Gaz TTF sur la période sélectionnée"""
    try:
        # TTF=F est le contrat Future de référence pour le prix du gaz en Europe (donc France)
        gas = yf.download("TTF=F", start=start_date, end=end_date, progress=False)
        if gas.empty:
            return pd.DataFrame()
        # Nettoyage si MultiIndex
        if isinstance(gas.columns, pd.MultiIndex):
            gas = gas['Close']['TTF=F']
        else:
            gas = gas['Close']
        return gas.dropna()
    except:
        return pd.DataFrame()

@st.cache_data
def get_historical_weather(lat, lon, start_date, end_date):
    """Récupère les températures historiques via Open-Meteo Archive API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Paris"
    }
    try:
        response = requests.get(url, params=params, timeout=10).json()
        df = pd.DataFrame(response['daily'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={"temperature_2m_mean": "Temp_Moyenne"})
        return df.set_index('time')
    except:
        return pd.DataFrame()

# --- INTERFACE SIDEBAR ---

st.sidebar.title("🔥 Volt-Alpha Historique")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

selected_region = st.sidebar.selectbox("Choisir la région (France)", list(REGIONS.keys()))
year_range = st.sidebar.slider("Période d'analyse (Jours en arrière)", 30, 730, 365)

end_date = datetime.now() - timedelta(days=2) # Archive a souvent 2 jours de délai
start_date = end_date - timedelta(days=year_range)

# --- CHARGEMENT DES DONNÉES ---

with st.spinner(f"Analyse des corrélations pour {selected_region}..."):
    gas_data = get_historical_gas(start_date, end_date)
    weather_data = get_historical_weather(
        REGIONS[selected_region]['lat'], 
        REGIONS[selected_region]['lon'], 
        start_date, 
        end_date
    )

# --- PRÉPARATION DU DATAFRAME COMMUN ---

if not gas_data.empty and not weather_data.empty:
    # On aligne les données sur les dates communes
    combined = pd.merge(gas_data, weather_data, left_index=True, right_index=True, how='inner')
    combined.columns = ['Prix_Gaz', 'Temp_Moyenne']
    
    # Calcul de corrélation
    correlation = combined['Prix_Gaz'].corr(combined['Temp_Moyenne'])
    
    # --- AFFICHAGE ---

    st.title(f"Impact Climatique sur le Prix du Gaz - {selected_region}")
    st.info(f"Analyse sur les {year_range} derniers jours. Source : Yahoo Finance (TTF) & Open-Meteo Archive.")

    # Métriques
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Prix Gaz Moyen", f"{combined['Prix_Gaz'].mean():.2f} €/MWh")
    with m2:
        st.metric("Température Moyenne", f"{combined['Temp_Moyenne'].mean():.1f} °C")
    with m3:
        color = "inverse" if correlation < 0 else "normal"
        st.metric("Corrélation Gaz/Temp", f"{correlation:.2f}", delta="Forte" if abs(correlation) > 0.5 else "Modérée", delta_color=color)

    # Graphique Double Axe
    fig = go.Figure()

    # Trace Gaz
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined['Prix_Gaz'],
        name="Prix Gaz (TTF) - €/MWh",
        line=dict(color='#ff4b4b', width=3)
    ))

    # Trace Température (Axe Y secondaire)
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined['Temp_Moyenne'],
        name="Température (°C)",
        line=dict(color='#00d4ff', width=2, dash='dot'),
        yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"Evolution temporelle : Prix du Gaz vs Rigueur Climatique ({selected_region})",
        yaxis=dict(title="Prix Gaz (€/MWh)", titlefont=dict(color="#ff4b4b"), tickfont=dict(color="#ff4b4b")),
        yaxis2=dict(title="Température (°C)", titlefont=dict(color="#00d4ff"), tickfont=dict(color="#00d4ff"), anchor="x", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analyse de Saisonnalité
    st.divider()
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Analyse de Dispersion (Scatter)")
        fig_scatter = px.scatter(
            combined, x="Temp_Moyenne", y="Prix_Gaz", 
            trendline="ols",
            title="Relation Prix / Température",
            labels={"Temp_Moyenne": "Température (°C)", "Prix_Gaz": "Prix Gaz (€/MWh)"},
            template="plotly_dark",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.subheader("Note de Synthèse Stratégique")
        st.markdown(f"""
        <div class='report-text'>
        En tant que <b>Titulaire du Master 2 Finance et Banque</b>, l'analyse des données pour la région <b>{selected_region}</b> révèle les points suivants :
        
        1. <b>Sensibilité Thermique :</b> La corrélation de <b>{correlation:.2f}</b> indique une dépendance {'inverse forte (logique de chauffage)' if correlation < -0.5 else 'modérée'}.
        2. <b>Seuils Critiques :</b> Observez les pics de prix lorsque la température descend sous les 5°C. C'est le moment où les stockages sont sollicités.
        3. <b>Arbitrage Temps/Prix :</b> Le décalage (lag) entre une chute de température et la hausse du TTF permet d'anticiper des positions sur les contrats futures de court terme.
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("Impossible de récupérer les données pour cette période. Vérifiez la connexion aux APIs.")

st.divider()
st.caption("Volt-Alpha v2.2 - Logiciel de modélisation financière appliqué à l'énergie.")
