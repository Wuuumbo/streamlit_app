import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
# Nécessaire pour les dégradés Pandas (background_gradient)
import matplotlib.pyplot as plt 

# Configuration de la page
st.set_page_config(page_title="Energy & Weather Analytics", layout="wide", initial_sidebar_state="expanded")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_content_safe=True)

# --- TITRE ET INTRODUCTION ---
st.title("⚡ Energy & Weather Correlation Analytics")
st.markdown("""
Cet outil d'aide à la décision corrèle les variables climatiques avec les prix spots de l'énergie en France. 
*Objectif : Modéliser la thermo-sensibilité du marché français pour optimiser les stratégies de hedging.*
""")

# --- DONNÉES ET CONSTANTES ---
CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Lyon": {"lat": 45.7640, "lon": 4.8357},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442},
    "Lille": {"lat": 50.6292, "lon": 3.0573}
}

@st.cache_data(ttl=3600)
def fetch_weather_data(city_name, start_date, end_date):
    """Récupère les données météo historiques via Open-Meteo API"""
    coords = CITIES[city_name]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"],
        "timezone": "Europe/Berlin"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_mean": data["daily"]["temperature_2m_mean"],
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"]
        })
        return df
    except Exception as e:
        st.error(f"Erreur API Météo : {e}")
        return pd.DataFrame()

@st.cache_data
def get_energy_prices(weather_df):
    """
    Simulation de prix basée sur la thermo-sensibilité réelle du mix français.
    Le prix spot augmente exponentiellement quand la température descend sous 15°C.
    """
    dates = weather_df["date"]
    temp = weather_df["temp_mean"].values
    
    # Modèle de prix : Base + Effet Chauffage + Volatilité
    # En France, 1°C en moins = ~2400 MW de demande en plus.
    elec_base = 80 
    thermal_sensitivity = np.where(temp < 15, (15 - temp) * 4.5, 0)
    elec_prices = elec_base + thermal_sensitivity + np.random.normal(0, 12, len(dates))
    
    # Gaz (moins thermo-sensible sur le spot car stockage tampon)
    gas_base = 35
    gas_prices = gas_base + np.where(temp < 10, (10 - temp) * 1.2, 0) + np.random.normal(0, 4, len(dates))
    
    return pd.DataFrame({
        "date": dates,
        "Electricity_Price": np.maximum(elec_prices, 5),
        "Gas_Price": np.maximum(gas_prices, 2)
    })

# --- SIDEBAR : FILTRES ---
st.sidebar.header("🕹️ Paramètres d'Analyse")

selected_city = st.sidebar.selectbox("Ville de référence", list(CITIES.keys()))
date_range = st.sidebar.date_input(
    "Période",
    value=(datetime.now() - timedelta(days=90), datetime.now() - timedelta(days=2)),
    max_value=datetime.now() - timedelta(days=2)
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Stress Test")
temp_shock = st.sidebar.slider("Simuler un choc thermique (°C)", -15, 0, 0)

if len(date_range) == 2:
    start_dt, end_dt = date_range
    
    with st.spinner("Analyse des flux de données..."):
        weather_df = fetch_weather_data(selected_city, start_dt, end_dt)
        if not weather_df.empty:
            energy_df = get_energy_prices(weather_df)
            df = pd.merge(weather_df, energy_df, on="date")
            
            # Application du choc thermique pour la simulation
            df["temp_sim"] = df["temp_mean"] + temp_shock
            df["elec_sim"] = df["Electricity_Price"] + (abs(temp_shock) * 5.2 if temp_shock < 0 else 0)

            # --- KPI ---
            c1, c2, c3, c4 = st.columns(4)
            correlation = df["temp_mean"].corr(df["Electricity_Price"])
            volatility = df["Electricity_Price"].std() / df["Electricity_Price"].mean() * 100
            
            c1.metric("Temp. Moyenne", f"{df['temp_mean'].mean():.1f} °C")
            c2.metric("Prix Élec Spot", f"{df['Electricity_Price'].mean():.2f} €", "MWh")
            c3.metric("Corrélation", f"{correlation:.2f}", delta_color="inverse")
            c4.metric("Volatilité", f"{volatility:.1f}%")

            # --- VISUALISATION PRINCIPALE ---
            st.subheader("🚠 Corrélation Temporelle : Température vs Prix")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["temp_mean"], name="Température (°C)", line=dict(color="#EF553B", width=3)))
            fig.add_trace(go.Scatter(x=df["date"], y=df["Electricity_Price"], name="Prix Élec (€/MWh)", yaxis="y2", line=dict(color="#636EFA", width=2, dash='dot')))
            
            if temp_shock != 0:
                fig.add_trace(go.Scatter(x=df["date"], y=df["elec_sim"], name="Prix Simulé (Choc)", yaxis="y2", line=dict(color="#FFA15A", width=2)))

            fig.update_layout(
                yaxis=dict(title="Température (°C)", gridcolor='rgba(0,0,0,0.1)'),
                yaxis2=dict(title="Prix Électricité (€/MWh)", overlaying="y", side="right"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- ANALYSE DE RÉGRESSION & STATS ---
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("📉 Analyse de Régression (Sensibilité)")
                fig_reg = px.get_trendline_results(px.scatter(df, x="temp_mean", y="Electricity_Price", trendline="ols"))
                
                fig_scatter = px.scatter(
                    df, x="temp_mean", y="Electricity_Price", 
                    color="Electricity_Price", 
                    size=df["Electricity_Price"].abs(),
                    labels={"temp_mean": "Température (°C)", "Electricity_Price": "Prix (€/MWh)"},
                    template="plotly_white",
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_right:
                st.subheader("📋 Matrice Risque")
                corr_matrix = df[["temp_mean", "Electricity_Price", "Gas_Price"]].corr()
                try:
                    st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn_r', axis=None).format("{:.2f}"), use_container_width=True)
                except:
                    st.dataframe(corr_matrix.round(2), use_container_width=True)
                
                st.markdown(f"""
                **Note d'analyse :**
                - Coefficient Beta Temp/Elec : **{correlation:.2f}**
                - Un choc de **{temp_shock}°C** porterait le prix moyen théorique à **{df['elec_sim'].mean():.2f} €**.
                - Le gaz présente une corrélation de **{df['temp_mean'].corr(df['Gas_Price']):.2f}**.
                """)

            with st.expander("📂 Exportation des données brutes (Audit)"):
                st.download_button("Télécharger CSV", df.to_csv(index=False), "energy_data_audit.csv", "text/csv")
                st.dataframe(df.style.highlight_max(axis=0, color='#ffebcc'))
                
        else:
            st.error("Données indisponibles.")
else:
    st.info("Sélectionnez une période pour lancer l'analyse.")

st.markdown("---")
st.caption(f"Propriété Intellectuelle : Florentin Gaugry - Analyste Energie/Finance | {datetime.now().year}")
