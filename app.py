import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Configuration de la page
st.set_page_config(page_title="Corrélation Énergie & Météo France", layout="wide")

# --- TITRE ET INTRODUCTION ---
st.title("⚡ Dashboard : Météo & Marchés de l'Énergie en France")
st.markdown("""
Cette application permet d'analyser l'impact des variations climatiques sur les prix de gros de l'électricité et du gaz en France. 
En tant qu'analyste, vous pouvez visualiser si une baisse de température corrèle effectivement avec une hausse des prix (effet chauffage).
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
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_mean": data["daily"]["temperature_2m_mean"]
        })
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération météo : {e}")
        return pd.DataFrame()

@st.cache_data
def get_energy_prices(start_date, end_date, weather_df):
    """
    Simule des prix de l'énergie corrélés à la température pour la démo.
    Dans un cas réel, on importerait des données de RTE (Eco2Mix) ou Powernext.
    """
    dates = weather_df["date"]
    # Base de prix : Électricité ~50-150€/MWh, Gaz ~30-80€/MWh
    # Logique : Prix = Base + (Inertie) - (0.5 * Température) + Bruit
    
    temp_factor = weather_df["temp_mean"].values
    
    # Simulation Électricité (Forte corrélation thermique en France)
    elec_base = 100
    elec_prices = elec_base - (2.5 * temp_factor) + np.random.normal(0, 10, len(dates))
    
    # Simulation Gaz
    gas_base = 40
    gas_prices = gas_base - (0.8 * temp_factor) + np.random.normal(0, 5, len(dates))
    
    df = pd.DataFrame({
        "date": dates,
        "Electricity_Price": np.maximum(elec_prices, 10), # Prix plancher
        "Gas_Price": np.maximum(gas_prices, 5)
    })
    return df

# --- SIDEBAR : FILTRES ---
st.sidebar.header("Configuration de l'Analyse")

selected_city = st.sidebar.selectbox("Sélectionner une ville (Référence Température)", list(CITIES.keys()))
date_range = st.sidebar.date_input(
    "Période d'analyse",
    value=(datetime.now() - timedelta(days=60), datetime.now() - timedelta(days=2)),
    max_value=datetime.now() - timedelta(days=2)
)

if len(date_range) == 2:
    start_dt, end_dt = date_range
    
    # Récupération des données
    with st.spinner("Récupération des données en cours..."):
        weather_df = fetch_weather_data(selected_city, start_dt, end_dt)
        if not weather_df.empty:
            energy_df = get_energy_prices(start_dt, end_dt, weather_df)
            full_df = pd.merge(weather_df, energy_df, on="date")
            
            # --- LAYOUT : KPI ---
            col1, col2, col3 = st.columns(3)
            avg_temp = full_df["temp_mean"].mean()
            avg_elec = full_df["Electricity_Price"].mean()
            correlation = full_df["temp_mean"].corr(full_df["Electricity_Price"])
            
            col1.metric("Température Moyenne", f"{avg_temp:.1f} °C")
            col2.metric("Prix Élec Moyen", f"{avg_elec:.2f} €/MWh")
            col3.metric("Corrélation Temp/Élec", f"{correlation:.2f}", 
                        help="Une valeur proche de -1 indique que les prix montent quand il fait froid.")

            # --- GRAPHIQUES ---
            
            st.subheader("📊 Évolution Temporelle")
            
            # Graphique à double axe (Température vs Électricité)
            fig_timeseries = go.Figure()
            
            fig_timeseries.add_trace(go.Scatter(
                x=full_df["date"], y=full_df["temp_mean"],
                name="Température (°C)", line=dict(color="#FF4B4B")
            ))
            
            fig_timeseries.add_trace(go.Scatter(
                x=full_df["date"], y=full_df["Electricity_Price"],
                name="Prix Électricité (€/MWh)", line=dict(color="#1F77B4"),
                yaxis="y2"
            ))
            
            fig_timeseries.update_layout(
                yaxis=dict(title="Température (°C)"),
                yaxis2=dict(title="Prix Électricité (€/MWh)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("📉 Nuage de points : Corrélation")
                fig_scatter = px.scatter(
                    full_df, x="temp_mean", y="Electricity_Price",
                    trendline="ols",
                    labels={"temp_mean": "Température (°C)", "Electricity_Price": "Prix Électricité (€/MWh)"},
                    title=f"Relation Température / Prix ({selected_city})",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_b:
                st.subheader("🧪 Analyse Statistique")
                st.write("Matrice de corrélation (Pearson) :")
                corr_matrix = full_df[["temp_mean", "Electricity_Price", "Gas_Price"]].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)
                
                st.info("""
                **Interprétation :** En France, le mix électrique est très sensible au chauffage électrique. 
                Une corrélation négative forte (ex: -0.8) confirme que la demande augmente significativement 
                lors des vagues de froid, tirant les prix vers le haut.
                """)

            # --- TABLEAU DE DONNÉES ---
            with st.expander("Voir les données brutes"):
                st.dataframe(full_df)
                
        else:
            st.warning("Impossible de récupérer les données météo pour cette période.")
else:
    st.info("Veuillez sélectionner une plage de dates valide dans la barre latérale.")

# Footer
st.markdown("---")
st.caption("Données météo : Open-Meteo API | Prix de l'énergie : Données simulées pour démonstration technique.")
