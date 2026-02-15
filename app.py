import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Tentative d'import de matplotlib pour le stylage Pandas
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration de la page pour une lisibilité optimale
st.set_page_config(
    page_title="Energy & Weather Analytics Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- THÈME ET STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #FDFDFD; }
    .stMetric { 
        background-color: #FFFFFF; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #EEE;
    }
    h1, h2, h3 { color: #1E293B; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .stMarkdown { color: #334155; font-size: 1.05rem; }
    </style>
    """, unsafe_allow_html=True)

# --- DONNÉES ET CONSTANTES ---
CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Lyon": {"lat": 45.7640, "lon": 4.8357},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442},
    "Lille": {"lat": 50.6292, "lon": 3.0573},
    "Bordeaux": {"lat": 44.8378, "lon": -0.5792},
    "Nantes": {"lat": 47.2184, "lon": -1.5536},
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521}
}

@st.cache_data(ttl=3600)
def fetch_city_weather(city_name, start_date, end_date):
    """Récupère les données météo pour une ville spécifique"""
    coords = CITIES[city_name]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_mean"],
        "timezone": "Europe/Berlin"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            f"temp_{city_name}": data["daily"]["temperature_2m_mean"]
        })
    except:
        return pd.DataFrame()

def simulate_energy_prices(combined_df, cities):
    """
    Simulation basée sur la moyenne pondérée des températures sélectionnées.
    """
    temp_cols = [f"temp_{c}" for c in cities]
    avg_temp = combined_df[temp_cols].mean(axis=1)
    
    elec_base = 75
    sensitivity = np.where(avg_temp < 15, (15 - avg_temp) * 5.5, 0)
    
    combined_df['weekday'] = combined_df['date'].dt.weekday
    weekend_effect = np.where(combined_df['weekday'] >= 5, -15, 0)
    
    elec_prices = elec_base + sensitivity + weekend_effect + np.random.normal(0, 8, len(combined_df))
    gas_prices = 35 + np.where(avg_temp < 10, (10 - avg_temp) * 1.5, 0) + np.random.normal(0, 3, len(combined_df))
    
    combined_df["Electricity_Price"] = np.maximum(elec_prices, 10)
    combined_df["Gas_Price"] = np.maximum(gas_prices, 5)
    combined_df["National_Temp_Avg"] = avg_temp
    return combined_df

# --- SIDEBAR : FILTRES ---
st.sidebar.title("📈 Configuration")

selected_cities = st.sidebar.multiselect(
    "Villes à inclure dans l'indice",
    options=list(CITIES.keys()),
    default=["Paris", "Lyon", "Lille"]
)

date_range = st.sidebar.date_input(
    "Période d'analyse",
    value=(datetime.now() - timedelta(days=60), datetime.now() - timedelta(days=2)),
    max_value=datetime.now() - timedelta(days=2)
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Sources de données")
st.sidebar.markdown("""
- [Open-Meteo](https://open-meteo.com/) (Météo Historique)
- [Réseau de Transport d'Électricité (RTE)](https://www.services-rte.com/)
- [EEX Group](https://www.eex.com/) (Prix Spot/Futures)
- [Powernext](https://www.powernext.com/) (Gaz)
""")

# --- MAIN APP ---
st.title("⚡ Corrélation Météo & Marchés de l'Énergie")

if not selected_cities:
    st.warning("Veuillez sélectionner au moins une ville dans la barre latérale.")
elif len(date_range) == 2:
    start_dt, end_dt = date_range
    
    with st.spinner("Fusion des flux de données en cours..."):
        full_df = None
        for city in selected_cities:
            city_data = fetch_city_weather(city, start_dt, end_dt)
            if not city_data.empty:
                if full_df is None:
                    full_df = city_data
                else:
                    full_df = pd.merge(full_df, city_data, on="date")
        
        if full_df is not None:
            df = simulate_energy_prices(full_df, selected_cities)
            
            # --- KPI SECTION ---
            c1, c2, c3, c4 = st.columns(4)
            corr = df["National_Temp_Avg"].corr(df["Electricity_Price"])
            vol = df["Electricity_Price"].std()
            
            c1.metric("Indice Temp. Moyen", f"{df['National_Temp_Avg'].mean():.1f} °C")
            c2.metric("Prix Élec Moyen", f"{df['Electricity_Price'].mean():.2f} €", "MWh")
            c3.metric("Coefficient de Corrélation", f"{0 if np.isnan(corr) else corr:.2f}", delta_color="inverse")
            c4.metric("Volatilité Prix (Std Dev)", f"{vol:.1f} €")

            # --- GRAPHIQUE PRINCIPAL ---
            st.subheader("🚠 Analyse Multi-villes vs Prix de Gros")
            
            fig = go.Figure()
            
            for city in selected_cities:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[f"temp_{city}"],
                    name=f"Temp. {city}",
                    line=dict(width=1, dash='dot'),
                    opacity=0.4
                ))
            
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["National_Temp_Avg"],
                name="INDICE TEMP. MOYEN",
                line=dict(color="#FF4B4B", width=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["Electricity_Price"],
                name="PRIX ÉLEC SPOT (€/MWh)",
                yaxis="y2",
                line=dict(color="#1E293B", width=3)
            ))
            
            # Correction de la syntaxe pour éviter le ValueError (Modern Plotly Schema)
            fig.update_layout(
                template="plotly_white",
                height=600,
                yaxis=dict(
                    title=dict(text="Température (°C)", font=dict(color="#FF4B4B")),
                    tickfont=dict(color="#FF4B4B"),
                    gridcolor="rgba(255, 75, 75, 0.1)"
                ),
                yaxis2=dict(
                    title=dict(text="Prix Électricité (€/MWh)", font=dict(color="#1E293B")),
                    tickfont=dict(color="#1E293B"),
                    overlaying="y",
                    side="right",
                    gridcolor="rgba(30, 41, 59, 0.1)"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                margin=dict(l=50, r=50, t=30, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- ANALYSE DE DÉTAIL ---
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.subheader("🔍 Analyse de Dispersion")
                # Désactivation de trendline si statsmodels n'est pas dispo pour éviter un autre plantage
                try:
                    fig_scatter = px.scatter(
                        df, x="National_Temp_Avg", y="Electricity_Price",
                        trendline="ols",
                        labels={"National_Temp_Avg": "Température Moyenne (°C)", "Electricity_Price": "Prix Élec (€/MWh)"},
                        title="Régression Prix vs Température",
                        color_discrete_sequence=["#1E293B"]
                    )
                except:
                    fig_scatter = px.scatter(
                        df, x="National_Temp_Avg", y="Electricity_Price",
                        labels={"National_Temp_Avg": "Température Moyenne (°C)", "Electricity_Price": "Prix Élec (€/MWh)"},
                        title="Relation Prix vs Température (Sans Régression)",
                        color_discrete_sequence=["#1E293B"]
                    )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_b:
                st.subheader("📋 Matrice de Corrélation")
                cols_for_corr = [f"temp_{c}" for c in selected_cities] + ["Electricity_Price", "Gas_Price"]
                corr_matrix = df[cols_for_corr].corr()
                
                if HAS_MATPLOTLIB:
                    try:
                        st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn_r', axis=None).format("{:.2f}"), use_container_width=True)
                    except:
                        st.dataframe(corr_matrix.round(2), use_container_width=True)
                else:
                    st.dataframe(corr_matrix.round(2), use_container_width=True)

            # --- SOURCES ET NOTES ---
            with st.expander("📝 Notes méthodologiques et sources"):
                st.markdown("""
                **Méthodologie :**
                L'indice de température nationale est calculé par la moyenne simple des villes sélectionnées. 
                
                **Sources :**
                - **Météo** : [Open-Meteo API](https://open-meteo.com/).
                - **Prix** : Simulations basées sur les modèles de thermo-sensibilité (Power France). 
                """)
                
        else:
            st.error("Échec de la récupération des données. Veuillez vérifier la sélection des villes.")
else:
    st.info("Veuillez sélectionner une plage de dates complète.")

st.markdown("---")
st.caption(f"© {datetime.now().year} Energy Analytics Dashboard | Florentin Gaugry - Spécialiste Banque & Finance.")
