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
    page_title="Volt-Alpha | Monitor d'Arbitrage Énergétique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE RÉCUPÉRATION DE DONNÉES ---

def get_commodity_data():
    """Récupère les prix du Gaz (TTF) et du Carbone (EUA) via Yahoo Finance"""
    # TTF Gas Futures (ICE) & EUA Carbon (ICE) via proxies YF
    tickers = {
        "Gaz_TTF": "TTF=F", 
        "Carbone_EUA": "CFI.L" # Proxy pour le carbone
    }
    data = yf.download(list(tickers.values()), period="1mo", interval="1d")['Close']
    data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    return data

def get_weather_data(lat=48.8566, lon=2.3522):
    """Récupère les prévisions météo (Vent, Solaire, Temp) via Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_100m,shortwave_radiation&forecast_days=3"
    try:
        response = requests.get(url).json()
        df = pd.DataFrame(response['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except:
        return pd.DataFrame()

def get_mock_entsoe_data():
    """Simule les données ENTSO-E (Prix Spot & Intraday) pour la démonstration"""
    dates = pd.date_range(end=datetime.now(), periods=48, freq='H')
    spot = 60 + np.random.normal(0, 15, 48) + np.sin(np.linspace(0, 4*np.pi, 48)) * 20
    intraday = spot + np.random.normal(0, 5, 48)
    return pd.DataFrame({'Timestamp': dates, 'Spot_Price': spot, 'Intraday_Price': intraday}).set_index('Timestamp')

# --- LOGIQUE DE CALCUL DU MERIT ORDER ---

def calculate_marginal_cost(gas_price, carbon_price, efficiency=0.55):
    """
    Calcule le coût marginal d'une centrale CCGT (Cycle Combiné Gaz)
    Formule : (Prix Gaz / Efficacité) + (Facteur Emission * Prix Carbone)
    """
    emission_factor = 0.37 # tCO2 / MWh thermique pour le gaz
    cost = (gas_price / efficiency) + (emission_factor * carbon_price)
    return cost

# --- INTERFACE UTILISATEUR (UI) ---

st.sidebar.title("⚡ Volt-Alpha v1.0")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

market_zone = st.sidebar.selectbox("Zone de Marché", ["France (FR)", "Allemagne (DE)", "Espagne (ES)", "Italie (IT)"])
refresh_rate = st.sidebar.slider("Fréquence de rafraîchissement (min)", 1, 60, 15)

# --- CHARGEMENT DES DONNÉES ---
with st.spinner('Agrégation des Market Drivers en cours...'):
    commos = get_commodity_data()
    weather = get_weather_data()
    power_prices = get_mock_entsoe_data()
    
    current_gas = commos['Gaz_TTF'].iloc[-1]
    current_co2 = commos['Carbone_EUA'].iloc[-1]
    marginal_cost_ccgt = calculate_marginal_cost(current_gas, current_co2)

# --- DASHBOARD PRINCIPAL ---

st.title(f"Monitor de Corrélation & Arbitrage - Marché {market_zone}")
st.info("Visualisation des fondamentaux physiques et financiers pour l'arbitrage Intraday.")

# Row 1: Key Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Prix Spot Actuel", f"{power_prices['Spot_Price'].iloc[-1]:.2f} €", f"{power_prices['Spot_Price'].iloc[-1] - power_prices['Spot_Price'].iloc[-2]:.2f}")
with col2:
    st.metric("Gaz TTF (MWh)", f"{current_gas:.2f} €", "-1.2%")
with col3:
    st.metric("Carbone EUA (t)", f"{current_co2:.2f} €", "+0.5%")
with col4:
    st.metric("Coût Marginal CCGT", f"{marginal_cost_ccgt:.2f} €", help="Prix d'équilibre théorique pour une centrale gaz")

# Row 2: Charts
tab1, tab2, tab3 = st.tabs(["📈 Prix & Spreads", "☁️ Facteurs Physiques (Météo)", "🧪 Analyse de Corrélation"])

with tab1:
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(x=power_prices.index, y=power_prices['Spot_Price'], name="Spot FR", line=dict(color='#00d4ff', width=3)))
    fig_prices.add_trace(go.Scatter(x=power_prices.index, y=power_prices['Intraday_Price'], name="Intraday FR", line=dict(color='#ffaa00', dash='dot')))
    fig_prices.add_hline(y=marginal_cost_ccgt, line_dash="dash", line_color="red", annotation_text="Break-even CCGT")
    fig_prices.update_layout(title="Dynamique des Prix Intraday vs Coûts Marginaux", template="plotly_dark", height=500)
    st.plotly_chart(fig_prices, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_wind = px.line(weather, x='time', y='windspeed_100m', title="Vitesse du Vent (100m) - Zone de Production", color_discrete_sequence=['#5af2a5'])
        fig_wind.update_layout(template="plotly_dark")
        st.plotly_chart(fig_wind, use_container_width=True)
    with col_b:
        fig_rad = px.area(weather, x='time', y='shortwave_radiation', title="Irradiance Solaire (W/m²)", color_discrete_sequence=['#f9d71c'])
        fig_rad.update_layout(template="plotly_dark")
        st.plotly_chart(fig_rad, use_container_width=True)

with tab3:
    st.subheader("Matrice de Sensibilité du Merit Order")
    # Simulation d'une matrice de corrélation
    corr_data = pd.DataFrame({
        'Prix_Power': power_prices['Spot_Price'].values[-20:],
        'Gaz_TTF': np.random.normal(current_gas, 2, 20),
        'CO2_EUA': np.random.normal(current_co2, 1, 20),
        'Wind_Speed': weather['windspeed_100m'].iloc[:20].values
    })
    corr_matrix = corr_data.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Corrélation des Market Drivers (Last 20h)")
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.divider()
st.markdown(f"""
    **Volt-Alpha Strategy Note:** Le spread entre l'Intraday et le coût marginal CCGT est actuellement de **{power_prices['Intraday_Price'].iloc[-1] - marginal_cost_ccgt:.2f} €**. 
    Une augmentation de la vitesse du vent prévue pour demain pourrait compresser les marges sur le bloc 08-12h.
""")
