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
    .source-link { font-size: 0.8rem; color: #00d4ff; text-decoration: none; }
    .source-link:hover { text-decoration: underline; }
    .stAlert { background-color: #1e2130; border: 1px solid #ffaa00; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE RÉCUPÉRATION DE DONNÉES ---

def get_commodity_history():
    """Récupère l'historique réel du Gaz et du Carbone avec gestion robuste des erreurs"""
    # Tickers : TTF=F (Gaz), CFI.L (Carbon)
    tickers_list = ["TTF=F", "CFI.L"]
    try:
        # On télécharge les données
        raw_data = yf.download(tickers_list, period="1mo", interval="1d", progress=False)
        
        if raw_data.empty:
            raise ValueError("API Yahoo Finance a renvoyé un set vide.")

        # Extraction de la clôture (Close) et gestion du MultiIndex potentiel
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close'].copy()
        else:
            data = raw_data[['Close']].copy()

        # Renommage propre pour le traitement
        data = data.rename(columns={"TTF=F": "Gaz_TTF", "CFI.L": "Carbone_EUA"})
        
        # Nettoyage : suppression des lignes entièrement vides et remplissage des trous
        data = data.dropna(how='all').ffill().bfill()
        
        if "Gaz_TTF" not in data.columns or "Carbone_EUA" not in data.columns:
            raise ValueError("Colonnes manquantes dans les données reçues.")
            
        return data
    
    except Exception as e:
        # Fallback de secours : Données de marché réalistes pour l'analyse
        st.sidebar.warning(f"⚠️ Flux Finance instable. Utilisation de données de secours.")
        dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
        return pd.DataFrame({
            "Gaz_TTF": np.linspace(35.5, 37.2, 20) + np.random.normal(0, 0.4, 20),
            "Carbone_EUA": np.linspace(66.2, 64.8, 20) + np.random.normal(0, 0.3, 20)
        }, index=dates)

def get_weather_data(lat=48.8566, lon=2.3522):
    """Récupère les prévisions météo réelles via Open-Meteo"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_100m,shortwave_radiation&forecast_days=3"
    try:
        response = requests.get(url, timeout=5).json()
        df = pd.DataFrame(response['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except:
        return pd.DataFrame()

def get_power_data(gas_price, co2_price, api_key=None):
    """Génération du prix basé sur les fondamentaux (Modèle Merit Order TSM)"""
    dates = pd.date_range(end=datetime.now(), periods=48, freq='H')
    
    # Formule du coût marginal CCGT : (Gaz / Efficacité) + (Emission_Factor * CO2)
    # Efficacité standard 55%, Facteur d'émission 0.37t/MWh
    base_price = (gas_price / 0.55) + (0.37 * co2_price)
    
    # Composantes du prix : Fondamentaux + Cyclicité (Demande) + Bruit
    hour_effect = np.sin(np.linspace(0, 4*np.pi, 48)) * 15 
    noise = np.random.normal(0, 5, 48)
    spot = base_price + hour_effect + noise
    intraday = spot + np.random.normal(0, 3, 48)
    
    return pd.DataFrame({'Timestamp': dates, 'Spot_Price': spot, 'Intraday_Price': intraday}).set_index('Timestamp')

# --- LOGIQUE DE L'INTERFACE ---

st.sidebar.title("⚡ Volt-Alpha v1.5")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

with st.sidebar.expander("🔑 Paramètres & Statut"):
    entsoe_key = st.text_input("Clé API ENTSO-E", type="password")
    st.info("💡 Mode expert : Analyse des spreads Gaz/Power activée.")

market_zone = st.sidebar.selectbox("Zone de Marché", ["France (FR)", "Allemagne (DE)", "Espagne (ES)", "Italie (IT)"])

# --- TRAITEMENT DES DONNÉES ---
with st.spinner('Chargement des fondamentaux de marché...'):
    commos_hist = get_commodity_history()
    weather = get_weather_data()
    
    # On s'assure d'extraire des scalaires propres (float)
    try:
        current_gas = float(commos_hist['Gaz_TTF'].iloc[-1])
        current_co2 = float(commos_hist['Carbone_EUA'].iloc[-1])
    except:
        current_gas, current_co2 = 35.0, 65.0 # Valeurs par défaut en cas d'échec total

    power_prices = get_power_data(current_gas, current_co2, entsoe_key)
    marginal_cost_ccgt = (current_gas / 0.55) + (0.37 * current_co2)

# --- DASHBOARD ---

st.title(f"Monitor de Corrélation & Arbitrage - {market_zone}")

# Row 1: Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    delta_spot = power_prices['Spot_Price'].iloc[-1] - power_prices['Spot_Price'].iloc[-2]
    st.metric("Prix Power (Modèle)", f"{power_prices['Spot_Price'].iloc[-1]:.2f} €", f"{delta_spot:.2f}")

with col2:
    st.metric("Gaz TTF (Spot/Fut)", f"{current_gas:.2f} €")
    st.markdown("[🔗 Source: Yahoo Finance](https://finance.yahoo.com/quote/TTF=F/)", unsafe_allow_html=True)

with col3:
    st.metric("Carbone EUA", f"{current_co2:.2f} €")
    st.markdown("[🔗 Source: Yahoo Finance](https://finance.yahoo.com/quote/CFI.L/)", unsafe_allow_html=True)

with col4:
    st.metric("Break-even CCGT", f"{marginal_cost_ccgt:.2f} €")
    st.caption("Benchmark de rentabilité")

# Row 2: Graphiques
tab1, tab2, tab3 = st.tabs(["📊 Dynamique de Marché", "🌦️ Données Physiques", "🔍 Analyse de Corrélation"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=power_prices.index, y=power_prices['Spot_Price'], name="Spot", line=dict(color='#00d4ff', width=3)))
    fig.add_trace(go.Scatter(x=power_prices.index, y=power_prices['Intraday_Price'], name="Intraday", line=dict(color='#ffaa00', dash='dot')))
    fig.add_hline(y=marginal_cost_ccgt, line_dash="dash", line_color="red", annotation_text="Coût Marginal")
    fig.update_layout(title="Convergence du Prix vers les Coûts de Production", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not weather.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(weather, x='time', y='windspeed_100m', title="Vent (m/s)", template="plotly_dark", color_discrete_sequence=['#5af2a5']), use_container_width=True)
        with c2:
            st.plotly_chart(px.area(weather, x='time', y='shortwave_radiation', title="Solaire (W/m²)", template="plotly_dark", color_discrete_sequence=['#f9d71c']), use_container_width=True)

with tab3:
    st.subheader("Analyse de Co-intégration (30j)")
    st.plotly_chart(px.imshow(commos_hist.corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)
    st.line_chart(commos_hist)

st.divider()
st.markdown("**Stratégie Volt-Alpha :** Si le prix Spot diverge du coût marginal CCGT sans changement de météo, une opportunité d'arbitrage est détectée.")
