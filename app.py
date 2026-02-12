import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Volt-Alpha | Backtest Quant Gaz & Météo",
    page_icon="📉",
    layout="wide"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .source-link { font-size: 0.8rem; color: #00d4ff; text-decoration: none; }
    .source-link:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES ET MAPPAGE MÉTIER ---

CITY_MAP = {
    "Paris (Île-de-France)": {"lat": 48.8566, "lon": 2.3522, "zone": 2},
    "Toulouse (Occitanie)": {"lat": 43.6047, "lon": 1.4442, "zone": 1},
    "Lyon (Auvergne-Rhône-Alpes)": {"lat": 45.7640, "lon": 4.8357, "zone": 1},
    "Strasbourg (Grand Est)": {"lat": 48.5734, "lon": 7.7521, "zone": 1},
    "Brest (Bretagne)": {"lat": 48.3904, "lon": -4.4861, "zone": 6},
    "Biarritz (Nouvelle-Aquitaine)": {"lat": 43.4832, "lon": -1.5586, "zone": 1}
}

TICGN = 16.37  
ACHEMINEMENT_FIXE = 25.0  
VAT_COMMODITY = 1.20  

# --- FONCTIONS DE RÉCUPÉRATION DE DONNÉES ---

@st.cache_data(show_spinner=False)
def get_market_data(ticker, days):
    """Récupère les données de marché via yfinance"""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'][[ticker]].copy()
            df.columns = ['Close']
        else:
            df = data[['Close']].copy()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_weather_archive(lat, lon, days):
    """Récupère les températures historiques via Open-Meteo Archive"""
    end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Paris"
    }
    try:
        response = requests.get(url, params=params, timeout=10).json()
        df = pd.DataFrame(response['daily'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={"temperature_2m_mean": "temp_mean"})
        return df.set_index('time')
    except Exception:
        return pd.DataFrame()

# --- LOGIQUE DE CALCUL QUANTITATIF ---

def reconstruct_retail_price(wholesale_price, zone):
    """Transforme le prix de gros (TTF) en prix final estimé (€/MWh)"""
    zone_spread = (zone - 1) * 1.50 
    price_ht = wholesale_price + ACHEMINEMENT_FIXE + zone_spread + TICGN
    return price_ht * VAT_COMMODITY

# --- INTERFACE STREAMLIT ---

st.sidebar.title("📉 Volt-Alpha Quant")
st.sidebar.markdown(f"**Analyste :** Florentin Gaugry\n*Master 2 Finance & Banque*")
st.sidebar.divider()

# Sources de Données (Liens demandés)
st.sidebar.subheader("🔌 Sources des Données")
st.sidebar.markdown("""
- [📊 Marché Gaz (TTF Futures)](https://finance.yahoo.com/quote/TTF=F/)
- [🌡️ Météo (Open-Meteo Archive)](https://open-meteo.com/en/docs/historical-weather-api)
- [🏢 Zonage & Tarif (GRDF)](https://www.grdf.fr/fournisseurs/gaz-naturel/tarif-gaz-naturel)
""", unsafe_allow_html=True)
st.sidebar.divider()

# Configuration du Backtest
st.sidebar.subheader("Configuration du Backtest")
city = st.sidebar.selectbox("Ville de référence", list(CITY_MAP.keys()))
rolling_window = st.sidebar.slider("Moyenne Mobile (jours)", 1, 60, 30)
period = st.sidebar.radio("Historique", ["1 an", "2 ans"], index=1)
lookback = 730 if period == "2 ans" else 365

current_city_data = CITY_MAP[city]
st.sidebar.info(f"📍 Zone GRDF détectée : **Zone {current_city_data['zone']}**")

# --- TRAITEMENT DES DONNÉES ---

with st.spinner("Analyse quantitative en cours..."):
    gas_raw = get_market_data("TTF=F", lookback + 30)
    weather_raw = get_weather_archive(current_city_data['lat'], current_city_data['lon'], lookback + 30)
    
    if not gas_raw.empty and not weather_raw.empty:
        df = pd.merge(gas_raw, weather_raw, left_index=True, right_index=True, how='inner')
        df.columns = ['Market_TTF', 'Temp_Mean']
        df['DJU'] = np.maximum(0, 18 - df['Temp_Mean'])
        df['Price_Final'] = df['Market_TTF'].apply(lambda x: reconstruct_retail_price(x, current_city_data['zone']))
        df['Price_SMMA'] = df['Price_Final'].rolling(window=rolling_window).mean()
        df['DJU_SMMA'] = df['DJU'].rolling(window=rolling_window).mean()
        df = df.dropna()

        # --- DASHBOARD ---

        st.title(f"Backtest Gaz vs Météo : {city}")
        st.markdown(f"Analyse de la corrélation entre les **Degrés Jours Unifiés (DJU)** et le **Prix Final Estimé**.")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        corr_pearson = df['Price_Final'].corr(df['DJU'])
        
        with m1:
            st.metric("Prix Final (Moy)", f"{df['Price_Final'].mean():.2f} €/MWh")
        with m2:
            st.metric("Total DJU (Cumul)", f"{df['DJU'].sum():.0f}")
        with m3:
            st.metric("Corrélation Pearson", f"{corr_pearson:.2f}")
        with m4:
            vol = df['Price_Final'].std()
            st.metric("Volatilité Prix", f"{vol:.2f} €")

        # Graphique Principal
        st.subheader("Visualisation de la Stratégie de Convergence")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Price_SMMA'],
            name=f"Prix Final (MM {rolling_window}j)",
            line=dict(color='#ff4b4b', width=3),
            yaxis="y1"
        ))
        fig.add_trace(go.Bar(
            x=df.index, y=df['DJU_SMMA'],
            name=f"Rigueur Climatique (DJU MM {rolling_window}j)",
            marker_color='rgba(0, 212, 255, 0.3)',
            yaxis="y2"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            yaxis=dict(title_text="Prix Final Estimé (€/MWh)", title_font=dict(color="#ff4b4b"), tickfont=dict(color="#ff4b4b")),
            yaxis2=dict(title_text="Degrés Jours Unifiés (DJU)", title_font=dict(color="#00d4ff"), tickfont=dict(color="#00d4ff"), overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        peak_date = df['Price_Final'].idxmax()
        fig.add_annotation(x=peak_date, y=df.loc[peak_date, 'Price_Final'], text="Pic Prix", showarrow=True, arrowhead=1)

        st.plotly_chart(fig, use_container_width=True)

        # Analyse Statistique
        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Analyse de Dispersion & Régression")
            fig_scatter = px.scatter(
                df, x="DJU", y="Price_Final", 
                trendline="ols", trendline_color_override="white",
                title="Sensibilité : Prix = f(DJU)",
                template="plotly_dark",
                color="Temp_Mean", color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            st.subheader("Matrice de Corrélation")
            corr_matrix = df[['Price_Final', 'DJU', 'Temp_Mean', 'Market_TTF']].corr()
            fig_heat = px.imshow(
                corr_matrix, text_auto=True, 
                color_continuous_scale='RdBu_r',
                aspect="auto", template="plotly_dark"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        st.success(f"**Analyse de l'expert (TSM) :** Le coefficient de corrélation de **{corr_pearson:.2f}** démontre l'élasticité de la demande. Sources auditées via [Yahoo](https://finance.yahoo.com) et [Open-Meteo](https://open-meteo.com).")

    else:
        st.error("Échec de la récupération des données. Vérifiez les sources en barre latérale.")

st.divider()
st.caption("Volt-Alpha v3.2 | Sources de données transparentes et auditables.")
