# -*- coding: utf-8 -*-
"""
Streamlit Stock Analyzer
Version: 1.0
Auteur: Florentin Gaugry
Description: Cette application permet d'analyser en profondeur une action.
Elle récupère les données via Yahoo Finance, calcule les ratios financiers
et génère des graphiques interactifs et des tableaux.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ----------------------------
# Fonctions utilitaires
# ----------------------------

def get_stock_data(ticker, period='5y', interval='1d'):
    """
    Récupère les données historiques d'une action via Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            st.error("Aucune donnée trouvée pour le ticker fourni.")
        return hist
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()

def calculate_financial_ratios(ticker):
    """
    Calcule les ratios financiers principaux
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {}
        ratios['Price'] = info.get('currentPrice', np.nan)
        ratios['Market Cap'] = info.get('marketCap', np.nan)
        ratios['PE Ratio'] = info.get('trailingPE', np.nan)
        ratios['Forward PE'] = info.get('forwardPE', np.nan)
        ratios['PB Ratio'] = info.get('priceToBook', np.nan)
        ratios['PEG Ratio'] = info.get('pegRatio', np.nan)
        ratios['Dividend Yield'] = info.get('dividendYield', np.nan)
        ratios['ROE'] = info.get('returnOnEquity', np.nan)
        ratios['ROA'] = info.get('returnOnAssets', np.nan)
        ratios['Debt to Equity'] = info.get('debtToEquity', np.nan)
        ratios['Beta'] = info.get('beta', np.nan)
        return pd.DataFrame(ratios, index=[0])
    except Exception as e:
        st.error(f"Erreur lors du calcul des ratios financiers : {e}")
        return pd.DataFrame()

def plot_price_chart(df, ticker):
    """
    Génère un graphique du prix de l'action sur la période
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Prix de clôture'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Prix d\'ouverture'))
    fig.update_layout(title=f"Historique des prix pour {ticker}", xaxis_title='Date', yaxis_title='Prix (€)')
    st.plotly_chart(fig, use_container_width=True)

def plot_volume_chart(df, ticker):
    """
    Génère un graphique du volume
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig.update_layout(title=f"Volume des transactions pour {ticker}", xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig, use_container_width=True)

def moving_averages(df, windows=[20, 50, 100]):
    """
    Ajoute les moyennes mobiles au dataframe
    """
    for w in windows:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
    return df

def plot_moving_averages(df, ticker):
    """
    Graphique avec moyennes mobiles
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    ma_cols = [c for c in df.columns if 'MA_' in c]
    for col in ma_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=f"Prix et moyennes mobiles pour {ticker}", xaxis_title='Date', yaxis_title='Prix (€)')
    st.plotly_chart(fig, use_container_width=True)

def calculate_technical_indicators(df):
    """
    Calcule quelques indicateurs techniques simples
    """
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def plot_technical_indicators(df, ticker):
    """
    Graphique RSI et MACD
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('RSI', 'MACD'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'), row=2, col=1)
    fig.update_layout(height=600, title_text=f"Indicateurs techniques pour {ticker}")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Analyse approfondie d'une action")

# Input utilisateur
ticker_input = st.text_input("Entrez le symbole de l'action (ex: AAPL, MSFT, TSLA):", value="AAPL")
periode_input = st.selectbox("Période des données historiques:", ['1y','2y','5y','10y','max'], index=2)
interval_input = st.selectbox("Intervalle:", ['1d','1wk','1mo'], index=0)

if ticker_input:
    with st.spinner("Récupération des données..."):
        data = get_stock_data(ticker_input, period=periode_input, interval=interval_input)
        
        if not data.empty:
            st.subheader("Données historiques")
            st.dataframe(data.tail(10))
            
            st.subheader("Graphique des prix")
            plot_price_chart(data, ticker_input)
            
            st.subheader("Graphique du volume")
            plot_volume_chart(data, ticker_input)
            
            data = moving_averages(data)
            st.subheader("Moyennes mobiles")
            plot_moving_averages(data, ticker_input)
            
            data = calculate_technical_indicators(data)
            # plot_technical_indicators(data, ticker_input)  # nécessite make_subplots import
            
            st.subheader("Ratios financiers")
            ratios_df = calculate_financial_ratios(ticker_input)
            st.dataframe(ratios_df.T)
            
            st.success("Analyse terminée !")
