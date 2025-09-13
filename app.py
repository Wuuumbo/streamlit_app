# -*- coding: utf-8 -*-
"""
Streamlit Stock Analyzer - Version complète
Auteur: Florentin Gaugry
Description: Application Python Streamlit pour analyse complète d'une action.
Fonctionnalités:
- Récupération des données Yahoo Finance
- Calcul de ratios financiers et indicateurs techniques
- Graphiques interactifs (prix, volume, moyennes mobiles, RSI, MACD, Bollinger)
- Dashboard Streamlit interactif
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------

def get_stock_data(ticker, period='5y', interval='1d'):
    """Récupère les données historiques de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.error("Aucune donnée disponible pour ce ticker.")
        return df
    except Exception as e:
        st.error(f"Erreur récupération données: {e}")
        return pd.DataFrame()

def calculate_financial_ratios(ticker):
    """Calcul des principaux ratios financiers"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        ratios = {}
        ratios['Current Price'] = info.get('currentPrice', np.nan)
        ratios['Market Cap'] = info.get('marketCap', np.nan)
        ratios['Trailing P/E'] = info.get('trailingPE', np.nan)
        ratios['Forward P/E'] = info.get('forwardPE', np.nan)
        ratios['Price/Book'] = info.get('priceToBook', np.nan)
        ratios['PEG Ratio'] = info.get('pegRatio', np.nan)
        ratios['Dividend Yield'] = info.get('dividendYield', np.nan)
        ratios['Return on Equity'] = info.get('returnOnEquity', np.nan)
        ratios['Return on Assets'] = info.get('returnOnAssets', np.nan)
        ratios['Debt/Equity'] = info.get('debtToEquity', np.nan)
        ratios['Beta'] = info.get('beta', np.nan)
        ratios['EPS'] = info.get('trailingEps', np.nan)
        ratios['EBITDA'] = info.get('ebitda', np.nan)
        ratios['Free Cash Flow'] = info.get('freeCashflow', np.nan)
        return pd.DataFrame(ratios, index=[0])
    except Exception as e:
        st.error(f"Erreur calcul ratios: {e}")
        return pd.DataFrame()

def plot_price_chart(df, ticker):
    """Graphique prix de l'action"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open'))
    fig.update_layout(title=f"Prix de l'action {ticker}", xaxis_title="Date", yaxis_title="Prix (€)")
    st.plotly_chart(fig, use_container_width=True)

def plot_volume_chart(df, ticker):
    """Graphique volume des transactions"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig.update_layout(title=f"Volume de transactions {ticker}", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

def add_moving_averages(df, windows=[20,50,100]):
    """Ajoute les moyennes mobiles"""
    for w in windows:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
    return df

def plot_moving_averages(df, ticker):
    """Graphique prix avec moyennes mobiles"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    for col in df.columns:
        if 'MA_' in col:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=f"Moyennes mobiles {ticker}", xaxis_title="Date", yaxis_title="Prix (€)")
    st.plotly_chart(fig, use_container_width=True)

def calculate_technical_indicators(df):
    """Calcule RSI, MACD, Bollinger Bands"""
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2*df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2*df['Close'].rolling(20).std()
    return df

def plot_technical_indicators(df, ticker):
    """Graphique RSI, MACD, Bollinger Bands"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('RSI', 'MACD', 'Bollinger Bands'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle', line=dict(dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(dash='dash')), row=3, col=1)
    fig.update_layout(height=900, title=f"Indicateurs techniques {ticker}")
    st.plotly_chart(fig, use_container_width=True)

def plot_candlestick(df, ticker):
    """Graphique chandelier"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(title=f"Candlestick Chart {ticker}", xaxis_title='Date', yaxis_title='Prix (€)')
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# STREAMLIT DASHBOARD
# ----------------------------

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📊 Analyse complète d'une action avec Streamlit")

ticker_input = st.text_input("Entrez le symbole de l'action (ex: AAPL, MSFT, TSLA):", "AAPL")
period_input = st.selectbox("Période:", ['1y','2y','5y','10y','max'], index=2)
interval_input = st.selectbox("Intervalle:", ['1d','1wk','1mo'], index=0)

if ticker_input:
    data = get_stock_data(ticker_input, period_input, interval_input)
    if not data.empty:
        st.subheader("Données historiques")
        st.dataframe(data.tail(10))
        
        st.subheader("Graphiques prix et volume")
        plot_price_chart(data, ticker_input)
        plot_volume_chart(data, ticker_input)
        
        data = add_moving_averages(data)
        st.subheader("Moyennes mobiles")
        plot_moving_averages(data, ticker_input)
        
        data = calculate_technical_indicators(data)
        st.subheader("Indicateurs techniques")
        plot_technical_indicators(data, ticker_input)
        
        st.subheader("Graphique chandelier")
        plot_candlestick(data, ticker_input)
        
        st.subheader("Ratios financiers")
        ratios = calculate_financial_ratios(ticker_input)
        st.dataframe(ratios.T)
        
        st.success("Analyse complète terminée !")
