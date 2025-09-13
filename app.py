# -*- coding: utf-8 -*-
"""
Quantitative Trading Platform - Skeleton
File: quant_trading_skeleton.py
Author: Generated for Florentin Gaugry
Purpose:
    Single-file skeleton providing classes, methods and a clear structure
    for the 6-module quantitative trading system described in the design
    document. This skeleton is intentionally verbose and heavily commented
    so it acts both as a starting codebase and as documentation for a team.

How to use:
    - Drop this file into your project and import classes into your modules.
    - Fill the TODO sections with real implementations and API keys.
    - Use the provided Streamlit / CLI examples at the bottom to run a quick MVP.

Notes:
    - This is only a skeleton: it focuses on architecture, interfaces and
      data flow rather than on production-ready implementations.
    - Dependencies: pandas, numpy, yfinance, requests, statsmodels, scipy,
      plotly, streamlit (only for the demo at the bottom). Add them to your
      environment before using interactive parts.

"""

# Standard imports
import time
import math
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# Data & numerics
import pandas as pd
import numpy as np

# Finance data
import yfinance as yf

# Stats & econ
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.optimize as optimize

# Visualization (used in demo)
import plotly.graph_objects as go
import plotly.express as px

# Optional: streamlit demo at end
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities & Types
# ---------------------------------------------------------------------------

@dataclass
class TickerSpec:
    """Simple container for ticker metadata."""
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    exchange: Optional[str] = None


def safe_div(a, b):
    try:
        if b == 0 or b is None or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Module 1: Data Engine & Universe Management
# ---------------------------------------------------------------------------

class DataEngine:
    """
    Responsible for retrieving, caching and serving market and fundamental data.

    Responsibilities:
    - Connect to data providers (yfinance by default)
    - Normalize OHLCV, fundamentals, and options chains
    - Provide a light caching layer to avoid repeated downloads
    - Export and import snapshots
    """

    def __init__(self, cache_dir: str = './data_cache'):
        self.cache_dir = cache_dir
        # simple in-memory cache: { (ticker, period, interval): DataFrame }
        self._history_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        # fundamentals cache
        self._fund_cache: Dict[str, Dict[str, Any]] = {}
        # universe management
        self.universes: Dict[str, List[str]] = {}

    # -----------------
    # Universe management
    # -----------------
    def create_universe(self, name: str, tickers: List[str]):
        """Create or overwrite a universe (list of tickers)."""
        self.universes[name] = [t.upper() for t in tickers]
        logger.info(f"Universe '{name}' set with {len(tickers)} tickers")

    def get_universe(self, name: str) -> List[str]:
        return self.universes.get(name, [])

    def list_universes(self) -> List[str]:
        return list(self.universes.keys())

    # -----------------
    # Data retrieval
    # -----------------
    def fetch_history(self, ticker: str, period: str = '5y', interval: str = '1d', force_refresh: bool=False) -> pd.DataFrame:
        key = (ticker.upper(), period, interval)
        if not force_refresh and key in self._history_cache:
            logger.debug(f"Returning cached history for {ticker} {period}@{interval}")
            return self._history_cache[key].copy()

        # Retrieve via yfinance for prototype
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=False, actions=False)
            # Normalize column names and ensure datetime index
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            df = df.rename(columns=lambda c: c.strip())
            # Keep essential columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[['Open','High','Low','Close','Volume']]
            df.index = pd.to_datetime(df.index)
            # store
            self._history_cache[key] = df.copy()
            logger.info(f"Fetched history for {ticker}: {df.shape[0]} rows")
            return df
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str, force_refresh: bool=False) -> Dict[str, Any]:
        t = ticker.upper()
        if not force_refresh and t in self._fund_cache:
            return self._fund_cache[t]
        try:
            yf_t = yf.Ticker(t)
            info = yf_t.info if hasattr(yf_t, 'info') else {}
            # Keep a shallow subset to avoid huge payloads
            keys_wanted = ['sector','marketCap','trailingPE','forwardPE','priceToBook','pegRatio',
                           'returnOnEquity','returnOnAssets','debtToEquity','beta','ebitda','freeCashflow']
            out = {k: info.get(k, None) for k in keys_wanted}
            self._fund_cache[t] = out
            logger.info(f"Fetched fundamentals for {t}")
            return out
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {t}: {e}")
            return {}

    # -----------------
    # Persistence (simple JSON/CSV snapshots)
    # -----------------
    def save_snapshot(self, ticker: str, df: pd.DataFrame, filename: Optional[str] = None):
        if filename is None:
            filename = f"{self.cache_dir}/{ticker}_snapshot.csv"
        df.to_csv(filename)
        logger.info(f"Saved snapshot {filename}")

    def load_snapshot(self, filename: str) -> pd.DataFrame:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        return df


# ---------------------------------------------------------------------------
# Module 2: Market Regime Dashboard
# ---------------------------------------------------------------------------

class MarketRegimeDashboard:
    """
    Provides a fast, visual summary of market conditions (volatility, trend, breadth).

    Expected inputs: VIX/VSTOXX series, sector returns, breadth metrics.
    For a prototype, uses simple moving-average thresholds and volatility percentiles.
    """

    def __init__(self, data_engine: DataEngine):
        self.de = data_engine

    def compute_iv_rank(self, iv_series: pd.Series) -> float:
        """Simplified IV Rank: percentile of current IV in trailing window (1 year).

        Returns a value between 0 and 100.
        """
        window_days = min(len(iv_series), 252)
        if window_days < 10:
            return np.nan
        window = iv_series.dropna().tail(window_days)
        current = window.iloc[-1]
        rank = (window < current).sum() / len(window) * 100
        return float(rank)

    def market_trend_score(self, benchmark_series: pd.Series) -> float:
        """Compute a simple trend score: percent above MA50 & MA200"""
        s = benchmark_series.dropna()
        if len(s) < 200:
            return np.nan
        ma50 = s.rolling(50).mean()
        ma200 = s.rolling(200).mean()
        pct_above_50 = (s > ma50).tail(1).mean() * 100
        pct_above_200 = (s > ma200).tail(1).mean() * 100
        # custom aggregation
        score = 0.6 * pct_above_200 + 0.4 * pct_above_50
        return float(score)

    def compute_put_call_ratio(self, options_data: pd.DataFrame) -> float:
        """Compute put/call ratio given options volume data.

        options_data expected columns: ['type' ('put'/'call'), 'volume']
        """
        try:
            puts = options_data[options_data['type']=='put']['volume'].sum()
            calls = options_data[options_data['type']=='call']['volume'].sum()
            return safe_div(puts, calls)
        except Exception:
            return np.nan

    def regime_label(self, vix_series: pd.Series, benchmark_series: pd.Series) -> str:
        """Return a small label like 'Bull / Volatility Low'"""
        iv_rank = self.compute_iv_rank(vix_series)
        trend = self.market_trend_score(benchmark_series)
        label = []
        # Trend
        if trend is np.nan or math.isnan(trend):
            label.append('Trend: N/A')
        elif trend > 60:
            label.append('Bull')
        elif trend < 40:
            label.append('Bear')
        else:
            label.append('Neutral')
        # Vol
        if iv_rank is np.nan or math.isnan(iv_rank):
            label.append('IV: N/A')
        elif iv_rank < 30:
            label.append('Volatility Low')
        elif iv_rank > 70:
            label.append('Volatility High')
        else:
            label.append('Volatility Mid')
        return ' / '.join(label)


# ---------------------------------------------------------------------------
# Module 3: Quantitative Screener
# ---------------------------------------------------------------------------

@dataclass
class ScreenerConfig:
    pe_max: Optional[float] = None
    ev_ebitda_max: Optional[float] = None
    roe_min: Optional[float] = None
    debt_equity_max: Optional[float] = None
    momentum_6m_min: Optional[float] = None
    volatility_max: Optional[float] = None
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'quality': 0.4, 'value': 0.3, 'momentum': 0.3
    })


class Screener:
    """
    Quantitative screener that scores tickers across multiple factors.

    The screener performs per-ticker factor computation and returns a DataFrame
    with factor values and an aggregated score that can be sorted/filtered.
    """

    def __init__(self, data_engine: DataEngine, config: ScreenerConfig = ScreenerConfig()):
        self.de = data_engine
        self.config = config

    def compute_factors(self, ticker: str, period: str = '1y') -> Dict[str, Any]:
        """Compute factor values for one ticker (value, quality, momentum, volatility)."""
        # Fetch fundamentals & prices
        fund = self.de.fetch_fundamentals(ticker)
        hist = self.de.fetch_history(ticker, period=period)
        out: Dict[str, Any] = {}
        # Value factors
        out['PE'] = fund.get('trailingPE', np.nan)
        out['PriceToBook'] = fund.get('priceToBook', np.nan)
        out['EV_EBITDA'] = fund.get('enterpriseToEbitda', np.nan)
        # Quality
        out['ROE'] = fund.get('returnOnEquity', np.nan)
        out['DebtEquity'] = fund.get('debtToEquity', np.nan)
        # Momentum: 6 months return
        try:
            ret_6m = hist['Close'].pct_change().rolling(window=126).apply(lambda x: (1+x).prod()-1).dropna()
            out['Mom6M'] = ret_6m.iloc[-1] if len(ret_6m) else np.nan
        except Exception:
            out['Mom6M'] = np.nan
        # Volatility: annualized std
        out['Volatility'] = hist['Close'].pct_change().std() * math.sqrt(252) if len(hist)>10 else np.nan
        # Score composition
        out['Score'] = self.aggregate_score(out)
        out['Ticker'] = ticker
        return out

    def aggregate_score(self, factor_values: Dict[str, Any]) -> float:
        """Create a composite score using config weights.

        Expected keys in factor_values: ROE, PE, Mom6M, EV_EBITDA, DebtEquity
        The function normalizes each factor to a 0-1 scale using simple heuristics.
        """
        try:
            # Quality score (higher ROE better, lower DebtEquity better)
            roe = factor_values.get('ROE', np.nan) or 0
            de = factor_values.get('DebtEquity', np.nan) or 100
            q_score = (np.tanh((roe/0.15)) + (1 - np.tanh(de/1.0))) / 2
            # Value score (lower PE and EV/EBITDA better)
            pe = factor_values.get('PE', np.nan) or 1000
            ev = factor_values.get('EV_EBITDA', np.nan) or 1000
            v_score = (1 / (1 + np.log1p(pe))) * 0.6 + (1 / (1 + np.log1p(ev))) * 0.4
            # Momentum score (higher is better)
            mom = factor_values.get('Mom6M', 0) or 0
            m_score = min(max((mom + 1) / 2, 0), 1)
            weights = self.config.score_weights
            total = weights['quality'] * q_score + weights['value'] * v_score + weights['momentum'] * m_score
            return float(total)
        except Exception as e:
            logger.error(f"Error aggregating score: {e}")
            return 0.0

    def run_screener(self, tickers: List[str], period: str = '1y') -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                fv = self.compute_factors(t, period=period)
                rows.append(fv)
            except Exception as e:
                logger.warning(f"Skipping {t} due to error: {e}")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index('Ticker')
        df = df.sort_values(by='Score', ascending=False)
        return df


# ---------------------------------------------------------------------------
# Module 4: Tactical Analyzer (single-asset deep analysis & pair trading)
# ---------------------------------------------------------------------------

class TacticalAnalyzer:
    """
    Detailed per-asset analysis and pairs trading utilities.

    Features:
    - Price charting helpers
    - Volatility (historical vs implied if available)
    - Z-score calculations for price and spread
    - Cointegration (ADF) test for pair trading
    """

    def __init__(self, data_engine: DataEngine):
        self.de = data_engine

    def zscore(self, series: pd.Series, window: int = 200) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std

    def pair_spread(self, series_x: pd.Series, series_y: pd.Series, method: str = 'ratio') -> pd.Series:
        """Return the spread between two series; method: 'ratio' or 'diff'."""
        s_x = series_x.dropna()
        s_y = series_y.dropna()
        # Align indexes
        df = pd.concat([s_x, s_y], axis=1).dropna()
        if df.shape[0] < 10:
            return pd.Series(dtype=float)
        if method == 'ratio':
            return df.iloc[:,0] / df.iloc[:,1]
        else:
            return df.iloc[:,0] - df.iloc[:,1]

    def adf_test(self, series: pd.Series) -> Dict[str, Any]:
        """Run Augmented Dickey-Fuller test and return statistic and p-value."""
        try:
            res = adfuller(series.dropna())
            return {'adf_stat': res[0], 'pvalue': res[1], 'usedlag': res[2], 'nobs': res[3]}
        except Exception as e:
            logger.error(f"ADF test error: {e}")
            return {'adf_stat': np.nan, 'pvalue': np.nan}

    def analyze_ticker(self, ticker: str, period: str = '2y') -> Dict[str, Any]:
        hist = self.de.fetch_history(ticker, period=period)
        if hist.empty:
            return {}
        close = hist['Close']
        z = self.zscore(close)
        return {'history': hist, 'zscore': z}

    def analyze_pair(self, ticker_a: str, ticker_b: str, period: str = '2y') -> Dict[str, Any]:
        a = self.de.fetch_history(ticker_a, period=period)['Close']
        b = self.de.fetch_history(ticker_b, period=period)['Close']
        spread = self.pair_spread(a, b, method='ratio')
        z = self.zscore(spread, window=100)
        adf = self.adf_test(spread)
        return {'spread': spread, 'zscore': z, 'adf': adf}


# ---------------------------------------------------------------------------
# Module 5: Portfolio Analysis & Risk Management
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: float
    entry_price: float
    entry_date: Optional[pd.Timestamp] = None


class Portfolio:
    """
    Simple portfolio container to compute P&L, VaR, drawdown, and attribution.
    This is a prototyping class for analytics rather than execution.
    """

    def __init__(self, positions: Optional[List[Position]] = None, cash: float = 0.0):
        self.positions = positions or []
        self.cash = cash

    def market_value(self, price_map: Dict[str, float]) -> float:
        mv = 0.0
        for pos in self.positions:
            price = price_map.get(pos.ticker, pos.entry_price)
            mv += pos.quantity * price
        return mv + self.cash

    def current_pnl(self, price_map: Dict[str, float]) -> float:
        pnl = 0.0
        for pos in self.positions:
            price = price_map.get(pos.ticker, pos.entry_price)
            pnl += pos.quantity * (price - pos.entry_price)
        return pnl

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.positions:
            rows.append({'Ticker': p.ticker, 'Qty': p.quantity, 'Entry': p.entry_price, 'EntryDate': p.entry_date})
        return pd.DataFrame(rows)


class RiskEngine:
    """
    Risk analytics: portfolio VaR (parametric & historical), drawdowns, sharpe, sortino.
    """

    @staticmethod
    def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
        if returns.empty:
            return np.nan
        return -np.quantile(returns.dropna(), alpha)

    @staticmethod
    def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
        if returns.empty:
            return np.nan
        mu = returns.mean()
        sigma = returns.std()
        z = abs(scipy.stats.norm.ppf(alpha)) if 'scipy' in globals() else 1.645
        return -(mu - z * sigma)

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        return drawdown.min()

    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf_rate_annual: float = 0.0) -> float:
        if returns.empty:
            return np.nan
        rf_daily = (1 + rf_rate_annual) ** (1/252) - 1
        excess = returns - rf_daily
        return safe_div(excess.mean() * math.sqrt(252), excess.std())

    @staticmethod
    def sortino_ratio(returns: pd.Series, rf_rate_annual: float = 0.0) -> float:
        if returns.empty:
            return np.nan
        rf_daily = (1 + rf_rate_annual) ** (1/252) - 1
        excess = returns - rf_daily
        downside = excess[excess < 0]
        downside_std = downside.std()
        return safe_div(excess.mean() * math.sqrt(252), downside_std)


# ---------------------------------------------------------------------------
# Module 6: Backtesting Engine (simple event-driven backtest skeleton)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    ticker: str
    date: pd.Timestamp
    quantity: float
    price: float
    side: str  # 'buy' / 'sell'


class BacktestResult:
    def __init__(self):
        self.equity_curve = pd.Series(dtype=float)
        self.trades: List[Trade] = []
        self.metrics: Dict[str, Any] = {}


class Strategy:
    """
    Abstract base class for strategies. Implement `generate_signals` and optionally `on_fill`.
    """
    def __init__(self, name: str = 'abstract'):
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Takes historical OHLCV and returns a DataFrame with columns 'signal' (1 buy, -1 sell, 0 hold).

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def on_fill(self, trade: Trade):
        """Optional hook called when a trade is executed (for stateful strategies)."""
        pass


class SMA_Crossover_Strategy(Strategy):
    """Example strategy: SMA short/long crossover."""
    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__(name=f"SMA_{short_window}_{long_window}")
        self.short = short_window
        self.long = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['sma_short'] = df['Close'].rolling(self.short).mean()
        df['sma_long'] = df['Close'].rolling(self.long).mean()
        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
        return df[['signal']]


class SimpleBacktester:
    """
    Naive backtester: assumes fills at next open price and fixed position sizing per trade.
    Produces equity curve and trade list. Useful for quick hypothesis testing.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run(self, price_df: pd.DataFrame, signals: pd.DataFrame, ticker: str, position_size: float = 0.1) -> BacktestResult:
        # price_df: index datetime with columns Open/High/Low/Close/Volume
        # signals: index datetime with 'signal'
        res = BacktestResult()
        df = price_df.join(signals, how='left').fillna(method='ffill').fillna(0)
        cash = self.initial_capital
        position = 0.0
        equity_hist = []
        dates = []
        for idx, row in df.iterrows():
            sig = row.get('signal', 0)
            price = row.get('Open', row.get('Close', np.nan))
            if np.isnan(price):
                equity_hist.append(cash + position * 0.0)
                dates.append(idx)
                continue
            # Entry
            if sig == 1 and position == 0:
                # buy with fixed fraction of capital
                alloc = cash * position_size
                qty = alloc / price if price > 0 else 0
                cash -= qty * price
                position += qty
                trade = Trade(ticker=ticker, date=idx, quantity=qty, price=price, side='buy')
                res.trades.append(trade)
            # Exit
            elif sig == -1 and position > 0:
                cash += position * price
                trade = Trade(ticker=ticker, date=idx, quantity=position, price=price, side='sell')
                res.trades.append(trade)
                position = 0
            equity = cash + position * price
            equity_hist.append(equity)
            dates.append(idx)
        res.equity_curve = pd.Series(equity_hist, index=dates)
        # Compute metrics
        ret = res.equity_curve.pct_change().dropna()
        res.metrics['CAGR'] = ((res.equity_curve.iloc[-1] / res.equity_curve.iloc[0]) ** (252.0 / len(res.equity_curve)) - 1) if len(res.equity_curve)>1 else np.nan
        res.metrics['Sharpe'] = safe_div(ret.mean() * math.sqrt(252), ret.std()) if not ret.empty else np.nan
        res.metrics['MaxDrawdown'] = RiskEngine.max_drawdown(res.equity_curve)
        return res


# ---------------------------------------------------------------------------
# Repo skeleton generation and project scaffolding helpers
# ---------------------------------------------------------------------------

def create_repo_skeleton(path: str = './quant_project'):
    """Create a basic repository layout for the quant project.

    This function is intentionally minimal to avoid filesystem side effects in tests.
    """
    import os
    os.makedirs(path, exist_ok=True)
    subdirs = ['data', 'notebooks', 'src', 'tests', 'docs']
    for sd in subdirs:
        os.makedirs(f"{path}/{sd}", exist_ok=True)
    readme = f"{path}/README.md"
    if not os.path.exists(readme):
        with open(readme, 'w') as f:
            f.write('# Quant Project\n\nRepository scaffolded by quant_trading_skeleton.py')
    logger.info(f"Created repo skeleton at {path}")


# ---------------------------------------------------------------------------
# Example Streamlit demo (very simple dashboard using the modules above)
# ---------------------------------------------------------------------------

def streamlit_demo():
    if not STREAMLIT_AVAILABLE:
        print('Streamlit not available in this environment. Skipping demo.')
        return
    st.set_page_config(layout='wide')
    st.title('Quant Platform - Demo')

    de = DataEngine()
    md = MarketRegimeDashboard(de)
    sc = Screener(de)
    ta = TacticalAnalyzer(de)

    # Sidebar
    tickers = st.sidebar.text_input('Tickers (comma separated)', 'AAPL,MSFT,TSLA')
    period = st.sidebar.selectbox('Period', ['1y','2y','5y'], index=2)
    tick_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]

    st.sidebar.write('Universes')
    if st.sidebar.button('Create sample universe'):
        de.create_universe('sample', tick_list)
        st.sidebar.success('Universe created')

    # Screener run
    if st.sidebar.button('Run Screener'):
        with st.spinner('Running screener...'):
            df = sc.run_screener(tick_list, period=period)
            st.subheader('Screener Results')
            st.dataframe(df)

    # Single ticker deep analysis
    ticker = st.sidebar.selectbox('Deep analyze', tick_list)
    if ticker:
        with st.spinner('Fetching data...'):
            res = ta.analyze_ticker(ticker, period='2y')
            if res:
                st.subheader(f'{ticker} Price')
                hist = res['history']
                fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], name='Close')])
                st.plotly_chart(fig, use_container_width=True)
                st.subheader('Z-Score')
                st.line_chart(res['zscore'])


# ---------------------------------------------------------------------------
# If run as script, provide a CLI-style quick demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Quick demo of creating skeletons and running a minimal flow
    print('Quant trading skeleton module. Running a quick demo...')
    create_repo_skeleton('./quant_project_demo')
    de = DataEngine()
    # sample tickers
    tickers = ['AAPL', 'MSFT']
    de.create_universe('tech_us', tickers)
    sc = Screener(de)
    df = sc.run_screener(tickers, period='1y')
    print('Screener results (head):')
    print(df.head())
    print('\nTo run the Streamlit demo, execute: streamlit run quant_trading_skeleton.py')
