import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
import datetime
import seaborn as sns
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Logging Configuration ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Streamlit Page Configuration ----------
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Translation Dictionaries (Shortened for Example) ----------
translations = {
    'en': {
        "title": "Portfolio Optimization with Advanced Features",
        "user_inputs": "User Inputs",
        "select_universe": "Select an Asset Universe:",
        "custom_tickers": "Enter stock tickers (e.g., AAPL, MSFT):",
        "add_portfolio": "Add to My Portfolio",
        "my_portfolio": "My Portfolio",
        "no_assets": "No assets added yet.",
        "optimization_parameters": "Optimization Parameters",
        "start_date": "Start Date",
        "end_date": "End Date",
        "risk_free_rate": "Enter the risk-free rate (in %):",
        "investment_strategy": "Choose your Investment Strategy:",
        "strategy_risk_free": "Risk-free Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "train_lstm": "Train LSTM Model",
        "more_info_lstm": "More Information on LSTM",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "compare_portfolios": "Compare Portfolios",
        "portfolio_analysis": "Portfolio Analysis & Results",
        "success_lstm": "LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset before training LSTM.",
        "error_no_assets_opt": "Please add at least one asset before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "Portfolio Performance Metrics",
        "visual_analysis": "Visual Analysis",
        "portfolio_composition": "Portfolio Composition",
        "portfolio_metrics": "Portfolio Metrics",
        "correlation_heatmap": "Asset Correlation Heatmap",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_lstm": "LSTM is a type of neural network well-suited for time series forecasting.",
        "explanation_sharpe_button": "Optimization for highest Sharpe Ratio aims to achieve the best risk-adjusted returns."
    }
}

# Helper function to get translated text (you can extend for multiple languages)
def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

# ---------- Safe Price Fetching Function ----------
def fetch_price_data(tickers, start_date, end_date):
    """
    Safely fetch 'Adj Close' prices for given tickers and date range.
    Ensures no future end dates, warns if data is missing or empty.
    """
    # If user picked an end date in the future, override with today's date
    today = datetime.date.today()
    if end_date > today:
        end_date = today

    # Download raw data
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Check if DataFrame is empty or if 'Adj Close' is not in columns
    if raw_data.empty or 'Adj Close' not in raw_data.columns:
        raise ValueError(
            "No valid 'Adj Close' data was returned. "
            "Check your tickers and date range (avoid future dates or invalid tickers)."
        )

    # Extract the Adj Close columns
    adj_close = raw_data['Adj Close']

    # If single ticker, adj_close might be a Series. Convert it to a DataFrame:
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    # Drop rows that are entirely NaN
    adj_close.dropna(how="all", inplace=True)
    if adj_close.empty:
        raise ValueError("All returned price data were NaN or missing. Double-check your tickers/dates.")

    return adj_close

# ---------- Portfolio Optimizer Class ----------
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def load_data(self):
        """
        Use fetch_price_data to get Adjusted Close prices and compute daily returns.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = fetch_price_data(self.tickers, self.start_date, self.end_date)
        
        # Filter columns by valid tickers
        valid_cols = [c for c in data.columns if c in self.tickers]
        data = data[valid_cols].copy()
        
        self.tickers = valid_cols  # update tickers to only valid ones
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers: {self.tickers}")

    def portfolio_stats(self, weights):
        """Return annualized return, volatility, and Sharpe ratio."""
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # ensure weights sum to 1

        # Daily average returns and covariance
        mean_daily_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        # Annualize
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        if portfolio_volatility != 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def value_at_risk(self, weights, confidence_level=0.95):
        """Calculate 1-day VaR using historical simulation."""
        portfolio_returns = self.returns.dot(weights)
        var_level = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var_level

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        """Calculate CVaR (Expected Shortfall)."""
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        """Calculate maximum drawdown for the portfolio returns."""
        portfolio_returns = self.returns.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative - peak) / peak
        return dd.min()  # negative number

    def herfindahl_hirschman_inde
