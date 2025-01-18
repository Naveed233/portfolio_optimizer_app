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

# ---------- Translation Dictionaries ----------
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
        "error_date": "Invalid date range selected.",
        "error_future_date": "End date cannot be in the future.",
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

def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

def validate_dates(start_date, end_date):
    """
    Validate that the date range is valid and not in the future.
    Returns a tuple of (is_valid, error_message)
    """
    today = datetime.date.today()
    
    if start_date >= end_date:
        return False, get_translated_text('en', "error_date")
    
    if end_date > today:
        return False, get_translated_text('en', "error_future_date")
        
    return True, ""

# ---------- Safe Price Fetching Function ----------
def fetch_price_data(tickers, start_date, end_date):
    """
    Safely fetch 'Adj Close' prices for given tickers and date range.
    """
    today = datetime.date.today()
    if end_date > today:
        end_date = today
        logger.warning(f"Adjusted end date to today ({today})")

    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if raw_data.empty:
            raise ValueError("No data returned. Check your tickers and date range.")
            
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' not in raw_data.columns.levels[0]:
                raise ValueError("No valid 'Adj Close' data was returned.")
            adj_close = raw_data.xs('Adj Close', level=0, axis=1)
        else:
            if 'Adj Close' not in raw_data.columns:
                raise ValueError("No valid 'Adj Close' data was returned.")
            adj_close = raw_data['Adj Close']

        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame()

        adj_close.dropna(how="all", inplace=True)
        
        if adj_close.empty:
            raise ValueError("All returned price data were NaN or missing.")

        return adj_close

    except Exception as e:
        logger.error(f"Error fetching price data: {str(e)}")
        raise

[Rest of the original code remains exactly the same, starting from class PortfolioOptimizer through to the end of main()]

if __name__ == "__main__":
    main()
