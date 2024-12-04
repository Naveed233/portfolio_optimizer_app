import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from datetime import datetime
import seaborn as sns
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Portfolio Optimizer Class
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize the PortfolioOptimizer with user-specified parameters.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        """
        Fetch historical price data and calculate daily returns.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )["Adj Close"]

        missing_tickers = set(self.tickers) - set(data.columns)
        if missing_tickers:
            st.warning(f"The following tickers were not fetched: {', '.join(missing_tickers)}")
            logger.warning(f"Missing tickers: {missing_tickers}")

        data.dropna(axis=1, inplace=True)

        if data.empty:
            logger.error("No data fetched after dropping missing tickers.")
            raise ValueError("No data fetched. Please check the tickers and date range.")

        # Update tickers to match the columns in the fetched data
        self.tickers = list(data.columns)
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers.")
        return self.tickers

    def portfolio_stats(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        Ensure weights align with current tickers.
        """
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError("Weights array length does not match the number of tickers.")
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        portfolio_return = np.dot(weights, self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility(self, target_return, max_weight=0.3):
        """
        Optimize portfolio with added weight constraints
        """
        num_assets = len(self.returns.columns)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets

        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        
            st.markdown("**Details:** You selected a 'Profit-focused Investment' strategy, aiming for maximum potential returns with an acceptance of higher risk.", unsafe_allow_html=True)

            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)

            allocation = pd.DataFrame({
                "Asset": updated_tickers,
                "Weight": np.round(optimal_weights, 4)
            })
            allocation = allocation[allocation['Weight'] > 0].reset_index(drop=True)

            st.subheader(f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)")
            st.dataframe(allocation)

            st.subheader("📊 Portfolio Performance Metrics")
            metrics = {
                "Expected Annual Return": portfolio_return * 100,
                "Annual Volatility (Risk)": portfolio_volatility * 100,
                "Sharpe Ratio": sharpe_ratio
            }
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            st.table(metrics_df)

            # Visuals
            st.subheader("📊 Visual Analysis")
            # Pie Chart for Allocation
            fig1, ax1 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            ax1.pie(allocation['Weight'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontname': 'Calibri'})
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)

            # Bar Chart for Expected Annual Return, Volatility, and Sharpe Ratio
            fig2, ax2 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            bars = metrics_df.plot(kind='bar', legend=False, ax=ax2, color=['skyblue'])
            for p in bars.patches:
                ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha='center', va='bottom', fontsize=10)
            plt.xticks(rotation=0, fontsize=10)
            plt.title("Portfolio Performance Metrics", fontsize=12)
            plt.ylabel("Value (%)", fontsize=10)
            st.pyplot(fig2)

            # Heatmap for Correlation Matrix
            st.subheader("📈 Asset Correlation Heatmap")
            correlation_matrix = optimizer.returns.corr()
            fig3, ax3 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
            plt.title("Asset Correlation Heatmap", fontsize=10)
            st.pyplot(fig3)

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            logger.exception("An unexpected error occurred during optimization.")
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
