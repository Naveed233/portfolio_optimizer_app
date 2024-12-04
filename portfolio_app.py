import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import requests
from textblob import TextBlob
import logging
from datetime import datetime

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

        self.tickers = data.columns.tolist()
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers.")

    def denoise_returns(self):
        """
        Apply PCA for denoising the returns data.
        """
        logger.info("Applying PCA for denoising returns.")
        pca = PCA(n_components=len(self.returns.columns))
        pca_returns = pca.fit_transform(self.returns)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        num_components = np.argmax(explained_variance >= 0.95) + 1
        denoised_returns = pca.inverse_transform(pca_returns[:, :num_components])
        self.returns = pd.DataFrame(denoised_returns, index=self.returns.index, columns=self.returns.columns)
        logger.info(f"Denoised returns with {num_components} PCA components.")

    def cluster_assets(self, n_clusters=3):
        """
        Group similar assets into clusters using KMeans clustering.
        """
        logger.info(f"Clustering assets into {n_clusters} clusters.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        """
        if len(weights) != len(self.returns.columns):
            raise ValueError("Number of weights does not match number of assets.")
        
        portfolio_return = np.dot(weights, self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility(self, target_return):
        """
        Optimize portfolio to minimize volatility for a given target return.
        """
        num_assets = len(self.returns.columns)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets

        logger.info(f"Optimizing portfolio for target return: {target_return:.4f}")
        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.error("Optimization failed.")
            raise ValueError("Optimization failed. Please try different parameters.")
        
        logger.info("Optimization successful.")
        return result.x

    def backtest_portfolio(self, weights):
        """
        Evaluate portfolio performance over historical data.
        """
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

    def fetch_latest_news(self, ticker):
        """
        Fetch latest news for the given ticker using NewsAPI.
        """
        logger.info(f"Fetching news for ticker: {ticker}")
        api_key = 'c1b710a8638d4e55ab8ec4415e97388a'
        api_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&sortBy=publishedAt&language=en&pageSize=5'
        response = requests.get(api_url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return articles[:3]  # Return top 3 articles
        else:
            logger.warning(f"Failed to fetch news for {ticker}. Status code: {response.status_code}")
            return []

    def predict_movement(self, news_articles):
        """
        Predict prospective movement based on sentiment analysis of news articles.
        """
        overall_sentiment = 0
        for article in news_articles:
            try:
                analysis = TextBlob(article['title'] + '. ' + (article.get('description') or ''))
                overall_sentiment += analysis.sentiment.polarity
            except TypeError:
                continue
        
        if overall_sentiment > 0:
            return 'Up'
        elif overall_sentiment < 0:
            return 'Down'
        else:
            return 'Neutral'

# Helper Functions
def extract_ticker(asset_string):
    """
    Extract ticker symbol from asset string.
    """
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

# Streamlit App
def main():
    st.title("📈 Portfolio Optimization with Advanced Features")

    # Define preset universes
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon', 'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe', 'INTC - Intel', 'CSCO - Cisco'],
        'Finance Leaders': ['JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo', 'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley', 'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab', 'USB - U.S. Bancorp'],
        'Healthcare Majors': ['JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth', 'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott', 'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher', 'BMY - Bristol-Myers Squibb'],
        'Custom': []
    }

    universe_choice = st.selectbox("Select an asset universe:", options=list(universe_options.keys()), index=0)

    if universe_choice == 'Custom':
        custom_tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):")
        ticker_list = [ticker.strip() for ticker in custom_tickers.split(",") if ticker.strip()]
        if not ticker_list:
            st.error("Please enter at least one ticker.")
            st.stop()
    else:
        selected_universe_assets = st.multiselect("Select assets to add to My Portfolio:", universe_options[universe_choice])
        ticker_list = [extract_ticker(asset) for asset in selected_universe_assets] if selected_universe_assets else []

    # Session state for portfolio
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Update 'My Portfolio' with selected assets
    if ticker_list:
        updated_portfolio = st.session_state['my_portfolio'] + [ticker for ticker in ticker_list if ticker not in st.session_state['my_portfolio']]
        st.session_state['my_portfolio'] = updated_portfolio

    # Display 'My Portfolio'
    st.subheader("📁 My Portfolio")
    if st.session_state['my_portfolio']:
        st.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.write("No assets added yet.")

    # Button to fetch and display news sentiments
    if st.button("🔍 Get News Sentiment for Portfolio"):
        if not st.session_state['my_portfolio']:
            st.warning("Please add at least one asset to your portfolio.")
        else:
            for ticker in st.session_state['my_portfolio']:
                news_articles = PortfolioOptimizer([], '', '').fetch_latest_news(ticker)
                if news_articles:
                    sentiment = PortfolioOptimizer([], '', '').predict_movement(news_articles)
                    sentiment_icon = "🟢⬆️" if sentiment == 'Up' else "🔴⬇️" if sentiment == 'Down' else "⚪"
                    st.markdown(f"**{ticker}**: {sentiment_icon} ({sentiment})")
                    for article in news_articles:
                        try:
                            analysis = TextBlob(article['title'] + '. ' + (article.get('description') or ''))
                            article_sentiment = analysis.sentiment.polarity
                            sentiment_arrow = "🟢⬆️" if article_sentiment > 0 else "🔴⬇️" if article_sentiment < 0 else "⚪"
                            st.markdown(f"- [{article['title']}]({article['url']}) - Sentiment: {sentiment_arrow}")
                        except TypeError:
                            continue
                else:
                    st.write(f"No news available for **{ticker}**.")

    # Portfolio Optimization Section
    st.header("🔧 Optimize Your Portfolio")

    # Date Inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1), max_value=datetime.today())
    with col2:
        end_date = st.date_input("End Date", value=datetime.today(), max_value=datetime.today())

    # Risk-Free Rate Input
    risk_free_rate = st.number_input("Enter the risk-free rate (in %):", value=2.0, step=0.1) / 100

