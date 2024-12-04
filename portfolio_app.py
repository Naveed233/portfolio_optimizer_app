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

# Initialize TextBlob for sentiment analysis
@st.cache_resource
def initialize_textblob():
    return TextBlob

# Portfolio Optimizer Class
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize the PortfolioOptimizer with user-specified parameters.

        Args:
            tickers (list): List of stock ticker symbols.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD).
            risk_free_rate (float): Annual risk-free rate (default: 0.02).
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        """
        Fetch historical price data and calculate daily returns.

        Raises:
            ValueError: If no data is fetched or tickers mismatch after cleaning.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )["Adj Close"]

        # Identify and handle missing data
        missing_tickers = set(self.tickers) - set(data.columns)
        if missing_tickers:
            st.warning(f"The following tickers were not fetched and will be excluded: {', '.join(missing_tickers)}")
            logger.warning(f"Missing tickers: {missing_tickers}")

        # Drop columns with missing data
        data.dropna(axis=1, inplace=True)

        if data.empty:
            logger.error("No data fetched after dropping missing tickers.")
            raise ValueError("No data fetched. Please check the tickers and date range.")

        # Update the tickers list to match the fetched data
        self.tickers = data.columns.tolist()
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers.")

    def denoise_returns(self):
        """
        Apply PCA for denoising the returns data.
        
        Raises:
            ValueError: If denoising alters the number of assets.
        """
        logger.info("Applying PCA for denoising returns.")
        pca = PCA(n_components=len(self.returns.columns))
        pca_returns = pca.fit_transform(self.returns)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        num_components = np.argmax(explained_variance >= 0.95) + 1
        denoised_returns = pca.inverse_transform(pca_returns[:, :num_components])
        self.returns = pd.DataFrame(denoised_returns, index=self.returns.index, columns=self.returns.columns)

        # Verify dimensions remain unchanged
        if self.returns.shape[1] != len(self.tickers):
            logger.error("Denoising altered the number of assets.")
            raise ValueError("Denoising altered the number of assets.")
        logger.info(f"Denoised returns with {num_components} PCA components.")

    def cluster_assets(self, n_clusters=3):
        """
        Group similar assets into clusters using KMeans clustering.

        Args:
            n_clusters (int): Number of clusters (default: 3).

        Returns:
            array: Cluster labels for each asset.
        """
        logger.info(f"Clustering assets into {n_clusters} clusters.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.

        Args:
            weights (array): Portfolio weights.

        Returns:
            tuple: (portfolio_return, portfolio_volatility, sharpe_ratio)

        Raises:
            ValueError: If weights length does not match number of assets.
        """
        if len(weights) != len(self.returns.columns):
            raise ValueError(f"Number of weights ({len(weights)}) does not match number of assets ({len(self.returns.columns)}).")

        portfolio_return = np.dot(weights, self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility(self, target_return):
        """
        Optimize portfolio to minimize volatility for a given target return.

        Args:
            target_return (float): Target annual return.

        Returns:
            array: Optimal portfolio weights.

        Raises:
            ValueError: If optimization fails.
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

    def generate_efficient_frontier(self, target_returns):
        """
        Generate points on the efficient frontier.

        Args:
            target_returns (array): Array of target returns.

        Returns:
            array: Efficient portfolios (volatility, return).
        """
        efficient_portfolios = []
        logger.info("Generating efficient frontier.")
        for ret in target_returns:
            weights = self.min_volatility(ret)
            portfolio_return, portfolio_volatility, _ = self.portfolio_stats(weights)
            efficient_portfolios.append((portfolio_volatility, portfolio_return))
        return np.array(efficient_portfolios)

    def monte_carlo_simulation(self, num_simulations=10000):
        """
        Perform Monte Carlo simulation to generate random portfolios.

        Args:
            num_simulations (int): Number of simulations (default: 10000).

        Returns:
            tuple: (results array, weights_record list)
        """
        num_assets = len(self.returns.columns)
        results = np.zeros((3, num_simulations))
        weights_record = []

        logger.info(f"Running Monte Carlo simulation with {num_simulations} simulations.")

        for i in range(num_simulations):
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(weights)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio
            weights_record.append(weights)

        logger.info("Monte Carlo simulation completed.")
        return results, weights_record

    def backtest_portfolio(self, weights):
        """
        Evaluate portfolio performance over historical data.

        Args:
            weights (array): Portfolio weights.

        Returns:
            pandas.Series: Cumulative returns.
        """
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

    def fetch_latest_news(self, ticker, api_key):
        """
        Fetch latest news for the given ticker using NewsAPI.

        Args:
            ticker (str): Stock ticker symbol.
            api_key (str): NewsAPI key.

        Returns:
            list: List of news articles.
        """
        logger.info(f"Fetching news for ticker: {ticker}")
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

        Args:
            news_articles (list): List of news articles.

        Returns:
            str: Predicted movement ('Up', 'Down', 'Neutral').
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

    Args:
        asset_string (str): Asset string (e.g., "AAPL - Apple").

    Returns:
        str: Ticker symbol.
    """
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

def display_portfolio(allocation_df):
    """
    Display portfolio allocation as a bar chart.

    Args:
        allocation_df (pandas.DataFrame): DataFrame with 'Asset' and 'Weight' columns.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    allocation_df.set_index("Asset").plot(kind="bar", y="Weight", legend=False, ax=ax, color='skyblue')
    plt.ylabel("Weight")
    plt.title("Portfolio Allocation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def display_cumulative_returns(cumulative_returns):
    """
    Display cumulative returns as a line chart.

    Args:
        cumulative_returns (pandas.Series): Series of cumulative returns.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_returns.plot(ax=ax, color='green')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Backtesting Performance")
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("📈 Portfolio Optimization with Advanced Features")

    # Sidebar for API Key input
    st.sidebar.header("Configuration")
    news_api_key = st.sidebar.text_input("Enter your NewsAPI Key:", type="password")

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
        if universe_choice == 'Custom' and not ticker_list:
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
        if not news_api_key:
            st.warning("Please enter your NewsAPI key in the sidebar to fetch news.")
        elif not st.session_state['my_portfolio']:
            st.warning("Please add at least one asset to your portfolio.")
        else:
            for ticker in st.session_state['my_portfolio']:
                news_articles = PortfolioOptimizer([], '', '').fetch_latest_news(ticker, news_api_key)
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

    # Recommendation Button
    if st.button("💡 Recommend Assets"):
        if not st.session_state['my_portfolio']:
            st.warning("Please add assets to your portfolio before getting recommendations.")
        else:
            st.write("### Asset Recommendations for Diversification")
            current_universe = None
            for universe, assets in universe_options.items():
                for asset in assets:
                    ticker = extract_ticker(asset)
                    if ticker in st.session_state['my_portfolio']:
                        current_universe = universe
                        break
                if current_universe:
                    break

            if current_universe == 'Tech Giants':
                st.write("Consider adding assets from **Finance Leaders** or **Healthcare Majors** to diversify your tech-heavy portfolio.")
            elif current_universe == 'Finance Leaders':
                st.write("Consider adding assets from **Tech Giants** or **Healthcare Majors** to balance financial sector exposure.")
            elif current_universe == 'Healthcare Majors':
                st.write("Consider adding assets from **Tech Giants** or **Finance Leaders** to create a more well-rounded portfolio.")
            else:
                st.write("Consider adding assets from different sectors to diversify your portfolio.")

    st.markdown("---")

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

    # Strategy Selection
    strategy = st.radio("Select Your Strategy:", ("Risk-Free Safe Approach", "Profit-Aggressive Approach"))

    # Optimize Button
    if st.button("📈 Optimize Portfolio"):
        if not st.session_state['my_portfolio']:
            st.error("Please add at least one asset to your portfolio before optimization.")
            st.stop()

        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            st.stop()

        if not news_api_key:
            st.warning("For better insights, consider entering your NewsAPI key in the sidebar.")

        try:
            # Clean tickers
            clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]

            # Initialize optimizer
            optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
            optimizer.fetch_data()

            # Apply strategy
            if strategy == "Risk-Free Safe Approach":
                optimizer.denoise_returns()

            # Calculate annualized returns
            cumulative_returns = (1 + optimizer.returns).prod() - 1
            num_years = (end_date - start_date).days / 365.25
            annualized_returns = (1 + cumulative_returns) ** (1 / num_years) - 1

            min_return = annualized_returns.min() * 100  # Percentage
            max_return = annualized_returns.max() * 100  # Percentage

            # Adjust if min and max are equal
            if min_return == max_return:
                min_return -= 5
                max_return += 5

            # Target Return Slider
            specific_target_return = st.slider(
                "Select a specific target return (in %)",
                min_value=round(min_return, 2),
                max_value=round(max_return, 2),
                value=round(min_return, 2),
                step=0.1
            ) / 100

            # Validate target return
            tolerance = 1e-6
            if specific_target_return < (min_return / 100 - tolerance) or specific_target_return > (max_return / 100 + tolerance):
                st.error(f"The target return must be between {min_return:.2f}% and {max_return:.2f}%.")
                st.stop()

            # Optimize portfolio
            optimal_weights = optimizer.min_volatility(specific_target_return)
            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)

            # Prepare Allocation DataFrame
            allocation = pd.DataFrame({
                "Asset": optimizer.tickers,
                "Weight": np.round(optimal_weights, 4)
            })
            allocation = allocation[allocation['Weight'] > 0].reset_index(drop=True)

            # Display Allocation
            st.subheader(f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)")
            st.dataframe(allocation)

            # Display Performance Metrics
            st.subheader("📊 Portfolio Performance Metrics")
            metrics = {
                "Expected Annual Return": f"{portfolio_return * 100:.2f}%",
                "Annual Volatility (Risk)": f"{portfolio_volatility * 100:.2f}%",
                "Sharpe Ratio": f"{sharpe_ratio:.2f}"
            }
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            st.table(metrics_df)

            # Display Allocation Bar Chart
            st.subheader("📉 Portfolio Allocation")
            display_portfolio(allocation)

            # Backtest (Only for Profit-Aggressive Approach)
            if strategy == "Profit-Aggressive Approach":
                st.subheader("🔍 Backtest Portfolio Performance")
                backtest_cumulative_returns = optimizer.backtest_portfolio(optimal_weights)
                display_cumulative_returns(backtest_cumulative_returns)

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            logger.exception("An unexpected error occurred during optimization.")
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
