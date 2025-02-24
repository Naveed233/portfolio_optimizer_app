import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# Download VADER lexicon (only needed on first run)
nltk.download('vader_lexicon')

# Helper function to extract ticker from asset string (e.g., "AAPL - Apple")
def extract_ticker(asset_str):
    return asset_str.split(' - ')[0].strip()

# ─────────────────────────────────────────────
# Portfolio Optimizer Class
# ─────────────────────────────────────────────
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initializes the PortfolioOptimizer with given tickers, date range, and risk-free rate.
        Parameters:
        - tickers (str): Comma-separated stock tickers.
        - start_date (str or datetime): Start date for historical data.
        - end_date (str or datetime): End date for historical data.
        - risk_free_rate (float): Annual risk-free rate in percentage.
        """
        self.tickers = [ticker.strip().upper() for ticker in tickers.split(',')]
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate / 100  # Convert % to decimal
        self.returns = None

    def fetch_data(self):
        """
        Fetches historical adjusted closing prices and computes daily returns.
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        if data.empty:
            st.error("No data fetched. Please check the tickers and date range.")
            return

        # Check if data has MultiIndex columns (multiple tickers) or single-level columns (single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                price_data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                price_data = data["Close"]
            else:
                st.error("Neither 'Adj Close' nor 'Close' data available.")
                return
        else:
            if "Adj Close" in data.columns:
                price_data = data["Adj Close"]
            elif "Close" in data.columns:
                price_data = data["Close"]
            else:
                st.error("Neither 'Adj Close' nor 'Close' data available.")
                return

        self.returns = price_data.pct_change().dropna()

    def fetch_news(self):
        """
        Fetches the top 3 latest news articles for each ticker using NewsAPI.
        Returns:
        - news_data (dict): Dictionary with tickers as keys and list of articles as values.
        """
        news_data = {}
        API_KEY = 'c1b710a8638d4e55ab8ec4415e97388a'  # <-- Replace with your NewsAPI key
        for ticker in self.tickers:
            url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])[:3]
                news_data[ticker] = articles
            else:
                news_data[ticker] = []
        return news_data

    def denoise_returns(self):
        """
        Applies PCA to denoise the returns data by retaining components that explain 95% of the variance.
        """
        if self.returns is None or self.returns.empty:
            return
        pca = PCA(n_components=min(10, len(self.returns.columns)))
        pca_returns = pca.fit_transform(self.returns)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        num_components = np.argmax(explained_variance >= 0.95) + 1
        denoised_returns = pca.inverse_transform(pca_returns[:, :num_components])
        self.returns = pd.DataFrame(denoised_returns, index=self.returns.index, columns=self.returns.columns)

    def cluster_assets(self, n_clusters=3):
        """
        Clusters assets based on their return profiles using KMeans.
        """
        if self.returns is None or self.returns.empty:
            return None
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        """
        Calculates annual return, annual volatility, and Sharpe ratio for the portfolio.
        """
        if self.returns is None or self.returns.empty:
            return 0, 0, 0
        annual_return = np.dot(weights, self.returns.mean()) * 252
        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        return annual_return, annual_volatility, sharpe_ratio

    def value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculates the Value at Risk (VaR) for the portfolio.
        """
        if self.returns is None or self.returns.empty:
            return 0
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculates the Conditional Value at Risk (CVaR) for the portfolio.
        """
        if self.returns is None or self.returns.empty:
            return 0
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        """
        Calculates the Maximum Drawdown for the portfolio.
        """
        if self.returns is None or self.returns.empty:
            return 0
        portfolio_returns = self.returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def herfindahl_hirschman_index(self, weights):
        """
        Calculates the Herfindahl-Hirschman Index (HHI) to measure portfolio diversification.
        """
        return np.sum(weights ** 2)

    def sharpe_ratio_objective(self, weights):
        """
        Objective function to maximize the Sharpe Ratio.
        """
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe  # Negative for minimization

    def optimize_sharpe_ratio(self):
        """
        Optimizes portfolio weights to maximize the Sharpe Ratio.
        """
        num_assets = len(self.tickers)
        if num_assets == 0:
            return []
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(self.sharpe_ratio_objective, initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights

    def risk_parity_objective(self, weights):
        """
        Objective function to achieve Risk Parity (equal risk contribution).
        """
        if self.returns is None or self.returns.empty:
            return 0
        portfolio_variance = np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
        marginal_contrib = np.dot(self.returns.cov() * 252, weights)
        risk_contrib = weights * marginal_contrib
        risk_parity = risk_contrib / np.sqrt(portfolio_variance) if portfolio_variance != 0 else 0
        target = 1 / len(weights)
        return np.sum((risk_parity - target) ** 2)

    def optimize_risk_parity(self):
        """
        Optimizes portfolio weights to achieve Risk Parity.
        """
        num_assets = len(self.tickers)
        if num_assets == 0:
            return []
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(self.risk_parity_objective, initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.title("📈 Portfolio Optimization with Comprehensive Risk Management")

# ─────────────────────────────────────────────
# 1) Asset Selection Section
# ─────────────────────────────────────────────
st.sidebar.header("Asset Selection")

# Define preset universes
universe_options = {
    'Tech Giants': [
        'AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon',
        'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe',
        'INTC - Intel', 'CSCO - Cisco'
    ],
    'Finance Leaders': [
        'JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo',
        'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley',
        'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab',
        'USB - U.S. Bancorp'
    ],
    'Healthcare Majors': [
        'JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth',
        'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott',
        'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher',
        'BMY - Bristol-Myers Squibb'
    ],
    'Custom': []
}

universe_choice = st.sidebar.selectbox("Select Asset Universe", options=list(universe_options.keys()), index=0)

# Custom input or multi-select from preset
if universe_choice == 'Custom':
    custom_tickers = st.sidebar.text_input("Enter custom tickers (separated by commas)")
else:
    selected_assets = st.sidebar.multiselect(
        "Select assets to add to your portfolio",
        universe_options[universe_choice],
        default=[]
    )

# Initialize session state for portfolio if it doesn't exist
if 'my_portfolio' not in st.session_state:
    st.session_state['my_portfolio'] = []

# Add chosen assets to session state
if universe_choice != 'Custom':
    if selected_assets and st.sidebar.button("Add Assets"):
        new_tickers = [extract_ticker(asset) for asset in selected_assets]
        st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
        st.sidebar.success("Assets added to portfolio!")
else:
    if st.sidebar.button("Add Assets"):
        if custom_tickers and custom_tickers.strip():
            new_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
            st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
            st.sidebar.success("Assets added to portfolio!")

# Display My Portfolio
st.sidebar.subheader("My Portfolio")
if st.session_state['my_portfolio']:
    st.sidebar.write(", ".join(st.session_state['my_portfolio']))
else:
    st.sidebar.write("No assets selected.")

# ─────────────────────────────────────────────
# 2) Optimization Parameters
# ─────────────────────────────────────────────
st.sidebar.header("Optimization Parameters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
risk_free_rate = st.sidebar.number_input("Enter the risk-free rate (in %)", value=2.0, step=0.1)

# Choose Optimization Type
optimization_type = st.sidebar.selectbox("Choose Optimization Type", ["Maximize Sharpe Ratio", "Risk Parity"])

# ─────────────────────────────────────────────
# 3) Main Optimization Button
# ─────────────────────────────────────────────
if st.sidebar.button("Optimize Portfolio"):
    if not st.session_state['my_portfolio']:
        st.error("Please add at least one asset to your portfolio.")
    else:
        # Combine selected tickers into a comma-separated string
        tickers = ", ".join(st.session_state['my_portfolio'])
        with st.spinner("Fetching data and optimizing portfolio..."):
            # Initialize Portfolio Optimizer
            optimizer = PortfolioOptimizer(tickers, start_date, end_date, risk_free_rate)
            optimizer.fetch_data()

            if optimizer.returns is None or optimizer.returns.empty:
                st.error("Failed to fetch data. Please check the tickers and date range.")
            else:
                # 1) Denoise Returns
                optimizer.denoise_returns()

                # 2) Fetch News & Analyze Sentiment
                news = optimizer.fetch_news()
                analyzer = SentimentIntensityAnalyzer()

                st.header("📰 Latest News")
                for ticker, articles in news.items():
                    if articles:
                        st.subheader(f"Top News for {ticker}")
                        for article in articles:
                            title = article['title']
                            sentiment = analyzer.polarity_scores(title)
                            compound = sentiment['compound']
                            if compound >= 0.05:
                                icon = "🟢⬆️"  # Good news
                            elif compound <= -0.05:
                                icon = "🔴⬇️"  # Bad news
                            else:
                                icon = "🔵—"   # Neutral
                            st.markdown(f"• {icon} [{title}]({article['url']}) - *{article['source']['name']}*")
                    else:
                        st.subheader(f"No recent news found for {ticker}.")

                # 3) Perform Optimization
                if optimization_type == "Maximize Sharpe Ratio":
                    opt_weights = optimizer.optimize_sharpe_ratio()
                else:
                    opt_weights = optimizer.optimize_risk_parity()

                # 4) Calculate Portfolio Statistics
                annual_return, annual_volatility, sharpe_ratio = optimizer.portfolio_stats(opt_weights)
                var_95 = optimizer.value_at_risk(opt_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(opt_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(opt_weights)
                hhi = optimizer.herfindahl_hirschman_index(opt_weights)

                # 5) Display Results
                st.header("📊 Optimized Portfolio Weights")
                weights_df = pd.DataFrame({
                    'Ticker': optimizer.tickers,
                    'Weight (%)': np.round(opt_weights * 100, 2)
                }).sort_values(by='Weight (%)', ascending=False)
                st.table(weights_df.set_index('Ticker'))

                st.header("📈 Portfolio Risk Metrics")
                st.write(f"**Annual Return:** {annual_return:.2%}")
                st.write(f"**Annual Volatility:** {annual_volatility:.2%}")
                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                st.write(f"**95% Value at Risk (VaR):** {var_95:.2%}")
                st.write(f"**95% Conditional Value at Risk (CVaR):** {cvar_95:.2%}")
                st.write(f"**Maximum Drawdown:** {max_dd:.2%}")
                st.write(f"**Herfindahl-Hirschman Index (HHI):** {hhi:.4f}")

                fig = px.pie(weights_df, names='Ticker', values='Weight (%)', title='Portfolio Composition')
                st.plotly_chart(fig)

                metrics = {
                    'Annual Return (%)': annual_return * 100,
                    'Annual Volatility (%)': annual_volatility * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    '95% VaR (%)': var_95 * 100,
                    '95% CVaR (%)': cvar_95 * 100,
                    'Maximum Drawdown (%)': max_dd * 100,
                    'HHI': hhi
                }
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                fig_metrics = px.bar(metrics_df, x='Metric', y='Value', title='Risk Metrics', text='Value')
                st.plotly_chart(fig_metrics)

                # 6) Display Clusters
                clusters = optimizer.cluster_assets()
                if clusters is not None:
                    st.header("📂 Asset Clusters")
                    cluster_df = pd.DataFrame({
                        'Ticker': optimizer.tickers,
                        'Cluster': clusters
                    }).sort_values(by='Cluster')
                    st.write(cluster_df)

                    # Optional: PCA scatter
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(optimizer.returns.T)
                    cluster_fig = px.scatter(
                        x=principal_components[:, 0],
                        y=principal_components[:, 1],
                        color=clusters.astype(str),
                        hover_data=[optimizer.tickers],
                        title='Asset Clusters Visualization (PCA)'
                    )
                    st.plotly_chart(cluster_fig)

                st.success("Portfolio optimization completed successfully!")
