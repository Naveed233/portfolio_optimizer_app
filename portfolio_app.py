import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import plotly.express as px

# Black-Litterman Optimizer
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        # Initialize with user-specified parameters
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        # Fetch historical price data and calculate daily returns
        data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )["Adj Close"]
        if data.empty:
            raise ValueError("No data fetched. Please check the tickers and date range.")
        self.returns = data.pct_change().dropna()

    def denoise_returns(self):
        # Apply PCA for denoising
        pca = PCA(n_components=len(self.returns.columns))
        pca_returns = pca.fit_transform(self.returns)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        num_components = np.argmax(explained_variance >= 0.95) + 1
        denoised_returns = pca.inverse_transform(pca_returns[:, :num_components])
        self.returns = pd.DataFrame(denoised_returns, index=self.returns.index, columns=self.returns.columns)

    def cluster_assets(self, n_clusters=3):
        # Group similar assets into clusters using KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        # Calculate portfolio return, volatility, and Sharpe ratio
        portfolio_return = np.dot(weights, self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility(self, target_return):
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        result = minimize(lambda weights: self.portfolio_stats(weights)[1],
                          init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def generate_efficient_frontier(self, target_returns):
        efficient_portfolios = []
        for ret in target_returns:
            weights = self.min_volatility(ret)
            portfolio_return, portfolio_volatility, _ = self.portfolio_stats(weights)
            efficient_portfolios.append((portfolio_volatility, portfolio_return))
        return np.array(efficient_portfolios)

    def monte_carlo_simulation(self, num_simulations=10000):
        num_assets = len(self.returns.columns)
        results = np.zeros((3, num_simulations))
        weights_record = []

        for i in range(num_simulations):
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(weights)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio
            weights_record.append(weights)

        return results, weights_record

    def backtest_portfolio(self, weights):
        # Evaluate portfolio performance over historical data
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

# Streamlit App to interact with the user
if __name__ == "__main__":
    st.title("Portfolio Optimization with Advanced Features")

    # User Inputs
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
        ticker_list = st.multiselect("Select assets from the chosen universe:", options=universe_options[universe_choice], default=universe_options[universe_choice])
        if not ticker_list:
            st.error("Please select at least one asset.")
            st.stop()

    # Display selected assets in 'My Portfolio'
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    my_portfolio = st.session_state['my_portfolio']

    add_to_portfolio = st.multiselect("Select assets to add to My Portfolio:", options=ticker_list, default=[], help="Select assets to add them to your portfolio.")

    # Update the session state with new selections
    if add_to_portfolio:
        my_portfolio.extend([asset for asset in add_to_portfolio if asset not in my_portfolio])
        st.session_state['my_portfolio'] = my_portfolio

    # Display the updated 'My Portfolio'
    st.multiselect("My Portfolio:", options=my_portfolio, default=my_portfolio, help="These are the assets you have selected for your portfolio.")

    # Add recommend assets button under 'My Portfolio'
    recommend_button = st.button("Recommend Assets")

    if recommend_button:
        st.write("Based on your selected assets, a balanced and diversified portfolio could include a mix of assets from different sectors to minimize risk while maximizing potential gains.")
        if universe_choice == 'Tech Giants':
            st.write("Consider adding assets from Finance Leaders or Healthcare Majors to diversify your tech-heavy portfolio.")
        elif universe_choice == 'Finance Leaders':
            st.write("Consider adding assets from Tech Giants or Healthcare Majors to balance financial sector exposure.")
        elif universe_choice == 'Healthcare Majors':
            st.write("Consider adding assets from Tech Giants or Finance Leaders to create a more well-rounded portfolio.")

    start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"), max_value=pd.to_datetime("today"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"), max_value=pd.to_datetime("today"))
    risk_free_rate = st.number_input("Enter the risk-free rate (in %):", value=2.0, step=0.1) / 100

    # Add info button for why only historical data can be used
    if st.button("Why can I only use historical data?"):
        st.info("Portfolio optimizers use historical data as a proxy to estimate key inputs like expected returns, risks (volatility), and correlations between assets. This approach is based on the assumption that past performance and relationships can provide insights into future behavior.")

    strategy = st.radio("Select your strategy:", ("Risk-Free Safe Approach", "Profit-Aggressive Approach"))

    optimize_button = st.button("Optimize Portfolio")

    if optimize_button:
        try:
            # Validate dates
            if start_date >= end_date:
                st.error("Start date must be earlier than end date.")
                st.stop()

            optimizer = PortfolioOptimizer(my_portfolio, start_date, end_date, risk_free_rate)
            optimizer.fetch_data()

            # Apply selected strategy
            if strategy == "Risk-Free Safe Approach":
                optimizer.denoise_returns()
                clusters = optimizer.cluster_assets()
            elif strategy == "Profit-Aggressive Approach":
                clusters = optimizer.cluster_assets()

            # Calculate annualized returns using geometric mean
            cumulative_returns = (1 + optimizer.returns).prod() - 1
            num_years = (end_date - start_date).days / 365.25
            annualized_returns = (1 + cumulative_returns) ** (1 / num_years) - 1

            min_return = annualized_returns.min() * 100  # Convert to percentage
            max_return = annualized_returns.max() * 100  # Convert to percentage

            # Adjust min and max if they are equal
            if min_return == max_return:
                min_return -= 5
                max_return += 5

            # Define the target return slider dynamically
            specific_target_return = st.slider("Select a specific target return (in %)", min_value=round(min_return, 2), max_value=round(max_return, 2), value=round(min_return, 2), step=0.1) / 100

            # Adjust the target return validation
            tolerance = 1e-6
            if specific_target_return < (min_return / 100 - tolerance) or specific_target_return > (max_return / 100 + tolerance):
                st.error(f"The target return must be between {min_return:.2f}% and {max_return:.2f}%.")
                st.stop()

            # Optimize the portfolio for the user's specific target return
            optimal_weights = optimizer.min_volatility(specific_target_return)

            # Get portfolio stats
            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)

            # Display the optimal portfolio allocation
            allocation = pd.DataFrame({"Asset": optimizer.returns.columns, "Weight": optimal_weights.round(4)})
            allocation = allocation[allocation['Weight'] > 0]

            st.subheader(f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)")
            st.write(allocation)

            # Show portfolio performance metrics
            st.write("Portfolio Performance Metrics:")
            st.write(f"Expected Annual Return: {portfolio_return * 100:.2f}%")
            st.write(f"Annual Volatility (Risk): {portfolio_volatility * 100:.2f}%")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            # Backtest the portfolio and display cumulative returns if aggressive strategy is chosen
            if strategy == "Profit-Aggressive Approach":
                st.subheader("Backtest Portfolio Performance")
                cumulative_returns = optimizer.backtest_portfolio(optimal_weights)
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(cumulative_returns.index, cumulative_returns.values, label="Portfolio Cumulative Returns")
                plt.xlabel("Date")
                plt.ylabel("Cumulative Return")
                plt.title("Portfolio Backtesting Performance")
                plt.legend()
                st.pyplot(fig)

            # Show a bar chart of the portfolio allocation
            st.subheader("Portfolio Allocation")
            fig, ax = plt.subplots()
            allocation.set_index("Asset").plot(kind="bar", y="Weight", legend=False, ax=ax)
            plt.ylabel("Weight")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
