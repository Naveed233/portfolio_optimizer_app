import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define a class to handle portfolio optimization tasks
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
        # Reduce noise in returns data using Principal Component Analysis (PCA)
        num_assets = len(self.returns.columns)
        n_components = min(num_assets, len(self.returns))
        pca = PCA(n_components=n_components)
        pca_returns = pca.fit_transform(self.returns)
        denoised_returns = pca.inverse_transform(pca_returns)
        self.returns = pd.DataFrame(
            denoised_returns, index=self.returns.index, columns=self.returns.columns
        )

    def cluster_assets(self, n_clusters=3):
        # Group similar assets into clusters using KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        # Calculate portfolio return, volatility, and Sharpe ratio
        weights = np.array(weights).flatten()
        num_assets = len(self.returns.columns)
        if weights.shape[0] != num_assets:
            raise ValueError(
                f"Weights dimension {weights.shape[0]} does not match number of assets {num_assets}"
            )

        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values * 252  # Annualize covariance
        portfolio_return = np.dot(weights, mean_returns) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility(self, target_return):
        # Optimize portfolio to achieve minimum volatility for a target return
        num_assets = len(self.returns.columns)
        constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {
                "type": "eq",
                "fun": lambda weights: np.dot(weights, self.returns.mean().values)
                * 252
                - target_return,
            },
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1.0 / num_assets]
        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            raise ValueError("Optimization did not converge.")
        return result.x

    def generate_efficient_frontier(self, target_returns):
        # Generate efficient frontier curve by optimizing for different target returns
        efficient_portfolios = []
        for ret in target_returns:
            try:
                weights = self.min_volatility(ret)
                portfolio_return, portfolio_volatility, _ = self.portfolio_stats(weights)
                efficient_portfolios.append((portfolio_volatility, portfolio_return))
            except ValueError:
                continue  # Skip target returns that are not achievable
        return np.array(efficient_portfolios)

    def monte_carlo_simulation(self, num_simulations=10000):
        # Perform Monte Carlo simulations to explore possible portfolio outcomes
        num_assets = len(self.returns.columns)
        results = np.zeros((3, num_simulations))
        weights_record = []

        for i in range(num_simulations):
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_stats(
                weights
            )
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
    st.title(
        "Portfolio Optimization with Denoising, Clustering, Backtesting, Efficient Frontier, and Monte Carlo Simulation"
    )

    # User inputs for tickers, dates, risk-free rate, and target return
    tickers = st.text_input(
        "Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):"
    )
    start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
    risk_free_rate = (
        st.number_input(
            "Enter the risk-free rate (in %), typically between 0.5 and 3.0 depending on the economic environment",
            value=2.0,
            step=0.1,
        )
        / 100
    )
    specific_target_return = (
        st.slider(
            "Select a specific target return (in %)",
            min_value=-5.0,
            max_value=30.0,
            value=15.0,
            step=0.1,
        )
        / 100
    )

    if st.button("Optimize Portfolio"):
        try:
            # Validate user inputs and initialize optimizer
            if not tickers:
                st.error("Please enter at least one ticker.")
                st.stop()
            if start_date >= end_date:
                st.error("Start date must be earlier than end date.")
                st.stop()

            ticker_list = [ticker.strip() for ticker in tickers.split(",")]
            optimizer = PortfolioOptimizer(
                ticker_list, start_date, end_date, risk_free_rate
            )
            optimizer.fetch_data()

            # Apply denoising to the returns data
            optimizer.denoise_returns()

            # Cluster assets to identify similar behavior
            clusters = optimizer.cluster_assets(n_clusters=3)
            st.subheader("Asset Clusters")
            cluster_df = pd.DataFrame(
                {"Asset": optimizer.returns.columns, "Cluster": clusters}
            )
            st.write(cluster_df)

            # Generate the efficient frontier curve
            mean_returns = optimizer.returns.mean() * 252
            min_return = mean_returns.min()
            max_return = mean_returns.max()
            target_returns = np.linspace(min_return, max_return, 50)
            efficient_frontier = optimizer.generate_efficient_frontier(target_returns)

            # Perform Monte Carlo simulations
            monte_carlo_results, weights_record = optimizer.monte_carlo_simulation()

            # Plot the efficient frontier and simulation results
            st.subheader("Efficient Frontier with Monte Carlo Simulations")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.scatter(
                monte_carlo_results[0, :],
                monte_carlo_results[1, :],
                c=monte_carlo_results[2, :],
                cmap="viridis",
                alpha=0.5,
                label="Monte Carlo Simulations",
            )
            plt.colorbar(label="Sharpe Ratio")
            plt.plot(
                efficient_frontier[:, 0],
                efficient_frontier[:, 1],
                label="Efficient Frontier",
                color="red",
            )
            plt.xlabel("Risk (Standard Deviation)")
            plt.ylabel("Return")
            plt.title("Efficient Frontier and Simulations")
            plt.legend()
            st.pyplot(fig)

            # Optimize the portfolio for the user's specific target return
            if specific_target_return < min_return or specific_target_return > max_return:
                st.error(
                    f"The target return must be between {min_return*100:.2f}% and {max_return*100:.2f}%."
                )
                st.stop()

            optimal_weights = optimizer.min_volatility(specific_target_return)
            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(
                optimal_weights
            )

            # Display the optimal portfolio allocation
            allocation = pd.DataFrame(
                {
                    "Asset": optimizer.returns.columns,
                    "Weight": optimal_weights,
                }
            )
            st.subheader(
                f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)"
            )
            st.write(allocation)

            # Show portfolio performance metrics
            st.write("Portfolio Performance Metrics:")
            st.write(f"Expected Annual Return: {portfolio_return * 100:.2f}%")
            st.write(f"Annual Volatility (Risk): {portfolio_volatility * 100:.2f}%")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            # Backtest the portfolio and display cumulative returns
            st.subheader("Backtest Portfolio Performance")
            cumulative_returns = optimizer.backtest_portfolio(optimal_weights)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(
                cumulative_returns.index,
                cumulative_returns.values,
                label="Portfolio Cumulative Returns",
            )
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

            # Provide an option to download the portfolio allocation
            st.subheader("Download Portfolio Allocation and Metrics")
            buffer = io.StringIO()
            allocation.to_csv(buffer, index=False)
            st.download_button(
                label="Download Portfolio Allocation (CSV)",
                data=buffer.getvalue(),
                file_name="portfolio_allocation.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
