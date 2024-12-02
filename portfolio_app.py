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
import plotly.express as px

# Black-Litterman Optimizer
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)["Adj Close"]
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
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
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

# Streamlit App
if __name__ == "__main__":
    st.title("Portfolio Optimization with Advanced Features")

    # User Inputs
    tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):")
    start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
    risk_free_rate = st.number_input("Enter the risk-free rate (in %)", value=2.0, step=0.1) / 100
    specific_target_return = st.slider("Select a specific target return (in %)", min_value=-5.0, max_value=30.0, value=15.0, step=0.1) / 100

    # Allow uploading custom datasets
    uploaded_file = st.file_uploader("Upload your dataset (CSV with columns as asset returns)", type="csv")

    if st.button("Optimize Portfolio"):
        try:
            # Validate user inputs
            if not tickers and not uploaded_file:
                st.error("Please enter tickers or upload a dataset.")
                st.stop()
            if start_date >= end_date:
                st.error("Start date must be earlier than end date.")
                st.stop()

            if uploaded_file:
                user_data = pd.read_csv(uploaded_file, index_col=0)
                optimizer = PortfolioOptimizer([], start_date, end_date, risk_free_rate)
                optimizer.returns = user_data
                st.write("Custom data loaded successfully!")
            else:
                ticker_list = [ticker.strip() for ticker in tickers.split(",")]
                optimizer = PortfolioOptimizer(ticker_list, start_date, end_date, risk_free_rate)
                optimizer.fetch_data()

            # Denoise Returns
            optimizer.denoise_returns()

            # Clustering
            clusters = optimizer.cluster_assets(n_clusters=3)
            st.subheader("Asset Clusters")
            cluster_df = pd.DataFrame({'Asset': optimizer.returns.columns, 'Cluster': clusters})
            st.write(cluster_df)

            # Efficient Frontier
            target_returns = np.linspace(optimizer.returns.mean().min() * 252, optimizer.returns.mean().max() * 252, 50)
            efficient_frontier = optimizer.generate_efficient_frontier(target_returns)

            # Monte Carlo Simulations
            monte_carlo_results, weights_record = optimizer.monte_carlo_simulation()

            # Optimize for Specific Target Return
            optimal_weights = optimizer.min_volatility(specific_target_return)
            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)

            # Plot Efficient Frontier with Portfolio
            st.subheader("Efficient Frontier with Monte Carlo Simulations")
            fig = px.scatter(
                x=monte_carlo_results[0, :],
                y=monte_carlo_results[1, :],
                color=monte_carlo_results[2, :],
                labels={'x': 'Risk', 'y': 'Return', 'color': 'Sharpe Ratio'},
                title="Monte Carlo Simulations and Efficient Frontier",
                template="plotly_dark"
            )
            fig.add_scatter(x=[portfolio_volatility], y=[portfolio_return], mode='markers', marker=dict(size=10, color='red'), name='Optimal Portfolio')
            st.plotly_chart(fig)

            # Display Optimal Portfolio
            allocation = pd.DataFrame({
                'Asset': optimizer.returns.columns,
                'Weight': optimal_weights
            })
            st.subheader(f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)")
            st.write(allocation)

            # Risk Contribution
            st.subheader("Risk Contribution Analysis")
            def risk_contribution(weights, cov_matrix):
                total_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_risk = np.dot(cov_matrix, weights) / total_portfolio_risk
                contribution = weights * marginal_risk
                return contribution

            risk_contributions = risk_contribution(optimal_weights, optimizer.returns.cov() * 252)
            risk_df = pd.DataFrame({
                'Asset': optimizer.returns.columns,
                'Risk Contribution': risk_contributions
            })
            st.write(risk_df)

            # Backtesting
            st.subheader("Backtest Portfolio Performance")
            cumulative_returns = optimizer.backtest_portfolio(optimal_weights)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.title('Portfolio Backtesting Performance')
            plt.legend()
            st.pyplot(fig)

            # Download Portfolio Allocation
            st.subheader("Download Portfolio Allocation and Metrics")
            buffer = io.StringIO()
            allocation.to_csv(buffer, index=False)
            st.download_button(
                label="Download Portfolio Allocation (CSV)",
                data=buffer.getvalue(),
                file_name="portfolio_allocation.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
