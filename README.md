# portfolio_optimizer_app
A web app for portfolio optimization, featuring efficient frontier visualization, Monte Carlo simulations, and performance metrics like expected return, risk, and Sharpe ratio. Users can input stock tickers, set timeframes, and optimize portfolios for specific target returns with downloadable allocations.

Key Features
1. Asset Clustering
Groups assets into clusters using K-Means clustering based on their historical returns.
2. Historical Performance of Assets
Displays the cumulative historical performance of individual assets using a time-series plot.
3. Portfolio Optimization
Optimizes portfolio weights to minimize risk for a given target return using mean-variance optimization.
Supports Efficient Frontier generation.
4. Monte Carlo Simulations
Generates random portfolios to visualize their risk-return profiles and Sharpe ratios on a scatter plot.
Provides insights into the range of possible portfolio outcomes.
5. Denoising Returns
Uses Principal Component Analysis (PCA) to denoise asset returns for more robust portfolio optimization.
6. Portfolio Metrics
Calculates and displays:
Expected Annual Return
Annual Volatility (Risk)
Sharpe Ratio
7. Efficient Frontier Visualization
Dynamically visualizes the Efficient Frontier and Monte Carlo simulations using Plotly.
Marks the selected optimal portfolio on the frontier with an "X."
8. Risk Contribution Analysis
Displays the contribution of each asset to the portfolio's overall risk.
9. Sector Allocation
Fetches and displays the sector or industry allocation of selected tickers.
10. Custom Dataset Upload
Allows users to upload their own historical return data (CSV) for analysis.
11. Backtesting Portfolio Performance
Visualizes the cumulative performance of the optimized portfolio using historical data.
12. Rebalancing Recommendations
Provides recommendations for rebalancing portfolio weights based on current asset prices.
13. Sensitivity Analysis
Allows users to vary the risk-free rate and observe its impact on portfolio performance (e.g., Sharpe ratio).
14. Portfolio Diversification Score
Computes and displays a diversification score based on the correlation of assets in the portfolio.
15. Dynamic Visualizations
Uses Plotly for interactive scatter plots and comparisons, enhancing the user experience.
16. Alerts and Notifications
Provides alerts for:
High Sharpe Ratio (indicating strong performance).
Low Sharpe Ratio (suggesting possible adjustments).
17. Downloadable Portfolio Allocation
Allows users to download the optimized portfolio allocation as a CSV file.
18. User-Friendly Input Options
Inputs for:
Stock tickers (e.g., AAPL, TSLA, MSFT).
Custom datasets via file upload.
Target return slider for efficient frontier exploration.
Risk-free rate adjustment.
