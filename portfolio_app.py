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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define language options
languages = {
    'English': 'en',
    '日本語': 'ja'
}

# Define language strings without emojis in plot and main titles
translations = {
    'en': {
        "title": "Portfolio Optimization with Advanced Features",
        "user_inputs": "🔧 User Inputs",
        "select_universe": "Select an Asset Universe:",
        "custom_tickers": "Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):",
        "add_portfolio": "Add to My Portfolio",
        "my_portfolio": "📁 My Portfolio",
        "no_assets": "No assets added yet.",
        "optimization_parameters": "📅 Optimization Parameters",
        "start_date": "Start Date",
        "end_date": "End Date",
        "risk_free_rate": "Enter the risk-free rate (in %):",
        "investment_strategy": "Choose your Investment Strategy:",
        "strategy_risk_free": "Risk-free Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "train_lstm": "Train LSTM Model for Future Returns Prediction",
        "more_info_lstm": "ℹ️ More Information on LSTM",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "portfolio_analysis": "🔍 Portfolio Analysis & Optimization Results",
        "success_lstm": "🤖 LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset to your portfolio before training the LSTM model.",
        "error_no_assets_opt": "Please add at least one asset to your portfolio before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "🔑 Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "📊 Portfolio Performance Metrics",
        "visual_analysis": "📊 Visual Analysis",
        "portfolio_composition": "Portfolio Composition",
        "portfolio_metrics": "Portfolio Performance Metrics",
        "correlation_heatmap": "Asset Correlation Heatmap",
        "var": "Value at Risk (VaR)",
        "cvar": "Conditional Value at Risk (CVaR)",
        "max_drawdown": "Maximum Drawdown",
        "hhi": "Herfindahl-Hirschman Index (HHI)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "explanation_var": "**Value at Risk (VaR):** Estimates the maximum potential loss of a portfolio over a specified time frame at a given confidence level.",
        "explanation_cvar": "**Conditional Value at Risk (CVaR):** Measures the expected loss exceeding the VaR, providing insights into tail risk.",
        "explanation_max_drawdown": "**Maximum Drawdown:** Measures the largest peak-to-trough decline in the portfolio value, indicating the worst-case scenario.",
        "explanation_hhi": "**Herfindahl-Hirschman Index (HHI):** A diversification metric that measures the concentration of investments in a portfolio.",
        "explanation_sharpe_ratio": "**Sharpe Ratio:** Measures risk-adjusted returns, indicating how much excess return you receive for the extra volatility endured.",
        "explanation_sortino_ratio": "**Sortino Ratio:** Similar to the Sharpe Ratio but only considers downside volatility, providing a more targeted risk-adjusted return measure.",
        "explanation_calmar_ratio": "**Calmar Ratio:** Compares the portfolio's annualized return to its maximum drawdown, indicating return per unit of risk.",
        "explanation_beta": "**Beta:** Measures the portfolio's volatility relative to a benchmark index (e.g., S&P 500). A beta greater than 1 indicates higher volatility than the benchmark.",
        "explanation_alpha": "**Alpha:** Represents the portfolio's excess return relative to the expected return based on its beta. Positive alpha indicates outperformance.",
        "explanation_lstm": "**Explanation of LSTM Model:**\nLong Short-Term Memory (LSTM) is a type of artificial neural network used in machine learning. It is particularly effective for predicting sequences and time series data, such as stock returns. LSTM models can remember information over long periods, making them suitable for capturing trends and patterns in historical financial data. However, while LSTM can provide valuable insights, it's important to note that predictions are not guarantees and should be used in conjunction with other analysis methods.",
        "feedback_sharpe_good": "Great! A Sharpe Ratio above 1 indicates that your portfolio is generating good returns for the level of risk taken.",
        "feedback_sharpe_average": "Average. A Sharpe Ratio between 0.5 and 1 suggests that your portfolio returns are acceptable for the risk taken.",
        "feedback_sharpe_poor": "Poor. A Sharpe Ratio below 0.5 indicates that your portfolio may not be generating adequate returns for the level of risk taken. Consider diversifying your assets or adjusting your investment strategy.",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_sharpe_button": "**Optimize for Highest Sharpe Ratio:**\nThe Sharpe Ratio measures the performance of your portfolio compared to a risk-free asset, after adjusting for its risk. Optimizing for the highest Sharpe Ratio aims to achieve the best possible return for the level of risk you are willing to take. This helps in constructing a portfolio that maximizes returns while minimizing unnecessary risk."
    },
    'ja': {
        "title": "高度な機能を備えたポートフォリオ最適化アプリ",
        "user_inputs": "🔧 ユーザー入力",
        "select_universe": "資産ユニバースを選択してください：",
        "custom_tickers": "株式ティッカーをカンマで区切って入力してください（例：AAPL, MSFT, TSLA）：",
        "add_portfolio": "マイポートフォリオに追加",
        "my_portfolio": "📁 マイポートフォリオ",
        "no_assets": "まだ資産が追加されていません。",
        "optimization_parameters": "📅 最適化パラメータ",
        "start_date": "開始日",
        "end_date": "終了日",
        "risk_free_rate": "無リスク金利を入力してください（%）：",
        "investment_strategy": "投資戦略を選択してください：",
        "strategy_risk_free": "リスクフリー投資",
        "strategy_profit": "利益重視投資",
        "target_return": "特定の目標リターンを選択してください（%）",
        "train_lstm": "将来のリターン予測のためにLSTMモデルを訓練",
        "more_info_lstm": "ℹ️ LSTMに関する詳細情報",
        "optimize_portfolio": "ポートフォリオを最適化",
        "optimize_sharpe": "シャープレシオ最大化のために最適化",
        "portfolio_analysis": "🔍 ポートフォリオ分析と最適化結果",
        "success_lstm": "🤖 LSTMモデルが正常に訓練されました！",
        "error_no_assets_lstm": "LSTMモデルを訓練する前に、ポートフォリオに少なくとも1つの資産を追加してください。",
        "error_no_assets_opt": "最適化する前に、ポートフォリオに少なくとも1つの資産を追加してください。",
        "error_date": "開始日は終了日より前でなければなりません。",
        "allocation_title": "🔑 最適なポートフォリオ配分（目標リターン：{target}%)",
        "performance_metrics": "📊 ポートフォリオのパフォーマンス指標",
        "visual_analysis": "📊 視覚的分析",
        "portfolio_composition": "ポートフォリオ構成",
        "portfolio_metrics": "ポートフォリオのパフォーマンス指標",
        "correlation_heatmap": "資産相関ヒートマップ",
        "var": "リスク価値 (VaR)",
        "cvar": "条件付きリスク価値 (CVaR)",
        "max_drawdown": "最大ドローダウン",
        "hhi": "ハーフィンダール・ハーシュマン指数 (HHI)",
        "sharpe_ratio": "シャープレシオ",
        "sortino_ratio": "ソルティーノレシオ",
        "calmar_ratio": "カルマーレシオ",
        "beta": "ベータ",
        "alpha": "アルファ",
        "explanation_var": "**リスク価値 (VaR):** 指定された信頼水準で、特定の期間内にポートフォリオが被る最大損失を推定します。",
        "explanation_cvar": "**条件付きリスク価値 (CVaR):** VaRを超える損失の期待値を測定し、テールリスクに関する洞察を提供します。",
        "explanation_max_drawdown": "**最大ドローダウン:** ポートフォリオの価値がピークから谷に下落する最大幅を測定し、最悪のシナリオを示します。",
        "explanation_hhi": "**ハーフィンダール・ハーシュマン指数 (HHI):** ポートフォリオ内の投資集中度を測定する多様化指標です。",
        "explanation_sharpe_ratio": "**シャープレシオ:** リスク調整後のリターンを測定し、追加のボラティリティに対してどれだけの超過リターンを受け取っているかを示します。",
        "explanation_sortino_ratio": "**ソルティーノレシオ:** シャープレシオと似ていますが、下方のボラティリティのみを考慮し、よりターゲットを絞ったリスク調整後のリターンを提供します。",
        "explanation_calmar_ratio": "**カルマーレシオ:** ポートフォリオの年率リターンを最大ドローダウンと比較し、リスク単位あたりのリターンを示します。",
        "explanation_beta": "**ベータ:** ポートフォリオのベンチマーク指数（例：S&P 500）に対するボラティリティを測定します。ベータが1を超えると、ベンチマークよりも高いボラティリティを示します。",
        "explanation_alpha": "**アルファ:** ポートフォリオのベータに基づく期待リターンに対する超過リターンを表します。プラスのアルファはアウトパフォームを示します。",
        "explanation_lstm": "**LSTMモデルの説明：**\n長短期記憶（LSTM）は、機械学習で使用される人工ニューラルネットワークの一種です。特に株式リターンのようなシーケンスデータや時系列データの予測に効果的です。LSTMモデルは長期間にわたる情報を保持できるため、過去の金融データのトレンドやパターンを捉えるのに適しています。ただし、LSTMは過去のパターンに基づいて予測を行うため、市場のボラティリティによって予測が不確実になることを理解することが重要です。したがって、LSTMの予測は他の分析手法と組み合わせて使用することをお勧めします。",
        "feedback_sharpe_good": "素晴らしいです！シャープレシオが1以上であれば、リスクに対して良好なリターンを生成していることを示します。",
        "feedback_sharpe_average": "平均的です。シャープレシオが0.5〜1の間であれば、リスクに対して許容範囲内のリターンを示しています。",
        "feedback_sharpe_poor": "低いです。シャープレシオが0.5未満であれば、リスクに対して十分なリターンを生成していない可能性があります。資産の多様化や投資戦略の調整を検討してください。",
        "success_optimize": "ポートフォリオの最適化が正常に完了しました！",
        "explanation_sharpe_button": "**シャープレシオ最大化のために最適化：**\nシャープレシオは、リスクフリー資産と比較してポートフォリオのパフォーマンスを測定し、リスクを調整したリターンを評価します。シャープレシオを最大化することで、リスクに見合った最高のリターンを達成するポートフォリオを構築することを目指します。これにより、リスクを最小限に抑えつつ、リターンを最大化するバランスの取れた投資戦略を実現できます。"
    }
}

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
        self.benchmark_returns = None  # For Beta and Alpha

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

    def fetch_benchmark_data(self, benchmark_ticker='^GSPC'):
        """
        Fetch benchmark data for Beta and Alpha calculations.
        """
        logger.info(f"Fetching benchmark data for ticker: {benchmark_ticker}")
        data = yf.download(
            benchmark_ticker, start=self.start_date, end=self.end_date, progress=False
        )["Adj Close"]

        if data.empty:
            logger.warning(f"No data fetched for benchmark ticker: {benchmark_ticker}")
            self.benchmark_returns = None
        else:
            self.benchmark_returns = data.pct_change().dropna()
            logger.info(f"Fetched benchmark returns for {benchmark_ticker}")

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

    def sortino_ratio(self, weights, target=0):
        """
        Calculate Sortino Ratio for the portfolio.
        """
        weights = np.array(weights)
        portfolio_return = self.returns.dot(weights)
        downside_returns = portfolio_return[portfolio_return < target]
        expected_return = np.mean(portfolio_return) * 252
        downside_std = np.std(downside_returns) * np.sqrt(252)
        sortino = (expected_return - self.risk_free_rate) / downside_std if downside_std != 0 else np.nan
        return sortino

    def calmar_ratio(self, weights):
        """
        Calculate Calmar Ratio for the portfolio.
        """
        portfolio_return, _, _ = self.portfolio_stats(weights)
        max_dd = self.maximum_drawdown(weights)
        calmar = (portfolio_return) / abs(max_dd) if max_dd != 0 else np.nan
        return calmar

    def beta_alpha(self, weights, benchmark='^GSPC'):
        """
        Calculate Beta and Alpha for the portfolio relative to a benchmark.
        """
        if self.benchmark_returns is None:
            self.fetch_benchmark_data(benchmark)
        
        if self.benchmark_returns is None:
            return np.nan, np.nan  # Cannot calculate without benchmark data
        
        weights = np.array(weights)
        portfolio_return = self.returns.dot(weights)
        benchmark_return = self.benchmark_returns.reindex(portfolio_return.index).dropna()
        portfolio_return = portfolio_return.loc[benchmark_return.index]

        covariance = np.cov(portfolio_return, benchmark_return)[0][1]
        benchmark_variance = np.var(benchmark_return)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        alpha = (portfolio_return.mean() * 252) - (self.risk_free_rate + beta * (benchmark_return.mean() * 252 - self.risk_free_rate))
        return beta, alpha

    def value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        """
        Calculate Maximum Drawdown for the portfolio.
        """
        portfolio_returns = self.returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def herfindahl_hirschman_index(self, weights):
        """
        Calculate Herfindahl-Hirschman Index (HHI) for the portfolio.
        """
        return np.sum(weights ** 2)

    def sharpe_ratio_objective(self, weights):
        """
        Objective function to maximize Sharpe Ratio.
        """
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe  # Negative because we minimize

    def optimize_sharpe_ratio(self):
        """
        Optimize portfolio to maximize Sharpe Ratio.
        """
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(
            self.sharpe_ratio_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for Sharpe Ratio successfully.")
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return initial_weights  # Fallback to equal weights

    def optimize_sortino_ratio(self):
        """
        Optimize portfolio to maximize Sortino Ratio.
        """
        def sortino_objective(weights):
            return -self.sortino_ratio(weights)
        
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(
            sortino_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for Sortino Ratio successfully.")
            return result.x
        else:
            logger.warning(f"Sortino optimization failed: {result.message}")
            return initial_weights  # Fallback to equal weights

    def calmar_ratio_objective(self, weights):
        """
        Objective function to maximize Calmar Ratio.
        """
        return -self.calmar_ratio(weights)

    def optimize_calmar_ratio(self):
        """
        Optimize portfolio to maximize Calmar Ratio.
        """
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        result = minimize(
            self.calmar_ratio_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for Calmar Ratio successfully.")
            return result.x
        else:
            logger.warning(f"Calmar optimization failed: {result.message}")
            return initial_weights  # Fallback to equal weights

    def beta_alpha_objective(self, weights, target_alpha=0.0):
        """
        Objective function to maximize Alpha while controlling Beta.
        """
        beta, alpha = self.beta_alpha(weights)
        return -alpha  # Negative because we minimize

    def optimize_alpha(self, target_beta=1.0):
        """
        Optimize portfolio to maximize Alpha with a target Beta.
        """
        def objective(weights):
            beta, alpha = self.beta_alpha(weights)
            return -alpha  # Negative because we minimize

        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.beta_alpha(x)[0] - target_beta}
        ]

        result = minimize(
            objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        if result.success:
            logger.info("Optimized portfolio for Alpha successfully.")
            return result.x
        else:
            logger.warning(f"Alpha optimization failed: {result.message}")
            return initial_weights  # Fallback to equal weights

    def prepare_data_for_lstm(self):
        """
        Prepare data for LSTM model.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.returns.values)
        
        X, y = [], []
        look_back = 60  # Look-back period (e.g., 60 days)
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i])
        
        # Split into training and testing sets (e.g., 80% train, 20% test)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    
        if not X_train or not y_train:
            raise ValueError("Not enough data to create training samples. Please adjust the date range or add more data.")
    
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        return X_train, y_train, X_test, y_test, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        # Set random seed for reproducibility
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        """
        Train LSTM model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=X_train.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    def predict_future_returns(self, model, scaler, steps=30):
        """
        Predict future returns using the LSTM model.
        """
        if len(self.returns) < 60:
            raise ValueError("Not enough data to make predictions. Ensure there are at least 60 days of returns data.")

        last_data = self.returns[-60:].values
        scaled_last_data = scaler.transform(last_data)

        X_test = []
        X_test.append(scaled_last_data)
        X_test = np.array(X_test)
        
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)
        
        # Ensure the length matches the number of future steps requested
        future_returns = predicted[0][:steps] if len(predicted[0]) >= steps else predicted[0]
        return future_returns

    def evaluate_model(self, model, scaler, X_test, y_test):
        """
        Evaluate the LSTM model using MAE, RMSE, and R-squared metrics.
        """
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_inverse = scaler.inverse_transform(y_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_inverse, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions))
        r2 = r2_score(y_test_inverse, predictions)

        return mae, rmse, r2

    def compute_efficient_frontier(self, num_portfolios=10000):
        """
        Compute the Efficient Frontier by generating random portfolios.
        """
        results = np.zeros((5, num_portfolios))  # Added Sortino and Calmar
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            weights_record.append(weights)
            portfolio_return, portfolio_volatility, sharpe = self.portfolio_stats(weights)
            sortino = self.sortino_ratio(weights)
            calmar = self.calmar_ratio(weights)
            hhi = self.herfindahl_hirschman_index(weights)
            results[0,i] = portfolio_volatility
            results[1,i] = portfolio_return
            results[2,i] = sharpe
            results[3,i] = sortino
            results[4,i] = calmar
        return results, weights_record

# Helper Functions
def extract_ticker(asset_string):
    """
    Extract ticker symbol from asset string.
    """
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

def get_translated_text(lang, key):
    """
    Retrieve translated text based on selected language.
    """
    return translations.get(lang, translations['en']).get(key, key)

def analyze_var(var):
    """
    Analyze Value at Risk (VaR).
    """
    if var < -0.05:
        return "High Risk: Your portfolio has a significant potential loss."
    elif -0.05 <= var < -0.02:
        return "Moderate Risk: Your portfolio has a moderate potential loss."
    else:
        return "Low Risk: Your portfolio is relatively safe."

def analyze_cvar(cvar):
    """
    Analyze Conditional Value at Risk (CVaR).
    """
    if cvar < -0.07:
        return "High Tail Risk: Significant losses beyond VaR."
    elif -0.07 <= cvar < -0.04:
        return "Moderate Tail Risk: Moderate losses beyond VaR."
    else:
        return "Low Tail Risk: Minimal losses beyond VaR."

def analyze_max_drawdown(dd):
    """
    Analyze Maximum Drawdown.
    """
    if dd < -0.20:
        return "Severe Drawdown: The portfolio has experienced a major decline."
    elif -0.20 <= dd < -0.10:
        return "Moderate Drawdown: The portfolio has experienced a noticeable decline."
    else:
        return "Minor Drawdown: The portfolio has maintained stability."

def analyze_hhi(hhi):
    """
    Analyze Herfindahl-Hirschman Index (HHI).
    """
    if hhi > 0.6:
        return "High Concentration: Portfolio lacks diversification."
    elif 0.3 < hhi <= 0.6:
        return "Moderate Concentration: Portfolio has some diversification."
    else:
        return "Good Diversification: Portfolio is well-diversified."

def analyze_sharpe(sharpe):
    """
    Analyze Sharpe Ratio.
    """
    if sharpe > 2:
        return "Excellent Sharpe Ratio: High risk-adjusted returns."
    elif 1 < sharpe <= 2:
        return "Good Sharpe Ratio: Solid risk-adjusted returns."
    elif 0.5 < sharpe <= 1:
        return "Average Sharpe Ratio: Acceptable risk-adjusted returns."
    else:
        return "Poor Sharpe Ratio: Consider improving risk-adjusted returns."

def analyze_sortino(sortino):
    """
    Analyze Sortino Ratio.
    """
    if sortino > 2:
        return "Excellent Sortino Ratio: High risk-adjusted returns considering downside risk."
    elif 1 < sortino <= 2:
        return "Good Sortino Ratio: Solid risk-adjusted returns considering downside risk."
    elif 0.5 < sortino <= 1:
        return "Average Sortino Ratio: Acceptable risk-adjusted returns considering downside risk."
    else:
        return "Poor Sortino Ratio: Consider improving downside risk management."

def analyze_calmar(calmar):
    """
    Analyze Calmar Ratio.
    """
    if calmar > 0.5:
        return "Excellent Calmar Ratio: High return per unit of drawdown risk."
    elif 0.3 < calmar <= 0.5:
        return "Good Calmar Ratio: Solid return per unit of drawdown risk."
    elif 0.1 < calmar <= 0.3:
        return "Average Calmar Ratio: Acceptable return per unit of drawdown risk."
    else:
        return "Poor Calmar Ratio: Consider strategies to improve return or reduce drawdown."

def analyze_beta(beta):
    """
    Analyze Beta.
    """
    if beta > 1.2:
        return "High Beta: Portfolio is significantly more volatile than the benchmark."
    elif 0.8 < beta <= 1.2:
        return "Moderate Beta: Portfolio volatility is comparable to the benchmark."
    else:
        return "Low Beta: Portfolio is less volatile than the benchmark."

def analyze_alpha(alpha):
    """
    Analyze Alpha.
    """
    if alpha > 0.05:
        return "Positive Alpha: Portfolio is outperforming the benchmark."
    elif -0.05 <= alpha <= 0.05:
        return "Neutral Alpha: Portfolio is performing in line with the benchmark."
    else:
        return "Negative Alpha: Portfolio is underperforming the benchmark."

def display_metrics(metrics, lang):
    """
    Display metrics with larger font, analysis in smaller text, and spacing.
    """
    metric_keys = [
        "var", "cvar", "max_drawdown", "hhi", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "beta", "alpha"
    ]
    analysis_functions = {
        "var": analyze_var,
        "cvar": analyze_cvar,
        "max_drawdown": analyze_max_drawdown,
        "hhi": analyze_hhi,
        "sharpe_ratio": analyze_sharpe,
        "sortino_ratio": analyze_sortino,
        "calmar_ratio": analyze_calmar,
        "beta": analyze_beta,
        "alpha": analyze_alpha
    }

    for key in metric_keys:
        display_key = get_translated_text(lang, key)
        value = metrics.get(key, None)
        if value is not None:
            if key in ["hhi"]:
                display_value = f"{value:.4f}"
            elif key in ["beta"]:
                display_value = f"{value:.2f}"
            elif key in ["alpha"]:
                display_value = f"{value:.4f}"
            elif key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:.2%}"
            # Metric with larger font
            st.markdown(f"<h3>{display_key}: {display_value}</h3>", unsafe_allow_html=True)
            # Explanation
            explanation_key = f"explanation_{key}"
            explanation = translations[lang].get(explanation_key, "")
            st.markdown(explanation)
            # Provide feedback based on the metric
            feedback = analysis_functions[key](value)
            if feedback:
                # Analysis in smaller font
                st.markdown(f"<p style='font-size:12px;'><strong>Analysis:</strong> {feedback}</p>", unsafe_allow_html=True)
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)

# Streamlit App
def main():
    # Language Selection
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]

    # Title
    st.title(get_translated_text(lang, "title"))

    # Sidebar for User Inputs
    st.sidebar.header(get_translated_text(lang, "user_inputs"))

    # Define preset universes
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon', 'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe', 'INTC - Intel', 'CSCO - Cisco'],
        'Finance Leaders': ['JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo', 'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley', 'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab', 'USB - U.S. Bancorp'],
        'Healthcare Majors': ['JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth', 'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott', 'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher', 'BMY - Bristol-Myers Squibb'],
        'Custom': []
    }

    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), options=list(universe_options.keys()), index=0)

    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(
            get_translated_text(lang, "custom_tickers"),
            value=""
        )
    else:
        selected_universe_assets = st.sidebar.multiselect(
            get_translated_text(lang, "add_portfolio"),
            universe_options[universe_choice],
            default=[]  # No default selection to prevent auto-adding
        )

    # Initialize Session State for Portfolio
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Add Selected Universe Assets to Portfolio
    if universe_choice != 'Custom':
        if selected_universe_assets:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [extract_ticker(asset) for asset in selected_universe_assets]
                # Add only unique tickers
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio") + " " + get_translated_text(lang, "my_portfolio"))
    else:
        # Add Custom Tickers to Portfolio
        if custom_tickers:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
                # Add only unique tickers
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio") + " " + get_translated_text(lang, "my_portfolio"))

    # Display 'My Portfolio' in Sidebar
    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if st.session_state['my_portfolio']:
        st.sidebar.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.sidebar.write(get_translated_text(lang, "no_assets"))

    # Portfolio Optimization Parameters in Sidebar
    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))

    # Date Inputs
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), value=datetime(2024, 1, 1), max_value=datetime.today())
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), value=datetime.today(), max_value=datetime.today())

    # Risk-Free Rate Input
    risk_free_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1) / 100

    # Investment Strategy Options
    investment_strategy = st.sidebar.radio(
        get_translated_text(lang, "investment_strategy"),
        (get_translated_text(lang, "strategy_risk_free"), get_translated_text(lang, "strategy_profit"))
    )

    # Display Target Return Slider only if "Risk-free Investment" is selected
    if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
        specific_target_return = st.sidebar.slider(
            get_translated_text(lang, "target_return"), 
            min_value=-5.0, max_value=20.0, value=5.0, step=0.1
        ) / 100
    else:
        specific_target_return = None  # Not used in Profit-focused Investment

    # Train LSTM Button
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))

    # Optimize Portfolio Button
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))

    # Optimize for Highest Sharpe Ratio Button
    optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))

    # Main Area for Outputs
    st.header(get_translated_text(lang, "portfolio_analysis"))

    # Train LSTM Model Section
    if train_lstm:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                optimizer.fetch_data()

                # Prepare data for LSTM
                X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X_train, y_train, epochs=10, batch_size=32)
                mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)

                st.success(get_translated_text(lang, "success_lstm"))

                # Display Evaluation Metrics
                st.subheader("LSTM Model Evaluation Metrics")
                eval_metrics = {
                    "Mean Absolute Error (MAE)": mae,
                    "Root Mean Squared Error (RMSE)": rmse,
                    "R-squared (R²)": r2
                }
                eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value'])
                st.table(eval_df.style.format({"Value": "{:.4f}"}))

                # Predict future returns for the next 30 days
                future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B').to_pydatetime().tolist()  # 'B' for business days

                # Create a DataFrame for plotting
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Returns': future_returns
                })

                # Plot future predictions
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prediction_df['Date'], prediction_df['Predicted Returns'], label="Predicted Returns", color='blue')
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Returns")
                ax.set_title(get_translated_text(lang, "train_lstm"))
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Add LSTM Explanation using Expander
                with st.expander(get_translated_text(lang, "more_info_lstm")):
                    explanation = get_translated_text(lang, "explanation_lstm")
                    st.markdown(explanation)

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An error occurred during LSTM training or prediction.")
                st.error(f"{e}")

    # Function to Display Optimization Results
    def display_optimization_results(optimizer, optimal_weights, lang, target_return=None, strategy="Sharpe Ratio"):
        portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
        sortino_ratio = optimizer.sortino_ratio(optimal_weights)
        calmar_ratio = optimizer.calmar_ratio(optimal_weights)
        beta, alpha = optimizer.beta_alpha(optimal_weights)
        var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
        cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
        max_dd = optimizer.maximum_drawdown(optimal_weights)
        hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

        allocation = pd.DataFrame({
            "Asset": optimizer.tickers,
            "Weight (%)": np.round(optimal_weights * 100, 2)
        })
        allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)

        # Display Allocation
        if strategy == "Sharpe Ratio":
            allocation_title = get_translated_text(lang, "allocation_title").format(target=round(target_return*100, 2) if target_return else "N/A")
            st.markdown(f"<h3>{allocation_title}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3>🔑 Optimal Portfolio Allocation ({strategy})</h3>", unsafe_allow_html=True)
        st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))

        # Display Performance Metrics
        st.markdown(f"<h3>{get_translated_text(lang, 'performance_metrics')}</h3>", unsafe_allow_html=True)
        metrics = {
            "var": var_95,
            "cvar": cvar_95,
            "max_drawdown": max_dd,
            "hhi": hhi,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "beta": beta,
            "alpha": alpha
        }

        display_metrics(metrics, lang)

        # Display Visuals
        st.markdown(f"<h3>{get_translated_text(lang, 'visual_analysis')}</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # Pie Chart for Allocation
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title(get_translated_text(lang, "portfolio_composition"))
            st.pyplot(fig1)

        with col2:
            # Bar Chart for Performance Metrics
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            performance_metrics = {
                "Expected Annual Return (%)": portfolio_return * 100,
                "Annual Volatility\n(Risk) (%)": portfolio_volatility * 100,
                "Sharpe Ratio": sharpe_ratio,
                "Sortino Ratio": sortino_ratio,
                "Calmar Ratio": calmar_ratio
            }
            metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
            sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
            ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
            for p in ax2.patches:
                ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10)
            plt.xticks(rotation=0, ha='center')  # Adjust rotation if needed
            plt.tight_layout()
            st.pyplot(fig2)

        # Correlation Heatmap
        st.markdown(f"<h3>{get_translated_text(lang, 'correlation_heatmap')}</h3>", unsafe_allow_html=True)
        correlation_matrix = optimizer.returns.corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
        plt.title(get_translated_text(lang, "correlation_heatmap"))
        plt.tight_layout()
        st.pyplot(fig3)

        # Additional Visual: Cumulative Returns
        st.markdown(f"<h3>Cumulative Returns</h3>", unsafe_allow_html=True)
        cumulative_returns = (1 + optimizer.returns.dot(optimal_weights)).cumprod()
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(cumulative_returns.index, cumulative_returns.values, label="Portfolio Cumulative Returns", color='green')
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Cumulative Returns")
        ax4.set_title("Cumulative Returns Over Time")
        ax4.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)

    # Optimize Portfolio Section
    if optimize_portfolio:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                # Fetch data and update tickers in case some are dropped
                updated_tickers = optimizer.fetch_data()

                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    # Optimize for minimum volatility
                    if specific_target_return is None:
                        st.error("Please select a target return for Risk-free Investment strategy.")
                        st.stop()
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                    details = "Details: You selected a 'Risk-free Investment' strategy, aiming for minimal risk exposure while attempting to achieve the specified target return."
                else:
                    # Optimize for Sharpe Ratio
                    optimal_weights = optimizer.optimize_sharpe_ratio()
                    details = "Details: You selected a 'Profit-focused Investment' strategy, aiming for maximum potential returns with an acceptance of higher risk."

                st.markdown(f"<p><strong>{details}</strong></p>", unsafe_allow_html=True)

                display_optimization_results(optimizer, optimal_weights, lang, specific_target_return, strategy="Sharpe Ratio")

                st.success(get_translated_text(lang, "success_optimize"))

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An unexpected error occurred during optimization.")
                st.error(f"{e}")

    # Optimize for Highest Sharpe Ratio Section
    if optimize_sharpe:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                # Fetch data and update tickers in case some are dropped
                updated_tickers = optimizer.fetch_data()

                # Optimize for Highest Sharpe Ratio
                optimal_weights = optimizer.optimize_sharpe_ratio()
                display_optimization_results(optimizer, optimal_weights, lang, strategy="Sharpe Ratio")

                st.success(get_translated_text(lang, "success_optimize"))

                # Provide explanation
                st.markdown(get_translated_text(lang, "explanation_sharpe_button"))

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An unexpected error occurred during Sharpe Ratio optimization.")
                st.error(f"{e}")

if __name__ == "__main__":
    main()
