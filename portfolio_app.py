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

st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

languages = {
    'English': 'en',
    '日本語': 'ja'
}

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
        "explanation_lstm": "**Explanation of LSTM Model:**\nLong Short-Term Memory (LSTM) is a type of artificial neural network used in machine learning. It is particularly effective for predicting sequences and time series data, such as stock returns. LSTM models can remember information over long periods, making them suitable for capturing trends and patterns in historical financial data. However, while LSTM can provide valuable insights, it's important to note that predictions are not guarantees and should be used in conjunction with other analysis methods.",
        "success_optimize": "Portfolio optimization completed successfully!"
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
        "explanation_lstm": "**LSTMモデルの説明：**\nLSTMは、時系列データ予測に適したニューラルネットワークであり、過去の傾向を捉えて将来のパターンを推定できます。ただし、これは保証ではなく、他の分析手法と併用することが望まれます。",
        "success_optimize": "ポートフォリオの最適化が正常に完了しました！"
    }
}

def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

def extract_ticker(asset_string):
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02, benchmark_ticker=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_returns = None

    def fetch_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)["Adj Close"]
        missing_tickers = set(self.tickers) - set(data.columns)
        if missing_tickers:
            st.warning(f"The following tickers were not fetched: {', '.join(missing_tickers)}")
            logger.warning(f"Missing tickers: {missing_tickers}")

        data.dropna(axis=1, inplace=True)
        if data.empty:
            raise ValueError("No data fetched. Please check the tickers and date range.")

        self.tickers = list(data.columns)
        self.returns = data.pct_change().dropna()

        # If benchmark is provided, fetch and compute benchmark returns
        if self.benchmark_ticker:
            benchmark_data = yf.download(self.benchmark_ticker, start=self.start_date, end=self.end_date, progress=False)["Adj Close"]
            benchmark_data.dropna(inplace=True)
            if benchmark_data.empty:
                st.warning("No data fetched for benchmark. Beta and Alpha cannot be computed.")
                self.benchmark_ticker = None
            else:
                self.benchmark_returns = benchmark_data.pct_change().dropna()

        return self.tickers

    def portfolio_stats(self, weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        mu = self.returns.mean() * 252
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        portfolio_return = np.dot(weights, mu)
        sharpe = (portfolio_return - self.risk_free_rate) / sigma if sigma != 0 else 0.0

        # Sortino Ratio: Use only negative returns for downside
        downside_returns = self.returns[self.returns < 0].dropna()
        if not downside_returns.empty:
            downside_std = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * 252, weights)))
        else:
            downside_std = 0.0001  # Avoid division by zero
        sortino = (portfolio_return - self.risk_free_rate) / downside_std

        # Calmar Ratio: Annualized return / Max Drawdown (absolute)
        # Compute max drawdown:
        portfolio_cum = (1 + self.returns.dot(weights)).cumprod()
        peak = portfolio_cum.cummax()
        drawdown = (portfolio_cum - peak)/peak
        max_dd = drawdown.min()
        calmar = portfolio_return / abs(max_dd) if max_dd != 0 else 0.0

        # Beta and Alpha (if benchmark provided)
        if self.benchmark_returns is not None:
            # Merge benchmark returns with portfolio returns
            portfolio_ret_series = self.returns.dot(weights)
            merged = pd.concat([portfolio_ret_series, self.benchmark_returns], axis=1).dropna()
            merged.columns = ['portfolio', 'benchmark']
            cov_matrix = merged.cov()
            beta = cov_matrix.loc['portfolio','benchmark'] / cov_matrix.loc['benchmark','benchmark']
            # Alpha = (Portfolio Return - Risk-free) - Beta*(Benchmark Return - Risk-free)
            benchmark_ann_return = self.benchmark_returns.mean()*252
            alpha = (portfolio_return - self.risk_free_rate) - beta*(benchmark_ann_return - self.risk_free_rate)
        else:
            beta = None
            alpha = None

        return {
            'return': portfolio_return,
            'volatility': sigma,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'beta': beta,
            'alpha': alpha,
            'max_drawdown': max_dd
        }

    def value_at_risk(self, weights, confidence_level=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def herfindahl_hirschman_index(self, weights):
        return np.sum(weights**2)

    def sharpe_ratio_objective(self, weights):
        stats = self.portfolio_stats(weights)
        return -stats['sharpe_ratio']

    def optimize_sharpe_ratio(self):
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets)/num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}
        result = minimize(self.sharpe_ratio_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            return initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1},
            {'type': 'eq', 'fun': lambda weights: (self.portfolio_stats(weights)['return']) - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1./num_assets]*num_assets

        result = minimize(lambda w: self.portfolio_stats(w)['volatility'], init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            return np.ones(num_assets)/num_assets

    def prepare_data_for_lstm(self):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(self.returns.values)
        X,y = [], []
        look_back = 60
        for i in range(look_back,len(scaled_data)):
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i])
        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        if not X_train or not y_train:
            raise ValueError("Not enough data to create training samples.")
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        return X_train, y_train, X_test, y_test, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=X_train.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    def predict_future_returns(self, model, scaler, steps=30):
        if len(self.returns) < 60:
            raise ValueError("Not enough data for predictions.")
        last_data = self.returns[-60:].values
        scaled_last_data = scaler.transform(last_data)
        X_test = np.array([scaled_last_data])
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)
        future_returns = predicted[0][:steps] if len(predicted[0])>=steps else predicted[0]
        return future_returns

    def evaluate_model(self, model, scaler, X_test, y_test):
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_inverse = scaler.inverse_transform(y_test)
        mae = mean_absolute_error(y_test_inverse, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions))
        r2 = r2_score(y_test_inverse, predictions)
        return mae, rmse, r2

    def compute_efficient_frontier(self, num_portfolios=5000):
        results = np.zeros((4, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            weights_record.append(weights)
            stats = self.portfolio_stats(weights)
            results[0,i] = stats['volatility']
            results[1,i] = stats['return']
            results[2,i] = stats['sharpe_ratio']
            results[3,i] = self.herfindahl_hirschman_index(weights)
        return results, weights_record

# Additional Analysis: Display metrics conditionally
def display_metrics(metrics, lang):
    df_data = []
    metric_names = {
        "return": "Expected Annual Return (%)",
        "volatility": "Annual Volatility (Risk) (%)",
        "sharpe_ratio": get_translated_text(lang, "sharpe_ratio"),
        "sortino_ratio": get_translated_text(lang, "sortino_ratio"),
        "calmar_ratio": get_translated_text(lang, "calmar_ratio"),
        "beta": get_translated_text(lang, "beta"),
        "alpha": get_translated_text(lang, "alpha"),
        "max_drawdown": get_translated_text(lang, "max_drawdown"),
        "var": get_translated_text(lang, "var"),
        "cvar": get_translated_text(lang, "cvar"),
        "hhi": get_translated_text(lang, "hhi")
    }

    # Filter out metrics that are 0.0 or None (except return, volatility, sharpe)
    for k, v in metrics.items():
        if k in ["return", "volatility"] or (v is not None and v != 0.0):
            display_name = metric_names.get(k, k)
            if k == "return" or k == "volatility":
                display_val = f"{v*100:.2f}%"
            elif k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha", "beta"]:
                display_val = f"{v:.2f}"
            else:
                display_val = f"{v:.2f}"

            df_data.append({"Metric": display_name, "Value": display_val})

    df = pd.DataFrame(df_data)
    st.table(df)

def main():
    # Language Selection
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]

    st.title(get_translated_text(lang, "title"))

    st.sidebar.header(get_translated_text(lang, "user_inputs"))

    universe_options = {
        'Tech Giants': ['AAPL - Apple','MSFT - Microsoft','GOOGL - Alphabet','AMZN - Amazon','META - Meta','TSLA - Tesla','NVDA - NVIDIA'],
        'Finance Leaders': ['JPM - JPMorgan','BAC - Bank of America','WFC - Wells Fargo','C - Citigroup'],
        'Custom': []
    }

    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), list(universe_options.keys()))

    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(get_translated_text(lang, "custom_tickers"), value="")
    else:
        selected_universe_assets = st.sidebar.multiselect(
            get_translated_text(lang, "add_portfolio"),
            universe_options[universe_choice],
            default=[]
        )

    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Add selected assets
    if universe_choice != 'Custom':
        if selected_universe_assets:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [extract_ticker(a) for a in selected_universe_assets]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio']+new_tickers))
    else:
        if custom_tickers:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio']+new_tickers))

    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if st.session_state['my_portfolio']:
        st.sidebar.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.sidebar.write(get_translated_text(lang, "no_assets"))

    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), value=datetime(2024,1,1), max_value=datetime.today())
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), value=datetime.today(), max_value=datetime.today())
    risk_free_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1)/100

    investment_strategy = st.sidebar.radio(
        get_translated_text(lang, "investment_strategy"),
        (get_translated_text(lang, "strategy_risk_free"), get_translated_text(lang, "strategy_profit"))
    )

    if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
        specific_target_return = st.sidebar.slider(get_translated_text(lang, "target_return"), -5.0, 20.0, 5.0, 0.1)/100
    else:
        specific_target_return = None

    # Benchmark input for Beta and Alpha:
    st.sidebar.markdown("**Optional Benchmark for Beta/Alpha:**")
    benchmark_ticker = st.sidebar.text_input("Enter benchmark ticker (e.g. ^GSPC for S&P 500):", value="")


    # Train LSTM
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))
    # Optimize Portfolio
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))
    # Optimize Sharpe
    optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))
    
    st.header(get_translated_text(lang, "portfolio_analysis"))
    
    if train_lstm:
        st.info("Training LSTM model, please wait...")
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        else:
            try:
                optimizer = PortfolioOptimizer(
                    st.session_state['my_portfolio'],
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    risk_free_rate,
                    benchmark_ticker if benchmark_ticker else None
                )
                optimizer.fetch_data()
                X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X_train, y_train, epochs=10, batch_size=32)
                mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)
    
                st.success(get_translated_text(lang, "success_lstm"))
                eval_metrics = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2
                }
                st.table(pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value']).style.format("{:.4f}"))
    
                # Provide interpretation of the metrics
                st.markdown("""
                **Interpretation:**
                - **MAE & RMSE:** Lower values indicate predictions closer to actual values.
                - **R² Score:** Closer to 1.0 indicates more variance explained by the model.
                """)
    
                # Performance Analysis
                if r2 > 0.9 and rmse < 0.01:
                    st.success("**Excellent Performance:** The model’s predictions are very close to the actual values. It explains most of the variance in the data.")
                elif r2 > 0.75 and rmse < 0.05:
                    st.info("**Good Performance:** The model predicts reasonably well. It’s reliable, but further improvements may still yield better accuracy.")
                elif r2 > 0.5:
                    st.warning("**Moderate Performance:** The model captures some patterns, but there’s significant room for improvement. Consider adding more data or tuning the model.")
                else:
                    st.error("**Poor Performance:** The model does not predict well. Consider revisiting the model architecture, feature set, or training parameters.")
    
                future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B')
                pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Returns': future_returns})
    
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(pred_df['Date'], pred_df['Predicted Returns'], color='blue', label='Predicted Returns')
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
                with st.expander(get_translated_text(lang, "more_info_lstm")):
                    st.markdown(get_translated_text(lang, "explanation_lstm"))
    
                # Attempt to auto minimize sidebar after training LSTM
                hide_sidebar = """
                <script>
                var sidebar = parent.document.querySelector('section[aria-label="sidebar"]');
                var button = parent.document.querySelector('button[title="Collapse sidebar"]');
                if (button) {
                    button.click();
                }
                </script>
                """
                st.markdown(hide_sidebar, unsafe_allow_html=True)
    
            except Exception as e:
                st.error(str(e))

    if optimize_portfolio:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                optimizer = PortfolioOptimizer(st.session_state['my_portfolio'], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate, benchmark_ticker if benchmark_ticker else None)
                optimizer.fetch_data()

                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    if specific_target_return is None:
                        st.error("Please select a target return.")
                        st.stop()
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                else:
                    optimal_weights = optimizer.optimize_sharpe_ratio()

                stats = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation = pd.DataFrame({
                    "Asset": optimizer.tickers,
                    "Weight (%)": np.round(optimal_weights*100,2)
                })
                allocation = allocation[allocation['Weight (%)']>0].reset_index(drop=True)
                target_display = round(specific_target_return*100,2) if specific_target_return else "N/A"
                st.subheader(get_translated_text(lang, "allocation_title").format(target=target_display))
                st.dataframe(allocation)

                # Consolidate metrics
                metrics = {
                    "return": stats['return'],
                    "volatility": stats['volatility'],
                    "sharpe_ratio": stats['sharpe_ratio'],
                    "sortino_ratio": stats['sortino_ratio'],
                    "calmar_ratio": stats['calmar_ratio'],
                    "beta": stats['beta'],
                    "alpha": stats['alpha'],
                    "max_drawdown": stats['max_drawdown'],
                    "var": var_95,
                    "cvar": cvar_95,
                    "hhi": hhi
                }

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics(metrics, lang)

                # Portfolio Tracking Over Time
                st.subheader("📈 Portfolio Tracking Over Time")
                port_cum = (1 + optimizer.returns.dot(optimal_weights)).cumprod()
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(port_cum.index, port_cum.values, label="Portfolio Cumulative Returns")
                ax2.set_title("Cumulative Returns Over Time")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Cumulative Return")
                ax2.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig2)

                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5,4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)
                with col2:
                    performance_metrics = {
                        "Return (%)": stats['return']*100,
                        "Volatility (%)": stats['volatility']*100,
                        "Sharpe": stats['sharpe_ratio']
                    }
                    perf_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    fig3, ax3 = plt.subplots(figsize=(5,4))
                    sns.barplot(x=perf_df.index, y='Value', data=perf_df, ax=ax3)
                    ax3.set_title(get_translated_text(lang, "portfolio_metrics"))
                    plt.xticks(rotation=0)
                    for p in ax3.patches:
                        ax3.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
                    st.pyplot(fig3)

                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                corr = optimizer.returns.corr()
                fig4, ax4 = plt.subplots(figsize=(8,6))
                sns.heatmap(corr, annot=True, cmap='Spectral', linewidths=0.3, ax=ax4)
                ax4.set_title(get_translated_text(lang, "correlation_heatmap"))
                st.pyplot(fig4)

                # Compute and plot Efficient Frontier with a wait message
                st.text("Plotting Efficient Frontier curve, please wait...")
                results, weights_record = optimizer.compute_efficient_frontier()
                vol = results[0]
                ret = results[1]
                sr = results[2]

                max_sr_idx = np.argmax(sr)
                max_sr_vol = vol[max_sr_idx]
                max_sr_ret = ret[max_sr_idx]

                fig5, ax5 = plt.subplots(figsize=(10,6))
                scatter = ax5.scatter(vol, ret, c=sr, cmap='viridis', alpha=0.3)
                ax5.scatter(max_sr_vol, max_sr_ret, c='red', marker='*', s=200, label='Max Sharpe Ratio')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax5.set_xlabel('Annual Volatility')
                ax5.set_ylabel('Annual Return')
                ax5.set_title('Efficient Frontier')
                ax5.legend()
                st.pyplot(fig5)

                # Scenario Testing
                st.subheader("🔧 Scenario Testing")
                shock = st.number_input("Apply a return shock to all assets (in %, e.g., -10 for -10%)", value=0.0, step=1.0)
                if st.button("Test Scenario"):
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    # Recalculate stats under scenario
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0
                    st.write(f"Under a {shock}% shock, the annual return is {scenario_annual_ret*100:.2f}% and Sharpe Ratio is {scenario_sharpe:.2f}.")
                
                st.success(get_translated_text(lang, "success_optimize"))

                # Additional Recommendations Section
                st.markdown("### Additional Recommendations:")
                st.markdown("- Consider adding a benchmark ticker (already provided) to calculate meaningful Beta and Alpha.\n- Explore more advanced scenario testing methods.\n- Incorporate historical event stress testing or Monte Carlo simulations.\n- Add a feature to track portfolio performance over user-defined future periods with predicted returns.")

            except Exception as e:
                st.error(str(e))

    if optimize_sharpe:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                optimizer = PortfolioOptimizer(st.session_state['my_portfolio'], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate, benchmark_ticker if benchmark_ticker else None)
                optimizer.fetch_data()
                optimal_weights = optimizer.optimize_sharpe_ratio()

                stats = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation = pd.DataFrame({
                    "Asset": optimizer.tickers,
                    "Weight (%)": np.round(optimal_weights*100,2)
                })
                allocation = allocation[allocation['Weight (%)']>0].reset_index(drop=True)
                st.subheader("🔑 Optimal Portfolio Allocation (Highest Sharpe Ratio)")
                st.dataframe(allocation)

                metrics = {
                    "return": stats['return'],
                    "volatility": stats['volatility'],
                    "sharpe_ratio": stats['sharpe_ratio'],
                    "sortino_ratio": stats['sortino_ratio'],
                    "calmar_ratio": stats['calmar_ratio'],
                    "beta": stats['beta'],
                    "alpha": stats['alpha'],
                    "max_drawdown": stats['max_drawdown'],
                    "var": var_95,
                    "cvar": cvar_95,
                    "hhi": hhi
                }

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics(metrics, lang)

                # Portfolio Tracking Over Time
                st.subheader("📈 Portfolio Tracking Over Time")
                port_cum = (1 + optimizer.returns.dot(optimal_weights)).cumprod()
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(port_cum.index, port_cum.values, label="Portfolio Cumulative Returns")
                ax2.set_title("Cumulative Returns Over Time")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Cumulative Return")
                ax2.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig2)

                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5,4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)
                with col2:
                    performance_metrics = {
                        "Return (%)": stats['return']*100,
                        "Volatility (%)": stats['volatility']*100,
                        "Sharpe": stats['sharpe_ratio']
                    }
                    perf_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    fig3, ax3 = plt.subplots(figsize=(5,4))
                    sns.barplot(x=perf_df.index, y='Value', data=perf_df, ax=ax3)
                    ax3.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax3.patches:
                        ax3.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
                    st.pyplot(fig3)

                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                corr = optimizer.returns.corr()
                fig4, ax4 = plt.subplots(figsize=(8,6))
                sns.heatmap(corr, annot=True, cmap='Spectral', linewidths=0.3, ax=ax4)
                ax4.set_title(get_translated_text(lang, "correlation_heatmap"))
                st.pyplot(fig4)

                # Efficient Frontier
                st.text("Plotting Efficient Frontier curve, please wait...")
                results, weights_record = optimizer.compute_efficient_frontier()
                vol = results[0]
                ret = results[1]
                sr = results[2]
                max_sr_idx = np.argmax(sr)
                max_sr_vol = vol[max_sr_idx]
                max_sr_ret = ret[max_sr_idx]

                fig5, ax5 = plt.subplots(figsize=(10,6))
                scatter = ax5.scatter(vol, ret, c=sr, cmap='viridis', alpha=0.3)
                ax5.scatter(max_sr_vol, max_sr_ret, c='red', marker='*', s=200, label='Max Sharpe Ratio')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax5.set_xlabel('Annual Volatility')
                ax5.set_ylabel('Annual Return')
                ax5.set_title('Efficient Frontier')
                ax5.legend()
                st.pyplot(fig5)

                # Scenario Testing
                st.subheader("🔧 Scenario Testing")
                shock = st.number_input("Apply a return shock to all assets (in %, e.g., -10 for -10%)", value=0.0, step=1.0)
                if st.button("Test Scenario"):
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0
                    st.write(f"Under a {shock}% shock, the annual return is {scenario_annual_ret*100:.2f}% and Sharpe Ratio is {scenario_sharpe:.2f}.")

                st.success(get_translated_text(lang, "success_optimize"))

                # Additional Recommendations
                st.markdown("### Additional Recommendations:")
                st.markdown("- A benchmark has been added for Beta and Alpha calculations. If no benchmark is provided, these metrics are omitted.")
                st.markdown("- For more advanced scenario testing, consider applying different shocks to individual assets or using historical market stress periods.")
                st.markdown("- Portfolio tracking over time is now displayed. Consider adding forward-looking projections using predicted returns for scenario analysis.")

            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
