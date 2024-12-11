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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- LANGUAGE & TRANSLATIONS -----------------
languages = {
    'English': 'en',
    '日本語': 'ja'
}

# Comprehensive translations and explanations in both English and Japanese.
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
        "explanation_lstm": "**Explanation of LSTM Model:**\nLSTM can find patterns in time series data. Predictions are not guarantees. Use with other analysis methods.",
        "success_optimize": "Portfolio optimization completed successfully!",
        "benchmark_select": "Select a Benchmark:",
        "scenario_testing": "🔧 Scenario Testing",
        "scenario_input": "Apply a return shock to all assets (in %, e.g., -10 for -10%)",
        "scenario_button": "Test Scenario",
        "scenario_calculating": "Calculating scenario testing...",
        "train_lstm_info": "*Train LSTM:* Configure and train an LSTM to predict future returns.",
        "optimize_portfolio_info": "*Optimize Portfolio:* Optimize allocations based on your chosen strategy.",
        "optimize_sharpe_info": "*Optimize for Highest Sharpe Ratio:* Find allocations that maximize risk-adjusted return."
    },
    'ja': {
        "title": "高度な機能を備えたポートフォリオ最適化アプリ",
        "user_inputs": "🔧 ユーザー入力",
        "select_universe": "資産ユニバースを選択してください：",
        "custom_tickers": "カンマ区切りで株式ティッカーを入力してください（例：AAPL, MSFT, TSLA）：",
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
        "train_lstm": "LSTMモデルで将来リターンを予測",
        "more_info_lstm": "ℹ️ LSTMに関する詳細情報",
        "optimize_portfolio": "ポートフォリオを最適化",
        "optimize_sharpe": "最高のシャープレシオを目指して最適化",
        "portfolio_analysis": "🔍 ポートフォリオ分析＆最適化結果",
        "success_lstm": "🤖 LSTMモデルの訓練が完了しました！",
        "error_no_assets_lstm": "LSTMモデルを訓練する前に、ポートフォリオに少なくとも1つの資産を追加してください。",
        "error_no_assets_opt": "最適化する前に、ポートフォリオに少なくとも1つの資産を追加してください。",
        "error_date": "開始日は終了日より前でなければなりません。",
        "allocation_title": "🔑 最適なポートフォリオ配分（目標リターン：{target}%）",
        "performance_metrics": "📊 ポートフォリオパフォーマンス指標",
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
        "explanation_lstm": "**LSTMモデルの説明：**\nLSTMは時系列データのパターンを捉えることができますが、予測は保証ではありません。他の分析方法と組み合わせてご活用ください。",
        "success_optimize": "ポートフォリオ最適化が正常に完了しました！",
        "benchmark_select": "ベンチマークを選択してください：",
        "scenario_testing": "🔧 シナリオテスト",
        "scenario_input": "全資産にリターンショックを適用（%で入力、例：-10で-10%）",
        "scenario_button": "シナリオテスト実行",
        "scenario_calculating": "シナリオテストを計算中...",
        "train_lstm_info": "*LSTM訓練：* 将来リターンを予測するためにLSTMモデルを設定し、訓練できます。",
        "optimize_portfolio_info": "*ポートフォリオ最適化：* 選択した戦略に基づいて資産配分を最適化します。",
        "optimize_sharpe_info": "*最高シャープレシオ最適化：* リスク調整後リターンが最大になる配分を見つけます。"
    }
}

def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

def analyze_metric(lang, metric_name, value):
    # Simple heuristic-based analysis. Adjust as needed.
    # We'll provide Japanese and English versions.
    if lang == 'ja':
        # Japanese analysis
        if metric_name in ["シャープレシオ", "ソルティーノレシオ", "カルマーレシオ"]:
            if value > 1.0:
                return "良好"
            elif value > 0.5:
                return "平均的"
            else:
                return "不十分"
        elif metric_name == "ベータ":
            if value > 1.0:
                return "ボラティリティ高め"
            elif value < 1.0:
                return "ボラティリティ低め"
            else:
                return "ベンチマーク並み"
        elif metric_name == "アルファ":
            if value > 0.0:
                return "ベンチマーク上回り"
            elif value == 0.0:
                return "ベンチマーク並み"
            else:
                return "ベンチマーク下回り"
        elif metric_name in ["リスク価値 (VaR)", "条件付きリスク価値 (CVaR)"]:
            if value < -0.05:
                return "リスク高め"
            elif value < -0.02:
                return "中程度のリスク"
            else:
                return "リスク低め"
        elif metric_name == "最大ドローダウン":
            if value < -0.20:
                return "深刻な下落"
            elif value < -0.10:
                return "中程度の下落"
            else:
                return "軽微な下落"
        elif metric_name == "ハーフィンダール・ハーシュマン指数 (HHI)":
            if value > 0.6:
                return "集中度高い(多様化不足)"
            elif value > 0.3:
                return "中程度の多様化"
            else:
                return "良好な多様化"
        else:
            # For return or volatility or others
            return ""
    else:
        # English analysis
        if metric_name in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
            if value > 1.0:
                return "Good"
            elif value > 0.5:
                return "Average"
            else:
                return "Poor"
        elif metric_name == "Beta":
            if value > 1.0:
                return "High Volatility"
            elif value < 1.0:
                return "Low Volatility"
            else:
                return "Benchmark-like"
        elif metric_name == "Alpha":
            if value > 0.0:
                return "Outperforming Benchmark"
            elif value == 0.0:
                return "Performing at Benchmark"
            else:
                return "Underperforming Benchmark"
        elif metric_name in ["Value at Risk (VaR)", "Conditional Value at Risk (CVaR)"]:
            if value < -0.05:
                return "High Risk"
            elif value < -0.02:
                return "Moderate Risk"
            else:
                return "Low Risk"
        elif metric_name == "Maximum Drawdown":
            if value < -0.20:
                return "Severe Drawdown"
            elif value < -0.10:
                return "Moderate Drawdown"
            else:
                return "Minor Drawdown"
        elif metric_name == "Herfindahl-Hirschman Index (HHI)":
            if value > 0.6:
                return "Highly Concentrated"
            elif value > 0.3:
                return "Moderately Diversified"
            else:
                return "Well Diversified"
        else:
            return ""

def display_metrics(metrics, lang):
    df_data = []
    metric_names = {
        'en': {
            "return": "Expected Annual Return (%)",
            "volatility": "Annual Volatility (Risk) (%)",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "calmar_ratio": "Calmar Ratio",
            "beta": "Beta",
            "alpha": "Alpha",
            "max_drawdown": "Maximum Drawdown",
            "var": "Value at Risk (VaR)",
            "cvar": "Conditional Value at Risk (CVaR)",
            "hhi": "Herfindahl-Hirschman Index (HHI)"
        },
        'ja': {
            "return": "期待年間リターン (%)",
            "volatility": "年間ボラティリティ（リスク）(%)",
            "sharpe_ratio": "シャープレシオ",
            "sortino_ratio": "ソルティーノレシオ",
            "calmar_ratio": "カルマーレシオ",
            "beta": "ベータ",
            "alpha": "アルファ",
            "max_drawdown": "最大ドローダウン",
            "var": "リスク価値 (VaR)",
            "cvar": "条件付きリスク価値 (CVaR)",
            "hhi": "ハーフィンダール・ハーシュマン指数 (HHI)"
        }
    }

    metric_labels = metric_names[lang]

    for k, v in metrics.items():
        if v is not None:
            if (k in ["return", "volatility"]) or (v != 0.0):
                display_name = metric_labels.get(k, k)
                if k in ["return", "volatility"]:
                    display_val = f"{v*100:.2f}%"
                elif k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha", "beta"]:
                    display_val = f"{v:.2f}"
                else:
                    display_val = f"{v:.2f}"
                analysis = analyze_metric(lang, display_name, v)
                df_data.append({"Metric": display_name, "Value": display_val, "Analysis": analysis})

    df = pd.DataFrame(df_data)
    st.table(df.style.set_properties(**{'text-align': 'left'}))

class PortfolioOptimizer:
    # same methods as before (copy from previous code)
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
        weights = np.array(weights) / np.sum(weights)
        mu = self.returns.mean() * 252
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        portfolio_return = np.dot(weights, mu)
        sharpe = (portfolio_return - self.risk_free_rate) / sigma if sigma != 0 else 0.0
        downside_returns = self.returns[self.returns < 0].dropna()
        downside_std = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * 252, weights))) if not downside_returns.empty else 0.0001
        sortino = (portfolio_return - self.risk_free_rate) / downside_std
        portfolio_cum = (1 + self.returns.dot(weights)).cumprod()
        peak = portfolio_cum.cummax()
        drawdown = (portfolio_cum - peak) / peak
        max_dd = drawdown.min()
        calmar = portfolio_return / abs(max_dd) if max_dd != 0 else 0.0
        if self.benchmark_returns is not None:
            portfolio_ret_series = self.returns.dot(weights)
            merged = pd.concat([portfolio_ret_series, self.benchmark_returns], axis=1).dropna()
            merged.columns = ['portfolio', 'benchmark']
            cov_matrix = merged.cov()
            beta = cov_matrix.loc['portfolio','benchmark'] / cov_matrix.loc['benchmark','benchmark']
            benchmark_ann_return = self.benchmark_returns.mean()*252
            alpha = (portfolio_return - self.risk_free_rate) - beta*(benchmark_ann_return - self.risk_free_rate)
        else:
            beta = None
            alpha = None

        return {
            'return': float(portfolio_return),
            'volatility': float(sigma),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'beta': float(beta) if beta is not None else None,
            'alpha': float(alpha) if alpha is not None else None,
            'max_drawdown': float(max_dd)
        }

    def value_at_risk(self, weights, confidence_level=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return float(var)

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        var = self.value_at_risk(weights, confidence_level)
        portfolio_returns = self.returns.dot(weights)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return float(cvar)

    def herfindahl_hirschman_index(self, weights):
        return float(np.sum(weights**2))

    def sharpe_ratio_objective(self, weights):
        stats = self.portfolio_stats(weights)
        return -stats['sharpe_ratio']

    def optimize_sharpe_ratio(self):
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets)/num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}
        result = minimize(self.sharpe_ratio_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'eq', 'fun': lambda w: (self.portfolio_stats(w)['return']) - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1./num_assets]*num_assets
        result = minimize(lambda w: self.portfolio_stats(w)['volatility'], init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else np.ones(num_assets)/num_assets

    def prepare_data_for_lstm(self, look_back=60, scaler_type="MinMaxScaler"):
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=(0,1))
        else:
            scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.returns.values)
        X,y = [], []
        for i in range(look_back,len(scaled_data)):
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i])
        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        if not X_train or not y_train:
            raise ValueError("Not enough data to create training samples.")
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32, lstm_units=50, lstm_layers=2, dropout_rate=0.0):
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        model = tf.keras.Sequential()
        for i in range(lstm_layers):
            return_sequences = True if i < lstm_layers - 1 else False
            model.add(tf.keras.layers.LSTM(units=lstm_units, return_sequences=return_sequences, input_shape=(X_train.shape[1], X_train.shape[2])))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))
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

def main():
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Language Selection
    selected_language = st.sidebar.selectbox("Language / 言語選択:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]

    st.title(get_translated_text(lang, "title"))
    st.header(get_translated_text(lang, "portfolio_analysis"))

    # Asset Universes
    tech_giants = [
        'AAPL - Apple','MSFT - Microsoft','GOOGL - Alphabet','AMZN - Amazon','META - Meta',
        'TSLA - Tesla','NVDA - NVIDIA','ADBE - Adobe','INTC - Intel','CSCO - Cisco'
    ]
    finance_leaders = [
        'JPM - JPMorgan Chase','BAC - Bank of America','WFC - Wells Fargo','C - Citigroup','GS - Goldman Sachs',
        'MS - Morgan Stanley','AXP - American Express','BLK - BlackRock','SCHW - Charles Schwab','USB - U.S. Bancorp'
    ]
    healthcare_majors = [
        'JNJ - Johnson & Johnson','PFE - Pfizer','UNH - UnitedHealth','MRK - Merck','ABBV - AbbVie',
        'ABT - Abbott','TMO - Thermo Fisher','MDT - Medtronic','DHR - Danaher','BMY - Bristol-Myers'
    ]
    broad_market = [
        'SPY - S&P 500 ETF','VOO - Vanguard S&P 500','IVV - iShares Core S&P 500','VTI - Vanguard Total Stock Market',
        'VEA - Vanguard FTSE Dev Markets','VWO - Vanguard FTSE Emerging','QQQ - Invesco QQQ','DIA - SPDR Dow Jones',
        'GLD - SPDR Gold','EFA - iShares MSCI EAFE'
    ]

    universe_options = {
        get_translated_text(lang, "select_universe"): get_translated_text(lang, "select_universe"), # Just a placeholder if needed
        'Tech Giants': tech_giants,
        'Finance Leaders': finance_leaders,
        'Healthcare Majors': healthcare_majors,
        'Broad Market ETFs': broad_market,
        'Custom': []
    }

    st.sidebar.header(get_translated_text(lang, "user_inputs"))
    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), list(universe_options.keys()))

    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(get_translated_text(lang, "custom_tickers"), value="", key="custom_tickers_input")
    else:
        selected_universe_assets = st.sidebar.multiselect(
            get_translated_text(lang, "add_portfolio"),
            universe_options[universe_choice],
            default=[]
        )

    if universe_choice != 'Custom':
        if selected_universe_assets and st.sidebar.button(get_translated_text(lang, "add_portfolio"), key="add_universe"):
            new_tickers = [extract_ticker(a) for a in selected_universe_assets]
            st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio']+new_tickers))
    else:
        if st.sidebar.button(get_translated_text(lang, "add_portfolio"), key="add_custom"):
            if custom_tickers:
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

    strategy_risk_free = get_translated_text(lang, "strategy_risk_free")
    strategy_profit = get_translated_text(lang, "strategy_profit")
    investment_strategy = st.sidebar.radio(
        get_translated_text(lang, "investment_strategy"),
        (strategy_risk_free, strategy_profit)
    )

    if investment_strategy == strategy_risk_free:
        specific_target_return = st.sidebar.slider(get_translated_text(lang, "target_return"), -5.0, 20.0, 5.0, 0.1)/100
    else:
        specific_target_return = None

    st.sidebar.markdown("**" + get_translated_text(lang, "benchmark_select") + "**")
    benchmark_options = {
        'None': None,
        'S&P 500 (^GSPC)': '^GSPC',
        'NASDAQ 100 (^NDX)': '^NDX',
        'Dow Jones (^DJI)': '^DJI',
        'Russell 2000 (^RUT)': '^RUT',
        'S&P 500 ETF (SPY)': 'SPY',
        'Vanguard S&P 500 (VOO)': 'VOO'
    }
    benchmark_choice = st.sidebar.selectbox("", list(benchmark_options.keys()), index=0)
    benchmark_ticker = benchmark_options[benchmark_choice]

    # Buttons with explanations
    train_lstm_btn = st.sidebar.button(get_translated_text(lang, "train_lstm"), key="train_lstm_btn")
    st.sidebar.markdown(get_translated_text(lang, "train_lstm_info"))

    optimize_portfolio_btn = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"), key="optimize_portfolio_btn")
    st.sidebar.markdown(get_translated_text(lang, "optimize_portfolio_info"))

    optimize_sharpe_btn = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"), key="optimize_sharpe_btn")
    st.sidebar.markdown(get_translated_text(lang, "optimize_sharpe_info"))

    # Flags to know if portfolio is optimized
    if 'portfolio_optimized' not in st.session_state:
        st.session_state['portfolio_optimized'] = False
    if 'sharpe_optimized' not in st.session_state:
        st.session_state['sharpe_optimized'] = False

    # LSTM parameter and training logic
    if train_lstm_btn:
        st.info(get_translated_text(lang, "train_lstm") + " ...")
        with st.expander(get_translated_text(lang, "more_info_lstm")):
            st.markdown(get_translated_text(lang, "explanation_lstm"))

        # Explanations for parameters already given above.
        if 'look_back_window' not in st.session_state:
            st.session_state['look_back_window'] = 60
        if 'lstm_units' not in st.session_state:
            st.session_state['lstm_units'] = 50
        if 'lstm_layers' not in st.session_state:
            st.session_state['lstm_layers'] = 2
        if 'dropout_rate' not in st.session_state:
            st.session_state['dropout_rate'] = 0.0
        if 'epochs' not in st.session_state:
            st.session_state['epochs'] = 10
        if 'batch_size' not in st.session_state:
            st.session_state['batch_size'] = 32
        if 'scaler_type' not in st.session_state:
            st.session_state['scaler_type'] = "MinMaxScaler"

        st.session_state['look_back_window'] = st.slider("Look-back Window", 30, 120, st.session_state['look_back_window'], 10)
        st.session_state['lstm_units'] = st.slider("LSTM Units per Layer", 10, 200, st.session_state['lstm_units'], 10)
        st.session_state['lstm_layers'] = st.selectbox("Number of LSTM Layers", [1,2,3], index=[1,2,3].index(st.session_state['lstm_layers']))
        st.session_state['dropout_rate'] = st.slider("Dropout Rate", 0.0, 0.5, st.session_state['dropout_rate'], 0.1)
        st.session_state['epochs'] = st.number_input("Training Epochs", 5, 100, st.session_state['epochs'], 5)
        st.session_state['batch_size'] = st.number_input("Batch Size", 16, 256, st.session_state['batch_size'],16)
        st.session_state['scaler_type'] = st.selectbox("Scaler Type", ["MinMaxScaler", "StandardScaler"], 0 if st.session_state['scaler_type']=="MinMaxScaler" else 1)

        if st.button("Run LSTM Training"):
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
                    X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm(
                        look_back=st.session_state['look_back_window'],
                        scaler_type=st.session_state['scaler_type']
                    )
                    model = optimizer.train_lstm_model(
                        X_train,
                        y_train,
                        epochs=st.session_state['epochs'],
                        batch_size=st.session_state['batch_size'],
                        lstm_units=st.session_state['lstm_units'],
                        lstm_layers=st.session_state['lstm_layers'],
                        dropout_rate=st.session_state['dropout_rate']
                    )

                    mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)
                    st.success(get_translated_text(lang, "success_lstm"))
                    eval_metrics = {"MAE": mae, "RMSE": rmse, "R²": r2}
                    st.table(pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value']).style.format("{:.4f}"))

                    if lang == 'ja':
                        st.markdown("""
                        **解釈:**
                        - MAE・RMSEが低いほど予測が実測値に近いことを示します。
                        - R²が1.0に近いほど、モデルがデータの分散をよく説明しています。
                        """)
                    else:
                        st.markdown("""
                        **Interpretation:**
                        - Lower MAE & RMSE = closer predictions.
                        - R² close to 1.0 = better fit.
                        """)

                    if r2 > 0.9 and rmse < 0.01:
                        st.success("Excellent performance.")
                    elif r2 > 0.75 and rmse < 0.05:
                        st.info("Good performance, consider minor tuning.")
                    elif r2 > 0.5:
                        st.warning("Moderate performance, consider adjustments.")
                    else:
                        st.error("Poor performance.")

                    future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                    future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B')
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Returns': future_returns})

                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(pred_df['Date'], pred_df['Predicted Returns'], color='blue', label='Predicted Returns')
                    ax.set_xlabel("Date" if lang=='en' else "日付")
                    ax.set_ylabel("Predicted Returns" if lang=='en' else "予測リターン")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    if lang=='ja':
                        st.markdown("**グラフ分析:** 予測リターンは時間とともに変動しています。初期の下落、ピークへの上昇、そして再度の下落が見られます。")
                        st.markdown("""
                        **改善可能なパラメータ：**
                        - Look-back Window: 過去日数を変える
                        - LSTM Units/Layers: 複雑性を増減
                        - Dropout: 過学習防止
                        - Epochs: 繰り返し回数調整
                        - Batch Size: 学習更新頻度調整
                        - Scaler Type: MinMaxとStandardを試す
                        """)
                    else:
                        st.markdown("**Graph Analysis:** Predicted returns fluctuate over time.")
                        st.markdown("""
                        **Adjustable Parameters:**
                        - Look-back Window
                        - LSTM Units/Layers
                        - Dropout Rate
                        - Training Epochs
                        - Batch Size
                        - Scaler Type
                        """)

                except Exception as e:
                    st.error(str(e))

    # Optimize Portfolio
    if optimize_portfolio_btn:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
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
                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    if specific_target_return is None:
                        st.error("Please select a target return." if lang=='en' else "目標リターンを選択してください。")
                        st.stop()
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                else:
                    optimal_weights = optimizer.optimize_sharpe_ratio()

                st.session_state['portfolio_optimized'] = True
                st.session_state['sharpe_optimized'] = False

                stats = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation_title = get_translated_text(lang, "allocation_title").format(target=round(specific_target_return*100,2) if specific_target_return else "N/A")
                st.subheader(allocation_title)
                allocation = pd.DataFrame({
                    "Asset": optimizer.tickers,
                    "Weight (%)": np.round(optimal_weights*100,2)
                })
                allocation = allocation[allocation['Weight (%)']>0].reset_index(drop=True)
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

                st.subheader(get_translated_text(lang, "visual_analysis"))
                port_cum = (1 + optimizer.returns.dot(optimal_weights)).cumprod()
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(port_cum.index, port_cum.values, label="Portfolio Cumulative Returns" if lang=='en' else "ポートフォリオ累積リターン")
                ax2.set_title("Cumulative Returns Over Time" if lang=='en' else "累積リターン推移")
                ax2.set_xlabel("Date" if lang=='en' else "日付")
                ax2.set_ylabel("Cumulative Return" if lang=='en' else "累積リターン")
                ax2.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig2)

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

                st.text("Plotting Efficient Frontier curve, please wait..." if lang=='en' else "効率的フロンティア曲線をプロット中、お待ちください...")
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
                ax5.set_xlabel('Annual Volatility' if lang=='en' else '年間ボラティリティ')
                ax5.set_ylabel('Annual Return' if lang=='en' else '年間リターン')
                ax5.set_title('Efficient Frontier' if lang=='en' else '効率的フロンティア')
                ax5.legend()
                st.pyplot(fig5)

                st.subheader(get_translated_text(lang, "scenario_testing"))
                st.markdown(get_translated_text(lang, "scenario_input"))
                shock = st.number_input("", value=0.0, step=1.0, key="shock_input")
                if st.button(get_translated_text(lang, "scenario_button"), key="test_scenario"):
                    st.write(get_translated_text(lang, "scenario_calculating"))
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0

                    # Re-plot cumulative returns with scenario factor in title
                    scenario_port_cum = (1 + shocked_returns.dot(optimal_weights)).cumprod()
                    fig_scenario, ax_scenario = plt.subplots(figsize=(10,4))
                    title_str = f"Cumulative Returns with {shock}% Shock" if lang=='en' else f"{shock}%ショック適用後の累積リターン"
                    ax_scenario.plot(scenario_port_cum.index, scenario_port_cum.values, label="Scenario Cumulative Returns")
                    ax_scenario.set_title(title_str)
                    ax_scenario.set_xlabel("Date" if lang=='en' else "日付")
                    ax_scenario.set_ylabel("Cumulative Return" if lang=='en' else "累積リターン")
                    ax_scenario.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig_scenario)

                    if lang=='en':
                        st.write(f"Under a {shock}% shock, annual return: {scenario_annual_ret*100:.2f}%, Sharpe Ratio: {scenario_sharpe:.2f}")
                    else:
                        st.write(f"{shock}%のショックを適用した場合、年間リターンは{scenario_annual_ret*100:.2f}%、シャープレシオは{scenario_sharpe:.2f}となります。")

                st.success(get_translated_text(lang, "success_optimize"))

    if optimize_sharpe_btn:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
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
                optimal_weights = optimizer.optimize_sharpe_ratio()

                st.session_state['portfolio_optimized'] = False
                st.session_state['sharpe_optimized'] = True

                stats = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                st.subheader("🔑 " + ( "Optimal Portfolio Allocation (Highest Sharpe Ratio)" if lang=='en' else "最高シャープレシオポートフォリオの配分"))
                allocation = pd.DataFrame({
                    "Asset": optimizer.tickers,
                    "Weight (%)": np.round(optimal_weights*100,2)
                })
                allocation = allocation[allocation['Weight (%)']>0].reset_index(drop=True)
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

                st.subheader(get_translated_text(lang, "visual_analysis"))
                port_cum = (1 + optimizer.returns.dot(optimal_weights)).cumprod()
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(port_cum.index, port_cum.values, label="Cumulative Returns" if lang=='en' else "累積リターン")
                ax2.set_title("Cumulative Returns Over Time" if lang=='en' else "累積リターン推移")
                ax2.set_xlabel("Date" if lang=='en' else "日付")
                ax2.set_ylabel("Cumulative Return" if lang=='en' else "累積リターン")
                ax2.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig2)

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

                st.text("Plotting Efficient Frontier curve, please wait..." if lang=='en' else "効率的フロンティア曲線をプロット中...")
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
                ax5.set_xlabel('Annual Volatility' if lang=='en' else '年間ボラティリティ')
                ax5.set_ylabel('Annual Return' if lang=='en' else '年間リターン')
                ax5.set_title('Efficient Frontier' if lang=='en' else '効率的フロンティア')
                ax5.legend()
                st.pyplot(fig5)

                st.subheader(get_translated_text(lang, "scenario_testing"))
                st.markdown(get_translated_text(lang, "scenario_input"))
                shock = st.number_input("", value=0.0, step=1.0, key="shock_input_sharpe")
                if st.button(get_translated_text(lang, "scenario_button"), key="test_scenario_sharpe"):
                    st.write(get_translated_text(lang, "scenario_calculating"))
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0
                    # Replot with scenario
                    scenario_port_cum = (1 + shocked_returns.dot(optimal_weights)).cumprod()
                    fig_scenario, ax_scenario = plt.subplots(figsize=(10,4))
                    title_str = f"Cumulative Returns with {shock}% Shock" if lang=='en' else f"{shock}%ショック適用後の累積リターン"
                    ax_scenario.plot(scenario_port_cum.index, scenario_port_cum.values, label="Scenario Cumulative Returns")
                    ax_scenario.set_title(title_str)
                    ax_scenario.set_xlabel("Date" if lang=='en' else "日付")
                    ax_scenario.set_ylabel("Cumulative Return" if lang=='en' else "累積リターン")
                    ax_scenario.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig_scenario)

                    if lang=='ja':
                        st.write(f"{shock}%ショック下での年間リターン: {scenario_annual_ret*100:.2f}%, シャープレシオ: {scenario_sharpe:.2f}")
                    else:
                        st.write(f"Under a {shock}% shock, annual return: {scenario_annual_ret*100:.2f}%, Sharpe Ratio: {scenario_sharpe:.2f}")

                st.success(get_translated_text(lang, "success_optimize"))

if __name__ == "__main__":
    main()
