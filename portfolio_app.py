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

# Language options and translations (simplified, can be expanded)
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
        "explanation_lstm": "**Explanation of LSTM Model:**\nLSTM can find patterns in time series data. Predictions are not guarantees. Use with other analysis methods.",
        "success_optimize": "Portfolio optimization completed successfully!"
    },
    'ja': {
        # Add Japanese translations if needed
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
            beta = cov_matrix.loc['portfolio', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            benchmark_ann_return = self.benchmark_returns.mean() * 252
            alpha = (portfolio_return - self.risk_free_rate) - beta * (benchmark_ann_return - self.risk_free_rate)
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
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
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
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        return X_train, y_train, X_test, y_test, scaler

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

    for k, v in metrics.items():
        if v is not None:
            if (k in ["return", "volatility"]) or (v != 0.0):
                display_name = metric_names.get(k, k)
                if k in ["return", "volatility"]:
                    display_val = f"{v*100:.2f}%"
                elif k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha", "beta"]:
                    display_val = f"{v:.2f}"
                else:
                    display_val = f"{v:.2f}"
                df_data.append({"Metric": display_name, "Value": display_val})

    df = pd.DataFrame(df_data)
    st.table(df)

def main():
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Language selection
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]

    st.title(get_translated_text(lang, "title"))
    st.header(get_translated_text(lang, "portfolio_analysis"))

    # Define preset universes (10 each)
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
    # Custom (no predefined tickers, user will input)

    universe_options = {
        'Tech Giants': tech_giants,
        'Finance Leaders': finance_leaders,
        'Healthcare Majors': healthcare_majors,
        'Broad Market ETFs': broad_market,
        'Custom': []
    }

    # Sidebar for universe selection
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

    # Add to portfolio
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

    # Optimization parameters
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

    # Benchmark Ticker as Dropdown
    st.sidebar.markdown("**Optional Benchmark for Beta/Alpha:**")
    benchmark_options = {
        'None': None,
        'S&P 500 (^GSPC)': '^GSPC',
        'NASDAQ 100 (^NDX)': '^NDX',
        'Dow Jones (^DJI)': '^DJI',
        'Russell 2000 (^RUT)': '^RUT',
        'S&P 500 ETF (SPY)': 'SPY',
        'Vanguard S&P 500 (VOO)': 'VOO'
    }
    benchmark_choice = st.sidebar.selectbox("Select a Benchmark:", list(benchmark_options.keys()), index=0)
    benchmark_ticker = benchmark_options[benchmark_choice]

    # Buttons for actions with explanations below them
    train_lstm_button = st.sidebar.button(get_translated_text(lang, "train_lstm"), key="train_lstm_btn")
    st.sidebar.markdown("*Train LSTM:* This will allow you to configure and train an LSTM model to predict future returns.")

    optimize_portfolio_button = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"), key="optimize_portfolio_btn")
    st.sidebar.markdown("*Optimize Portfolio:* Based on your selected strategy, this will optimize your asset allocations for minimal volatility or chosen target return.")

    optimize_sharpe_button = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"), key="optimize_sharpe_btn")
    st.sidebar.markdown("*Optimize for Highest Sharpe Ratio:* This finds a portfolio allocation that maximizes risk-adjusted returns.")

    # Main page logic

    # LSTM Parameter Setting & Training
    if train_lstm_button:
        st.info("Adjust LSTM parameters below and then click 'Run LSTM Training'")
        with st.expander("What do these LSTM settings mean?"):
            st.markdown("""
            - **Look-back Window:** How many past days are used as input.
            - **LSTM Units per Layer:** More units = more complexity.
            - **LSTM Layers:** Multiple layers capture more complex patterns.
            - **Dropout Rate:** Helps prevent overfitting.
            - **Training Epochs:** Number of passes over the training data.
            - **Batch Size:** Samples per training update.
            - **Scaler Type:** Normalize data with MinMax or Standard scaling.
            """)

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

        st.session_state['look_back_window'] = st.slider("LSTM Look-back Window (days)", 30, 120, st.session_state['look_back_window'], 10)
        st.session_state['lstm_units'] = st.slider("LSTM Units per layer", 10, 200, st.session_state['lstm_units'], 10)
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

                    st.markdown("""
                    **Interpretation:**
                    - MAE & RMSE: Lower = closer predictions.
                    - R²: Closer to 1.0 = better fit.
                    """)

                    if r2 > 0.9 and rmse < 0.01:
                        st.success("Excellent Performance: Predictions are very close to actual values.")
                    elif r2 > 0.75 and rmse < 0.05:
                        st.info("Good Performance: Reasonably accurate. Consider minor tuning.")
                    elif r2 > 0.5:
                        st.warning("Moderate Performance: Some patterns captured. More tuning needed.")
                    else:
                        st.error("Poor Performance: Model not predicting well.")

                    future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                    future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B')
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Returns': future_returns})

                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(pred_df['Date'], pred_df['Predicted Returns'], color='blue', label='Predicted Returns')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Predicted Returns")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    st.markdown("**Graph Analysis:** The predicted returns show variability over time (initial decline, rise to a peak, then fall).")

                    st.markdown("""
                    **Recommendations (Adjustable Parameters):**
                    - **Look-back Window:** Try different lengths.
                    - **LSTM Units/Layers:** Adjust complexity.
                    - **Dropout Rate:** Add dropout to reduce overfitting.
                    - **Training Epochs:** Increase if underfitting, decrease if overfitting.
                    - **Batch Size:** Adjust for better performance.
                    - **Scaler Type:** Try MinMax or Standard to see which yields better results.
                    """)

                    with st.expander(get_translated_text(lang, "more_info_lstm")):
                        st.markdown(get_translated_text(lang, "explanation_lstm"))

                except Exception as e:
                    st.error(str(e))

    # Optimize Portfolio
    if optimize_portfolio_button:
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

                # Scenario Testing after optimization
                st.subheader("🔧 Scenario Testing")
                st.markdown("Apply a percentage shock to all assets to see how the portfolio might perform under stress.")
                shock = st.number_input("Apply a return shock to all assets (in %, e.g., -10 for -10%)", value=0.0, step=1.0, key="shock_input")
                if st.button("Test Scenario", key="test_scenario"):
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0
                    st.write(f"Under a {shock}% shock, the annual return is {scenario_annual_ret*100:.2f}% and Sharpe Ratio is {scenario_sharpe:.2f}.")

                st.success(get_translated_text(lang, "success_optimize"))

            except Exception as e:
                st.error(str(e))

    if optimize_sharpe_button:
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

                st.subheader("🔧 Scenario Testing")
                st.markdown("Apply a percentage shock to see how the optimized Sharpe portfolio performs under stress.")
                shock = st.number_input("Apply a return shock to all assets (in %, e.g., -10 for -10%)", value=0.0, step=1.0, key="shock_input_sharpe")
                if st.button("Test Scenario", key="test_scenario_sharpe"):
                    shock_factor = 1 + shock/100
                    shocked_returns = optimizer.returns * shock_factor
                    scenario_ret = shocked_returns.dot(optimal_weights)
                    scenario_annual_ret = scenario_ret.mean()*252
                    scenario_vol = scenario_ret.std()*np.sqrt(252)
                    scenario_sharpe = (scenario_annual_ret - optimizer.risk_free_rate)/scenario_vol if scenario_vol!=0 else 0
                    st.write(f"Under a {shock}% shock, the annual return is {scenario_annual_ret*100:.2f}% and Sharpe Ratio is {scenario_sharpe:.2f}.")

                st.success(get_translated_text(lang, "success_optimize"))

            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
