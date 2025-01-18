import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
import datetime
import seaborn as sns
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Logging Configuration ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Streamlit Page Configuration ----------
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Translation Dictionaries (Shortened for Example) ----------
translations = {
    'en': {
        "title": "Portfolio Optimization with Advanced Features",
        "user_inputs": "User Inputs",
        "select_universe": "Select an Asset Universe:",
        "custom_tickers": "Enter stock tickers (e.g., AAPL, MSFT):",
        "add_portfolio": "Add to My Portfolio",
        "my_portfolio": "My Portfolio",
        "no_assets": "No assets added yet.",
        "optimization_parameters": "Optimization Parameters",
        "start_date": "Start Date",
        "end_date": "End Date",
        "risk_free_rate": "Enter the risk-free rate (in %):",
        "investment_strategy": "Choose your Investment Strategy:",
        "strategy_risk_free": "Risk-free Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "train_lstm": "Train LSTM Model",
        "more_info_lstm": "More Information on LSTM",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "compare_portfolios": "Compare Portfolios",
        "portfolio_analysis": "Portfolio Analysis & Results",
        "success_lstm": "LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset before training LSTM.",
        "error_no_assets_opt": "Please add at least one asset before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "Portfolio Performance Metrics",
        "visual_analysis": "Visual Analysis",
        "portfolio_composition": "Portfolio Composition",
        "portfolio_metrics": "Portfolio Metrics",
        "correlation_heatmap": "Asset Correlation Heatmap",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_lstm": "LSTM is a type of neural network well-suited for time series forecasting.",
        "explanation_sharpe_button": "Optimization for highest Sharpe Ratio aims to achieve the best risk-adjusted returns."
    }
}

# Helper function to get translated text (you can extend for multiple languages)
def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

# ---------- Safe Price Fetching Function ----------
def fetch_price_data(tickers, start_date, end_date):

# ---------- Portfolio Optimizer Class ----------
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def load_data(self):
        """
        Use fetch_price_data to get Adjusted Close prices and compute daily returns.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = fetch_price_data(self.tickers, self.start_date, self.end_date)
        
        # Filter columns by valid tickers (they'll appear as columns in multi-ticker mode)
        valid_cols = [c for c in data.columns if c in self.tickers]
        data = data[valid_cols].copy()
        
        self.tickers = valid_cols  # update tickers to only valid ones
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers: {self.tickers}")

    def portfolio_stats(self, weights):
        """Return annualized return, volatility, and Sharpe ratio."""
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # ensure weights sum to 1

        # Daily average returns and covariance
        mean_daily_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        # Annualize
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        if portfolio_volatility != 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def value_at_risk(self, weights, confidence_level=0.95):
        """Calculate 1-day VaR using historical simulation."""
        portfolio_returns = self.returns.dot(weights)
        var_level = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var_level

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        """Calculate CVaR (Expected Shortfall)."""
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        """Calculate maximum drawdown for the portfolio returns."""
        portfolio_returns = self.returns.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative - peak) / peak
        return dd.min()  # negative number

    def herfindahl_hirschman_index(self, weights):
        return np.sum(np.array(weights) ** 2)

    def compute_sortino_calmar_beta_alpha(self, weights, market_returns=None):
        """
        Compute Sortino, Calmar, Beta, and Alpha.
        If no market benchmark is given, Beta=Alpha=0 by default.
        """
        ann_return, ann_vol, _ = self.portfolio_stats(weights)

        # Sortino Ratio
        daily_portfolio_returns = self.returns.dot(weights)
        downside_returns = daily_portfolio_returns[daily_portfolio_returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = (ann_return - self.risk_free_rate) / downside_std
        else:
            sortino_ratio = 0.0

        # Calmar Ratio = Annual Return / |Max Drawdown|
        max_dd = self.maximum_drawdown(weights)  # negative
        calmar_ratio = ann_return / abs(max_dd) if max_dd < 0 else 0.0

        # Beta & Alpha (only if a market is provided)
        if market_returns is not None:
            df = pd.DataFrame({
                'portfolio': daily_portfolio_returns,
                'market': market_returns
            }).dropna()
            if not df.empty:
                cov = np.cov(df['portfolio'], df['market'])[0][1]
                var_market = np.var(df['market'])
                beta = cov / var_market if var_market != 0 else 0.0
                market_annual_return = df['market'].mean() * 252
                alpha = ann_return - (beta * market_annual_return)
            else:
                beta, alpha = 0.0, 0.0
        else:
            beta, alpha = 0.0, 0.0

        return sortino_ratio, calmar_ratio, beta, alpha

    # ---------- Optimization Functions ----------
    def sharpe_ratio_objective(self, weights):
        """Negative Sharpe ratio (for minimization)."""
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe

    def optimize_sharpe_ratio(self):
        """Maximize Sharpe Ratio via SLSQP."""
        num_assets = len(self.tickers)
        init_w = np.repeat(1 / num_assets, num_assets)
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        opt_result = minimize(self.sharpe_ratio_objective, init_w,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        if not opt_result.success:
            logger.warning(f"Max Sharpe optimization failed: {opt_result.message}")
        return opt_result.x

    def min_volatility(self, target_return, max_weight=0.3):
        """
        Minimize volatility subject to a target return and max weight constraint.
        """
        num_assets = len(self.tickers)
        init_w = np.repeat(1 / num_assets, num_assets)
        bounds = tuple((0, max_weight) for _ in range(num_assets))

        # Return constraint: portfolio_return(weights) - target_return = 0
        def ret_constraint(weights):
            pr, _, _ = self.portfolio_stats(weights)
            return pr - target_return

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': ret_constraint}
        )

        def vol_objective(weights):
            _, vol, _ = self.portfolio_stats(weights)
            return vol

        opt_result = minimize(vol_objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if not opt_result.success:
            logger.warning(f"Min Vol optimization failed: {opt_result.message}")
        return opt_result.x

    # ---------- LSTM Methods ----------
    def prepare_data_for_lstm(self, look_back=60):
        """
        Prepare data (scaled returns) for LSTM.
        Uses the average of all tickers if multiple tickers are present.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_array = self.returns.mean(axis=1).values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data_array)

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Reshape to [samples, time steps, features=1]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, y_train, X_test, y_test, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Build and train a simple LSTM model.
        """
        # For reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(50))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)  # verbose=1 to see progress
        return model

    def evaluate_model(self, model, scaler, X_test, y_test):
        """
        Evaluate LSTM on test data.
        """
        predictions = model.predict(X_test)
        # inverse scaling
        predictions = scaler.inverse_transform(predictions)
        y_test = y_test.reshape(-1, 1)
        y_test = scaler.inverse_transform(y_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return mae, rmse, r2

    def predict_future_returns(self, model, scaler, steps=30, look_back=60):
        """
        Predict the next `steps` daily returns from the last `look_back` days of data.
        """
        data_array = self.returns.mean(axis=1).values.reshape(-1, 1)
        scaled_data = scaler.transform(data_array)  # must use the same scaler

        if len(scaled_data) < look_back:
            raise ValueError("Not enough historical data to predict future returns.")

        last_seq = scaled_data[-look_back:].reshape(1, look_back, 1)
        predictions_scaled = []
        curr_seq = last_seq.copy()

        for _ in range(steps):
            pred = model.predict(curr_seq)[0][0]
            predictions_scaled.append(pred)
            # shift the sequence, append the new prediction
            curr_seq = np.roll(curr_seq, -1, axis=1)
            curr_seq[0, -1, 0] = pred

        # inverse scale predictions
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions_scaled)
        return predictions.flatten()

# ---------- Metric Analysis Helpers ----------
def analyze_var(var):
    if var < -0.05:
        return "High Risk: Potential for significant daily loss."
    elif var < -0.02:
        return "Moderate Risk."
    else:
        return "Low Risk."

def analyze_cvar(cvar):
    if cvar < -0.07:
        return "High Tail Risk."
    elif cvar < -0.04:
        return "Moderate Tail Risk."
    else:
        return "Low Tail Risk."

def analyze_max_drawdown(dd):
    if dd < -0.20:
        return "Severe Drawdown."
    elif dd < -0.10:
        return "Moderate Drawdown."
    else:
        return "Minor Drawdown."

def analyze_hhi(hhi):
    if hhi > 0.6:
        return "High concentration (poor diversification)."
    elif hhi > 0.3:
        return "Moderate concentration."
    else:
        return "Well-diversified."

def analyze_sharpe(sharpe):
    if sharpe > 1.0:
        return "Great risk-adjusted return."
    elif sharpe >= 0.5:
        return "Acceptable risk-adjusted return."
    else:
        return "Poor risk-adjusted return."

# ---------- Display and Comparison ----------
def display_metrics_table(metrics):
    table_data = []
    for k, v in metrics.items():
        analysis = ""
        if k == "VaR":
            analysis = analyze_var(v)
            val = f"{v*100:.2f}%"
        elif k == "CVaR":
            analysis = analyze_cvar(v)
            val = f"{v*100:.2f}%"
        elif k == "Max Drawdown":
            analysis = analyze_max_drawdown(v)
            val = f"{v*100:.2f}%"
        elif k == "HHI":
            analysis = analyze_hhi(v)
            val = f"{v:.4f}"
        elif k in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta", "Alpha"]:
            val = f"{v:.2f}"
            if k == "Sharpe Ratio":
                analysis = analyze_sharpe(v)
        else:
            val = f"{v:.2f}"

        table_data.append({
            "Metric": k,
            "Value": val,
            "Analysis": analysis
        })
    df = pd.DataFrame(table_data)
    st.table(df)

def compare_portfolios(base_metrics, opt_metrics):
    comparison = []
    for k in base_metrics.keys():
        base_val = base_metrics[k]
        opt_val = opt_metrics[k]

        # Format for display
        def fmt(metric_key, val):
            if metric_key in ["VaR", "CVaR", "Max Drawdown"]:
                return f"{val*100:.2f}%"
            elif metric_key == "HHI":
                return f"{val:.4f}"
            elif metric_key in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta", "Alpha"]:
                return f"{val:.2f}"
            else:
                return f"{val:.2f}"

        base_str = fmt(k, base_val)
        opt_str = fmt(k, opt_val)

        # Decide which is better
        if k in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Alpha"]:
            better = "Optimized" if opt_val > base_val else "Base"
        elif k in ["VaR", "CVaR", "Max Drawdown", "HHI", "Beta"]:
            better = "Optimized" if opt_val < base_val else "Base"
        else:
            better = "-"

        comparison.append({
            "Metric": k,
            "Base": base_str,
            "Optimized": opt_str,
            "Better": better
        })

    df = pd.DataFrame(comparison)
    st.table(df)

# ---------- Main Streamlit App ----------
def main():
    lang = 'en'  # You can extend multi-language support if desired.

    st.title(get_translated_text(lang, "title"))

    # Preset universes (example):
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet'],
        'Finance Leaders': ['JPM - JPMorgan', 'BAC - Bank of America', 'GS - Goldman Sachs', 'BLK - BlackRock'],
        'Custom': []
    }

    # Sidebar
    st.sidebar.header(get_translated_text(lang, "user_inputs"))
    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), list(universe_options.keys()))

    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(get_translated_text(lang, "custom_tickers"), value="")
        if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
            if "my_portfolio" not in st.session_state:
                st.session_state["my_portfolio"] = []
            new_tickers = [x.strip().upper() for x in custom_tickers.split(",") if x.strip() != ""]
            st.session_state["my_portfolio"] = list(set(st.session_state["my_portfolio"] + new_tickers))
    else:
        selected = st.sidebar.multiselect(get_translated_text(lang, "select_universe"), universe_options[universe_choice], default=[])
        if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
            if "my_portfolio" not in st.session_state:
                st.session_state["my_portfolio"] = []
            # Extract ticker from something like "AAPL - Apple"
            def extract_ticker(asset_string):
                return asset_string.split(" - ")[0].strip() if " - " in asset_string else asset_string.strip()
            new_tickers = [extract_ticker(x) for x in selected]
            st.session_state["my_portfolio"] = list(set(st.session_state["my_portfolio"] + new_tickers))

    # Display portfolio
    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if "my_portfolio" not in st.session_state or not st.session_state["my_portfolio"]:
        st.sidebar.write(get_translated_text(lang, "no_assets"))
    else:
        st.sidebar.write(", ".join(st.session_state["my_portfolio"]))

    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), datetime.date(2023,1,1))
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), datetime.date.today())
    rf_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1) / 100.0

    strategy = st.sidebar.radio(get_translated_text(lang, "investment_strategy"),
                                (get_translated_text(lang, "strategy_risk_free"),
                                 get_translated_text(lang, "strategy_profit")))

    target_return = None
    if strategy == get_translated_text(lang, "strategy_risk_free"):
        target_return = st.sidebar.slider(get_translated_text(lang, "target_return"), min_value=-5.0, max_value=20.0, value=5.0) / 100.0

    btn_train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))
    btn_optimize = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))
    btn_optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))
    btn_compare = st.sidebar.button(get_translated_text(lang, "compare_portfolios"))

    st.header(get_translated_text(lang, "portfolio_analysis"))

    # Session for storing results
    if "base_portfolio_metrics" not in st.session_state:
        st.session_state["base_portfolio_metrics"] = None
    if "optimized_portfolio_metrics" not in st.session_state:
        st.session_state["optimized_portfolio_metrics"] = None

    # -------- LSTM Training --------
    if btn_train_lstm:
        if "my_portfolio" not in st.session_state or not st.session_state["my_portfolio"]:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                po = PortfolioOptimizer(st.session_state["my_portfolio"], start_date, end_date, risk_free_rate=rf_rate)
                po.load_data()  # uses fetch_price_data internally
                X_train, y_train, X_test, y_test, scaler = po.prepare_data_for_lstm(look_back=30)
                model = po.train_lstm_model(X_train, y_train, epochs=5, batch_size=16)
                mae, rmse, r2 = po.evaluate_model(model, scaler, X_test, y_test)

                st.success(get_translated_text(lang, "success_lstm"))
                st.write(f"**MAE**: {mae:.4f}, **RMSE**: {rmse:.4f}, **R^2**: {r2:.4f}")

                # Predict future returns
                future_preds = po.predict_future_returns(model, scaler, steps=30, look_back=30)
                # Plot
                future_dates = pd.date_range(end_date, periods=len(future_preds)+1, freq='B')[1:]
                df_preds = pd.DataFrame({"Date": future_dates, "Predicted Return": future_preds})
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(df_preds["Date"], df_preds["Predicted Return"], marker='o')
                ax.set_title("Future Returns Prediction")
                ax.set_xticklabels(df_preds["Date"], rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                with st.expander(get_translated_text(lang, "more_info_lstm")):
                    st.markdown(get_translated_text(lang, "explanation_lstm"))

            except Exception as e:
                st.error(str(e))

    # -------- Optimize Portfolio (Min Vol or Max Sharpe) --------
    if btn_optimize:
        if "my_portfolio" not in st.session_state or not st.session_state["my_portfolio"]:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                po = PortfolioOptimizer(st.session_state["my_portfolio"], start_date, end_date, rf_rate)
                po.load_data()

                if strategy == get_translated_text(lang, "strategy_risk_free"):
                    if target_return is None:
                        st.error("Please select a target return.")
                        return
                    w_opt = po.min_volatility(target_return=target_return, max_weight=0.3)
                    strategy_desc = "Min Volatility (Risk-free style)"
                else:
                    w_opt = po.optimize_sharpe_ratio()
                    strategy_desc = "Max Sharpe (Profit-focused)"

                # Stats
                ret, vol, sr = po.portfolio_stats(w_opt)
                var = po.value_at_risk(w_opt, 0.95)
                cvar = po.conditional_value_at_risk(w_opt, 0.95)
                mdd = po.maximum_drawdown(w_opt)
                hhi = po.herfindahl_hirschman_index(w_opt)
                sortino, calmar, beta, alpha = po.compute_sortino_calmar_beta_alpha(w_opt)

                df_alloc = pd.DataFrame({
                    "Ticker": po.tickers,
                    "Weight(%)": [round(x*100,2) for x in w_opt]
                })
                df_alloc = df_alloc[df_alloc["Weight(%)"] > 0].reset_index(drop=True)

                tgt_str = f"{target_return*100:.2f}" if target_return else "N/A"
                st.subheader(get_translated_text(lang, "allocation_title").format(target=tgt_str))
                st.dataframe(df_alloc)

                metrics_dict = {
                    "Annual Return": ret,
                    "Annual Volatility": vol,
                    "Sharpe Ratio": sr,
                    "Sortino Ratio": sortino,
                    "Calmar Ratio": calmar,
                    "Beta": beta,
                    "Alpha": alpha,
                    "VaR": var,
                    "CVaR": cvar,
                    "Max Drawdown": mdd,
                    "HHI": hhi
                }

                if strategy == get_translated_text(lang, "strategy_risk_free"):
                    st.session_state["base_portfolio_metrics"] = metrics_dict
                else:
                    st.session_state["optimized_portfolio_metrics"] = metrics_dict

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics_dict)

                # Pie chart
                fig_alloc, ax_alloc = plt.subplots(figsize=(5,5))
                ax_alloc.pie(df_alloc["Weight(%)"], labels=df_alloc["Ticker"], autopct="%1.1f%%", startangle=140)
                ax_alloc.set_title(get_translated_text(lang, "portfolio_composition"))
                st.pyplot(fig_alloc)

                # Correlation Heatmap
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                corr_matrix = po.returns.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(6,5))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
                st.pyplot(fig_corr)

                st.success(f"{get_translated_text(lang, 'success_optimize')} ({strategy_desc})")

            except Exception as e:
                st.error(str(e))

    # -------- Optimize for Highest Sharpe (Shortcut Button) --------
    if btn_optimize_sharpe:
        if "my_portfolio" not in st.session_state or not st.session_state["my_portfolio"]:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                po = PortfolioOptimizer(st.session_state["my_portfolio"], start_date, end_date, rf_rate)
                po.load_data()

                w_opt = po.optimize_sharpe_ratio()
                ret, vol, sr = po.portfolio_stats(w_opt)
                var = po.value_at_risk(w_opt, 0.95)
                cvar = po.conditional_value_at_risk(w_opt, 0.95)
                mdd = po.maximum_drawdown(w_opt)
                hhi = po.herfindahl_hirschman_index(w_opt)
                sortino, calmar, beta, alpha = po.compute_sortino_calmar_beta_alpha(w_opt)

                df_alloc = pd.DataFrame({
                    "Ticker": po.tickers,
                    "Weight(%)": [round(x*100,2) for x in w_opt]
                })
                df_alloc = df_alloc[df_alloc["Weight(%)"] > 0].reset_index(drop=True)

                st.subheader("Optimal Portfolio Allocation (Highest Sharpe)")
                st.dataframe(df_alloc)

                metrics_dict = {
                    "Annual Return": ret,
                    "Annual Volatility": vol,
                    "Sharpe Ratio": sr,
                    "Sortino Ratio": sortino,
                    "Calmar Ratio": calmar,
                    "Beta": beta,
                    "Alpha": alpha,
                    "VaR": var,
                    "CVaR": cvar,
                    "Max Drawdown": mdd,
                    "HHI": hhi
                }
                st.session_state["optimized_portfolio_metrics"] = metrics_dict

                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics_dict)

                # Pie chart
                fig_alloc, ax_alloc = plt.subplots(figsize=(5,5))
                ax_alloc.pie(df_alloc["Weight(%)"], labels=df_alloc["Ticker"], autopct="%1.1f%%", startangle=140)
                ax_alloc.set_title(get_translated_text(lang, "portfolio_composition"))
                st.pyplot(fig_alloc)

                # Correlation Heatmap
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                corr_matrix = po.returns.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(6,5))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
                st.pyplot(fig_corr)

                st.success(get_translated_text(lang, "explanation_sharpe_button"))

            except Exception as e:
                st.error(str(e))

    # -------- Compare Portfolios --------
    if btn_compare:
        base = st.session_state["base_portfolio_metrics"]
        opt = st.session_state["optimized_portfolio_metrics"]
        if base is None or opt is None:
            st.error("Both 'base' (risk-free) and 'optimized' (max Sharpe) must be computed before comparison.")
        else:
            st.subheader("Comparison of Base vs Optimized Portfolio")
            compare_portfolios(base, opt)

if __name__ == "__main__":
    main()
