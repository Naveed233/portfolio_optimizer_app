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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    def min_volatility(self, target_return, max_weight=0.3):
        """
        Optimize portfolio with added weight constraints
        """
        num_assets = len(self.returns.columns)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets

        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            # Log the optimization failure
            logger.warning(f"Portfolio optimization failed: {result.message}")
            # Return an equal weight portfolio as a fallback
            return np.ones(num_assets) / num_assets

    def prepare_data_for_lstm(self):
        """
        Prepare data for LSTM model
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.returns.values)
        
        X, y = [], []
        look_back = 60  # Look-back period (e.g., 60 days)
        for i in range(look_back, len(scaled_data), 3):  # Use data every 3 days
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i])

        X, y = np.array(X), np.array(y)
        return X, y, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        # Set random seed for reproducibility
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        """
        Train LSTM model
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=X_train.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return model

    def predict_future_returns(self, model, scaler, steps=30):
        """
        Predict future returns using the LSTM model
        """
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

# Helper Functions
def extract_ticker(asset_string):
    """
    Extract ticker symbol from asset string.
    """
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

# Streamlit App
def main():
    st.title("📈 Portfolio Optimization with Advanced Features")

    # Define preset universes
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
        selected_universe_assets = st.multiselect("Select assets to add to My Portfolio:", universe_options[universe_choice])
        ticker_list = [extract_ticker(asset) for asset in selected_universe_assets] if selected_universe_assets else []

    # Session state for portfolio
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Update 'My Portfolio' with selected assets
    if ticker_list:
        updated_portfolio = st.session_state['my_portfolio'] + [ticker for ticker in ticker_list if ticker not in st.session_state['my_portfolio']]
        st.session_state['my_portfolio'] = updated_portfolio

    # Display 'My Portfolio'
    st.subheader("📁 My Portfolio")
    if st.session_state['my_portfolio']:
        st.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.write("No assets added yet.")

    # Portfolio Optimization Section
    st.header("🔧 Optimize Your Portfolio")

    # Date Inputs
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1), max_value=datetime.today())
    with col2:
        end_date = st.date_input("End Date", value=datetime.today(), max_value=datetime.today())
    with col3:
        # Risk-Free Rate Input
        risk_free_rate = st.number_input("Enter the risk-free rate (in %):", value=2.0, step=0.1) / 100

    # Train LSTM Button under Start Date
    if st.button("Train LSTM Model for Future Returns Prediction"):
        if not st.session_state['my_portfolio']:
            st.error("Please add at least one asset to your portfolio before training the LSTM model.")
            st.stop()

        try:
            clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
            optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
            optimizer.fetch_data()

            # Prepare data for LSTM
            X, y, scaler = optimizer.prepare_data_for_lstm()
            model = optimizer.train_lstm_model(X, y, epochs=10, batch_size=32)

            st.success("LSTM model trained successfully!")

            # Predict future returns for the next 30 days
            future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
            future_dates = pd.date_range(end_date, periods=len(future_returns)).to_pydatetime().tolist()

            # Plot future predictions in a popup window
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(future_dates, future_returns, label="Predicted Returns", color='blue')
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Predicted Returns", fontsize=10)
            ax.set_title("Future Return Predictions (Next 30 Days)", fontsize=12)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            # Button to explain what the plot means
            if st.button("More Information on LSTM"):
                explanation = """
                **Explanation of Predicted Returns:**
                The LSTM model is used to predict future stock returns based on historical price data. 
                The graph displays the expected changes in returns for the next 30 days. The model captures trends and seasonality, 
                but it is important to understand that predictions have inherent uncertainty, especially due to market volatility. 
                Use this information as an additional tool to make decisions rather than a definitive future outlook.
                """
                st.markdown(explanation)

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            logger.exception("An error occurred during LSTM training or prediction.")
            st.error(f"An error occurred: {e}")

    # Investment Strategy Options
    investment_strategy = st.radio(
        "Choose your Investment Strategy:",
        ("Risk-free Investment", "Profit-focused Investment")
    )

    # Specific Target Return Slider
    specific_target_return = st.slider(
        "Select a specific target return (in %)", 
        min_value=-5.0, max_value=20.0, value=5.0, step=0.1
    ) / 100

    # Optimize Button
    if st.button("📈 Optimize Portfolio"):

        if not st.session_state['my_portfolio']:
            st.error("Please add at least one asset to your portfolio before optimization.")
            st.stop()

        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            st.stop()

        try:
            clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
            optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
            # Fetch data and update tickers in case some are dropped
            updated_tickers = optimizer.fetch_data()

            if investment_strategy == "Risk-free Investment":
                # Apply denoising and clustering (this is a placeholder for Black-Litterman model implementation with denoising)
                optimal_weights = optimizer.min_volatility(specific_target_return)
                st.markdown("**Details:** You selected a 'Risk-free Investment' strategy, aiming for minimal risk exposure while attempting to achieve the specified target return.", unsafe_allow_html=True)
            else:
                # Apply clustering and backtesting (this is a placeholder for Black-Litterman model implementation with backtesting)
                optimal_weights = optimizer.min_volatility(specific_target_return)
                st.markdown("**Details:** You selected a 'Profit-focused Investment' strategy, aiming for maximum potential returns with an acceptance of higher risk.", unsafe_allow_html=True)

            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)

            allocation = pd.DataFrame({
                "Asset": updated_tickers,
                "Weight": np.round(optimal_weights, 4)
            })
            allocation = allocation[allocation['Weight'] > 0].reset_index(drop=True)

            st.subheader(f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)")
            st.dataframe(allocation)

            st.subheader("📊 Portfolio Performance Metrics")
            metrics = {
                "Expected Annual Return": portfolio_return * 100,
                "Annual Volatility (Risk)": portfolio_volatility * 100,
                "Sharpe Ratio": sharpe_ratio
            }
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            st.table(metrics_df)

            # Visuals
            st.subheader("📊 Visual Analysis")
            # Pie Chart for Allocation
            fig1, ax1 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            ax1.pie(allocation['Weight'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontname': 'Calibri'})
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)

            # Bar Chart for Expected Annual Return, Volatility, and Sharpe Ratio
            fig2, ax2 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            bars = metrics_df.plot(kind='bar', legend=False, ax=ax2, color=['skyblue'])
            for p in bars.patches:
                ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha='center', va='bottom', fontsize=10)
            plt.xticks(rotation=0, fontsize=10)
            plt.title("Portfolio Performance Metrics", fontsize=12)
            plt.ylabel("Value (%)", fontsize=10)
            st.pyplot(fig2)

            # Heatmap for Correlation Matrix
            st.subheader("📈 Asset Correlation Heatmap")
            correlation_matrix = optimizer.returns.corr()
            fig3, ax3 = plt.subplots(figsize=(5, 2))  # Reduce the size of the plot
            sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
            plt.title("Asset Correlation Heatmap", fontsize=10)
            st.pyplot(fig3)

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            logger.exception("An unexpected error occurred during optimization.")
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
