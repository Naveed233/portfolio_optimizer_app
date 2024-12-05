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
    page_title="📈 Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define language options
languages = {
    'English': 'en',
    'Español': 'es'
}

# Define language strings
translations = {
    'en': {
        "title": "📈 Portfolio Optimization with Advanced Features",
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
        "train_lstm": "🤖 Train LSTM Model for Future Returns Prediction",
        "more_info_lstm": "ℹ️ More Information on LSTM",
        "optimize_portfolio": "📈 Optimize Portfolio",
        "portfolio_analysis": "🔍 Portfolio Analysis & Optimization Results",
        "success_lstm": "🤖 LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset to your portfolio before training the LSTM model.",
        "error_no_assets_opt": "Please add at least one asset to your portfolio before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "🔑 Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "📊 Portfolio Performance Metrics",
        "visual_analysis": "📊 Visual Analysis",
        "portfolio_composition": "📈 Portfolio Composition",
        "portfolio_metrics": "📊 Portfolio Performance Metrics",
        "correlation_heatmap": "📈 Asset Correlation Heatmap",
        "var": "Value at Risk (VaR)",
        "cvar": "Conditional Value at Risk (CVaR)",
        "max_drawdown": "Maximum Drawdown",
        "hhi": "Herfindahl-Hirschman Index (HHI)",
        "sharpe_ratio": "Sharpe Ratio",
        "explanation_var": "**Value at Risk (VaR):** Estimates the maximum potential loss of a portfolio over a specified time frame at a given confidence level.",
        "explanation_cvar": "**Conditional Value at Risk (CVaR):** Measures the expected loss exceeding the VaR, providing insights into tail risk.",
        "explanation_max_drawdown": "**Maximum Drawdown:** Measures the largest peak-to-trough decline in the portfolio value, indicating the worst-case scenario.",
        "explanation_hhi": "**Herfindahl-Hirschman Index (HHI):** A diversification metric that measures the concentration of investments in a portfolio.",
        "explanation_sharpe_ratio": "**Sharpe Ratio:** Measures risk-adjusted returns, indicating how much excess return you receive for the extra volatility endured.",
        "explanation_lstm": "**Explanation of Predicted Returns:**\nThe LSTM model is used to predict future stock returns based on historical price data. The graph displays the expected changes in returns for the next 30 business days. The model captures trends and seasonality, but it is important to understand that predictions have inherent uncertainty, especially due to market volatility. Use this information as an additional tool to make decisions rather than a definitive future outlook.",
        "success_optimize": "Portfolio optimization completed successfully!"
    },
    'es': {
        "title": "📈 Aplicación de Optimización de Portafolios con Funciones Avanzadas",
        "user_inputs": "🔧 Entradas del Usuario",
        "select_universe": "Selecciona un Universo de Activos:",
        "custom_tickers": "Ingrese los símbolos de acciones separados por comas (ej., AAPL, MSFT, TSLA):",
        "add_portfolio": "Agregar a Mi Portafolio",
        "my_portfolio": "📁 Mi Portafolio",
        "no_assets": "No se han agregado activos todavía.",
        "optimization_parameters": "📅 Parámetros de Optimización",
        "start_date": "Fecha de Inicio",
        "end_date": "Fecha Final",
        "risk_free_rate": "Ingrese la tasa libre de riesgo (en %):",
        "investment_strategy": "Elige tu Estrategia de Inversión:",
        "strategy_risk_free": "Inversión sin Riesgo",
        "strategy_profit": "Inversión Orientada a Ganancias",
        "target_return": "Selecciona un retorno objetivo específico (en %)",
        "train_lstm": "🤖 Entrenar Modelo LSTM para Predicción de Retornos Futuros",
        "more_info_lstm": "ℹ️ Más Información sobre LSTM",
        "optimize_portfolio": "📈 Optimizar Portafolio",
        "portfolio_analysis": "🔍 Análisis y Resultados de Optimización del Portafolio",
        "success_lstm": "🤖 ¡Modelo LSTM entrenado exitosamente!",
        "error_no_assets_lstm": "Por favor, agrega al menos un activo a tu portafolio antes de entrenar el modelo LSTM.",
        "error_no_assets_opt": "Por favor, agrega al menos un activo a tu portafolio antes de la optimización.",
        "error_date": "La fecha de inicio debe ser anterior a la fecha final.",
        "allocation_title": "🔑 Asignación Óptima del Portafolio (Retorno Objetivo: {target}%)",
        "performance_metrics": "📊 Métricas de Desempeño del Portafolio",
        "visual_analysis": "📊 Análisis Visual",
        "portfolio_composition": "📈 Composición del Portafolio",
        "portfolio_metrics": "📊 Métricas de Desempeño del Portafolio",
        "correlation_heatmap": "📈 Mapa de Calor de Correlación de Activos",
        "var": "Valor en Riesgo (VaR)",
        "cvar": "Valor en Riesgo Condicional (CVaR)",
        "max_drawdown": "Máximo Drawdown",
        "hhi": "Índice Herfindahl-Hirschman (HHI)",
        "sharpe_ratio": "Ratio de Sharpe",
        "explanation_var": "**Valor en Riesgo (VaR):** Estima la pérdida máxima potencial de un portafolio durante un período de tiempo específico en un nivel de confianza dado.",
        "explanation_cvar": "**Valor en Riesgo Condicional (CVaR):** Mide la pérdida esperada que excede el VaR, proporcionando información sobre el riesgo de cola.",
        "explanation_max_drawdown": "**Máximo Drawdown:** Mide la mayor caída de pico a valle en el valor del portafolio, indicando el peor escenario posible.",
        "explanation_hhi": "**Índice Herfindahl-Hirschman (HHI):** Una métrica de diversificación que mide la concentración de inversiones en un portafolio.",
        "explanation_sharpe_ratio": "**Ratio de Sharpe:** Mide los retornos ajustados por riesgo, indicando cuánto retorno excedente se recibe por la volatilidad adicional soportada.",
        "explanation_lstm": "**Explicación de Retornos Predichos:**\nEl modelo LSTM se utiliza para predecir retornos futuros de acciones basados en datos históricos de precios. El gráfico muestra los cambios esperados en los retornos para los próximos 30 días hábiles. El modelo captura tendencias y estacionalidades, pero es importante entender que las predicciones tienen incertidumbre inherente, especialmente debido a la volatilidad del mercado. Usa esta información como una herramienta adicional para tomar decisiones en lugar de una perspectiva definitiva del futuro.",
        "success_optimize": "¡Optimización del portafolio completada exitosamente!"
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

        return result.x if result.success else initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        """
        Optimize portfolio with added weight constraints for minimum volatility.
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
        Prepare data for LSTM model.
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

def get_translated_text(lang, key):
    """
    Retrieve translated text based on selected language.
    """
    return translations.get(lang, translations['en']).get(key, key)

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
        ticker_list = [ticker.strip() for ticker in custom_tickers.split(",") if ticker.strip()]
    else:
        selected_universe_assets = st.sidebar.multiselect(
            get_translated_text(lang, "add_portfolio"),
            universe_options[universe_choice],
            default=universe_options[universe_choice][:5]  # Default selection
        )
        ticker_list = [extract_ticker(asset) for asset in selected_universe_assets] if selected_universe_assets else []

    # Session state for portfolio
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []

    # Update 'My Portfolio' with selected assets
    if ticker_list:
        updated_portfolio = st.session_state['my_portfolio'] + [ticker for ticker in ticker_list if ticker not in st.session_state['my_portfolio']]
        st.session_state['my_portfolio'] = updated_portfolio

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

    # Specific Target Return Slider
    specific_target_return = st.sidebar.slider(
        get_translated_text(lang, "target_return"), 
        min_value=-5.0, max_value=20.0, value=5.0, step=0.1
    ) / 100

    # Train LSTM Button
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))

    # Optimize Button
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))

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
                X, y, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X, y, epochs=10, batch_size=32)

                st.success(get_translated_text(lang, "success_lstm"))

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
                ax.set_xlabel(get_translated_text(lang, "end_date"))
                ax.set_ylabel(get_translated_text(lang, "sharpe_ratio"))  # Adjust label as needed
                ax.set_title(get_translated_text(lang, "train_lstm"))
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Button to explain what the plot means
                if st.sidebar.button(get_translated_text(lang, "more_info_lstm")):
                    explanation = get_translated_text(lang, "explanation_lstm")
                    st.markdown(explanation)

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An error occurred during LSTM training or prediction.")
                st.error(f"{e}")

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
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                    details = "Details: You selected a 'Risk-free Investment' strategy, aiming for minimal risk exposure while attempting to achieve the specified target return."
                else:
                    # Optimize for Sharpe Ratio
                    optimal_weights = optimizer.optimize_sharpe_ratio()
                    details = "Details: You selected a 'Profit-focused Investment' strategy, aiming for maximum potential returns with an acceptance of higher risk."

                portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)

                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)

                # Display Allocation
                st.subheader(get_translated_text(lang, "allocation_title").format(target=round(specific_target_return*100, 2)))
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))

                # Display Performance Metrics
                st.subheader(get_translated_text(lang, "performance_metrics"))
                metrics = {
                    get_translated_text(lang, "sharpe_ratio"): sharpe_ratio,
                    get_translated_text(lang, "var"): var_95,
                    get_translated_text(lang, "cvar"): cvar_95,
                    get_translated_text(lang, "max_drawdown"): max_dd,
                    get_translated_text(lang, "hhi"): hhi,
                    "Expected Annual Return (%)": portfolio_return * 100,
                    "Annual Volatility (Risk) (%)": portfolio_volatility * 100,
                }
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                st.table(metrics_df.style.format({"Value": lambda x: f"{x:.2f}"}))

                # Display Risk Metrics with Explanations
                st.subheader(get_translated_text(lang, "performance_metrics"))
                for key in [get_translated_text(lang, "var"), get_translated_text(lang, "cvar"),
                            get_translated_text(lang, "max_drawdown"), get_translated_text(lang, "hhi"),
                            get_translated_text(lang, "sharpe_ratio")]:
                    st.markdown(f"**{key}:** {metrics[key]:.2%}" if 'HHI' not in key else f"**{key}:** {metrics[key]:.4f}")
                    explanation_key = f"explanation_{key.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                    explanation = translations[lang].get(explanation_key, "")
                    st.markdown(explanation)

                # Display Visuals
                st.subheader(get_translated_text(lang, "visual_analysis"))
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
                        "Annual Volatility (Risk) (%)": portfolio_volatility * 100,
                        "Sharpe Ratio": sharpe_ratio
                    }
                    metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
                    ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax2.patches:
                        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig2)

                # Correlation Heatmap
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                correlation_matrix = optimizer.returns.corr()
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
                plt.title(get_translated_text(lang, "correlation_heatmap"))
                plt.tight_layout()
                st.pyplot(fig3)

                st.success(get_translated_text(lang, "success_optimize"))

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An unexpected error occurred during optimization.")
                st.error(f"{e}")

if __name__ == "__main__":
    main()
