import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Alpha Vantage API
API_KEY = 'GVRDOJPM18JD9YDK'
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Function to get stock data
def get_stock_data(symbol):
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
    return data

# Function to simulate news fetching (placeholder)
def get_news(symbol):
    # This is a placeholder. You should replace it with an actual news API call
    return [
        {"title": f"Breaking News for {symbol}", "url": "http://example.com/news1"},
        {"title": f"Latest Update on {symbol}", "url": "http://example.com/news2"},
        {"title": f"Market Analysis for {symbol}", "url": "http://example.com/news3"}
    ]

# Display function for stocks
def display_stock_data(symbol):
    st.write(f"### Stock Data for {symbol}")
    data = get_stock_data(symbol)
    st.line_chart(data['4. close'])

# Streamlit UI components
st.title("Alpha Vantage Portfolio Optimizer")

# Sidebar configurations
symbol = st.sidebar.text_input("Enter stock symbol, e.g., 'IBM'", 'IBM')
if st.sidebar.button("Fetch Stock Data"):
    display_stock_data(symbol)
    news_items = get_news(symbol)
    st.write("### News Articles")
    for item in news_items:
        st.markdown(f"[{item['title']}]({item['url']})")

# Plotting portfolio optimizations would use similar fetch and display logic
# Continue implementing portfolio management and optimization features as outlined previously
