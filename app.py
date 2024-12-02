import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit Title and Description
st.title("Stock Market Prediction & Insights")
st.subheader("By Ujwal Mojidra")
st.markdown(
    """
    **Explore stock market trends and predict future prices using machine learning.**
    - Select a stock ticker from the dropdown.
    - View historical trends and predictions for up to 4 years.
    """
)

# Dropdown for stock selection
stocks = (
    'GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA', 'PYPL', 'V', 'ENPH', '^NSEI',
    'TTM', 'SBUX', 'SBIN.NS', 'MGAM', 'BTC-USD', 'GC=F', 'NFT-USD', 'ETH-USD'
)
selected_stock = st.selectbox(
    'Choose your favorite stock for forecasting using their ticker:',
    stocks
)

# Slider for prediction years
n_years = st.slider('Years of prediction:', 1, 4, 1)
period = n_years * 365

# Load data function with new caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    return data.reset_index()  # Return the data without mutating it outside the function

# Load data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

# Display raw data
st.subheader("Raw Data")
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='orange')))
    fig.layout.update(
        title_text="Time Series Data with Range Slider",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig)

# Display raw data plot
plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()  # Use .copy() to avoid mutating cached data
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train Prophet model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Moving Average Plot
def plot_moving_averages():
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name="50-Day MA", line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name="200-Day MA", line=dict(color='green', dash='dot')))
    fig.layout.update(
        title_text="Moving Averages (50-Day & 200-Day)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

plot_moving_averages()

# Volume Analysis
st.subheader("Volume Analysis")
fig_volume = go.Figure()
fig_volume.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", marker=dict(color='purple')))
fig_volume.layout.update(
    title_text="Stock Volume Over Time",
    xaxis_title="Date",
    yaxis_title="Volume",
    template="plotly_dark"
)
st.plotly_chart(fig_volume)
