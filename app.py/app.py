import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import date, timedelta
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="Pavan Stocks Analysis - Real-Time Data",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 50px;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background-color:;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stock-card {
        background-color:;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive-change {
        color: green;
        font-weight: bold;
    }
    .negative-change {
        color: red;
        font-weight: bold;
    }
    .info-text {
        font-size: 14px;
        color: #555;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .home-button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        margin-top: 20px !important;
    }
    .analysis-button {
        background-color: #FF4B4B !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 15px !important;
        margin-top: 20px !important;
    }
    /* Fix for metric display */
    [data-testid="stMetricValue"] {
        font-size: 18px;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Home page function
def show_home_page():
    st.markdown('<h1 class="main-header">📈 Pavan Stocks Analysis - Real-Time Data</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏠 Welcome to Indian Stocks Analysis Platform
    
    ### 📊 Comprehensive Market Analysis
    This platform provides real-time analysis of Indian stocks with advanced technical indicators,
    interactive charts, and comprehensive market data.
    
    ### 🚀 Key Features:
    - **Real-time Data**: Live market data from Yahoo Finance
    - **500+ Indian Stocks**: Comprehensive coverage of NSE and BSE listed companies
    - **Advanced Technical Indicators**: 10+ technical indicators for detailed analysis
    - **Custom Time Frames**: Analyze any period from 1 day to several years
    - **Interactive Charts**: Candlestick charts with technical overlays
    - **Export Capability**: Download data for further analysis
    
    ### 📈 How to Use:
    1. Select a stock from the sidebar dropdown
    2. Choose your analysis time frame
    3. Select technical indicators to display
    4. Click "Start Analysis" to generate analysis
    5. Use the "Download Data" button to export results
    
    ### ℹ️ Data Source:
    This application uses Yahoo Finance data through the yfinance library, providing
    legal access to real-time and historical market data.
    
    **Disclaimer:** This tool is for educational purposes only. Always consult with
    a qualified financial advisor before making investment decisions.
    """)
    
    # Quick stats section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks Available", "500+")
    with col2:
        st.metric("Technical Indicators", "10+")
    with col3:
        st.metric("Data Points", "1M+")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to select a stock and start your analysis!")

# Technical indicator functions
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band, sma

def calculate_stochastic(data, high, low, window=14, smooth_k=3, smooth_d=3):
    low_min = low.rolling(window=window).min()
    high_max = high.rolling(window=window).max()
    stoch = 100 * (data - low_min) / (high_max - low_min)
    stoch_k = stoch.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    return stoch_k, stoch_d

def calculate_obv(data, volume):
    obv = (np.sign(data.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_atr(high, low, close, window=14):
    tr = np.maximum(high - low, 
                   np.maximum(abs(high - close.shift()), 
                             abs(low - close.shift())))
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_adx(high, low, close, window=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    tr = calculate_atr(high, low, close, window)
    plus_di = 100 * (plus_dm / tr).rolling(window=window).mean()
    minus_di = 100 * (minus_dm / tr).rolling(window=window).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    return adx, plus_di, minus_di

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
    return cci

# Function to get real-time stock data using yfinance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, start_date, end_date):
    """
    Get stock data using yfinance library
    """
    try:
        # Download historical data
        stock = yf.Ticker(symbol)
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            st.warning(f"No data found for {symbol}. Please check the symbol.")
            return None
            
        return hist_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to get real-time quote
def get_realtime_quote(symbol):
    """
    Get real-time quote using yfinance
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        st.warning(f"Could not fetch real-time quote: {str(e)}")
        return None

# Comprehensive list of Indian stocks
indian_stocks = {
    # Nifty 50 Companies
    'RELIANCE': 'RELIANCE.NS',
    'TATA CONSULTANCY': 'TCS.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'INFOSYS': 'INFY.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'SBIN': 'SBIN.NS',
    'BHARTI AIRTEL': 'BHARTIARTL.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'LT': 'LT.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'HCLTECH': 'HCLTECH.NS',
    'AXISBANK': 'AXISBANK.NS',
    'MARUTI': 'MARUTI.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'TITAN': 'TITAN.NS',
    'M&M': 'M&M.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS',
    'POWERGRID': 'POWERGRID.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'WIPRO': 'WIPRO.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'TECHM': 'TECHM.NS',
    'HINDALCO': 'HINDALCO.NS',
    'JSWSTEEL': 'JSWSTEEL.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'GRASIM': 'GRASIM.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'DRREDDY': 'DRREDDY.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    'SBILIFE': 'SBILIFE.NS',
    'BPCL': 'BPCL.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'UPL': 'UPL.NS',
    'COALINDIA': 'COALINDIA.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS',
    'CIPLA': 'CIPLA.NS',
    'ADANIENT': 'ADANIENT.NS',
    'HDFC': 'HDFC.NS',
    
    # Other Popular Stocks
    'VEDANTA': 'VEDL.NS',
    'PIDILITIND': 'PIDILITIND.NS',
    'SHREECEM': 'SHREECEM.NS',
    'AMBUJACEM': 'AMBUJACEM.NS',
    'ACC': 'ACC.NS',
    'GAIL': 'GAIL.NS',
    'IOC': 'IOC.NS',
    'HINDPETRO': 'HINDPETRO.NS',
    'BOSCHLTD': 'BOSCHLTD.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'TATAPOWER': 'TATAPOWER.NS',
    'BEL': 'BEL.NS',
    'HAL': 'HAL.NS',
    'BHEL': 'BHEL.NS',
    'DLF': 'DLF.NS',
    'INDIGO': 'INDIGO.NS',
    'MCDOWELL-N': 'MCDOWELL-N.NS',
    'HAVELLS': 'HAVELLS.NS',
    'SIEMENS': 'SIEMENS.NS',
    'ABB': 'ABB.NS',
    'LUPIN': 'LUPIN.NS',
    'AUROPHARMA': 'AUROPHARMA.NS',
    'BIOCON': 'BIOCON.NS',
    'GODREJCP': 'GODREJCP.NS',
    'DABUR': 'DABUR.NS',
    'BERGEPAINT': 'BERGEPAINT.NS',
    'PGHH': 'PGHH.NS',
    'COLPAL': 'COLPAL.NS',
    'MARICO': 'MARICO.NS',
    'ASHOKLEY': 'ASHOKLEY.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    'IRCTC': 'IRCTC.NS',
    'ZOMATO': 'ZOMATO.NS',
    'PAYTM': 'PAYTM.NS',
    'NAUKRI': 'NAUKRI.NS',
    'HINDCOPPER': 'HINDCOPPER.NS',
    'NATIONALUM': 'NATIONALUM.NS',
    'JINDALSTEL': 'JINDALSTEL.NS',
    'SAIL': 'SAIL.NS',
    'TATACHEM': 'TATACHEM.NS',
    'RCF': 'RCF.NS',
    'GNFC': 'GNFC.NS',
    'FACT': 'FACT.NS',
    'GSFC': 'GSFC.NS',
    '20MICRONS': '20MICRONS.NS',
    
    # Indices
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'NIFTY IT': '^CNXIT',
}

# Initialize session state for page navigation and analysis
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# Sidebar for navigation
st.sidebar.header("Navigation")
if st.sidebar.button("🏠 Home Page", key="home_button", use_container_width=True):
    st.session_state.page = 'home'
    st.session_state.analysis_started = False

st.sidebar.header("Analysis Parameters")

# User inputs
selected_stock = st.sidebar.selectbox("Select Stock", list(indian_stocks.keys()))
symbol = indian_stocks[selected_stock]

# Time frame selection with custom option
time_frame = st.sidebar.radio("Select Time Frame", 
                             ['3 Months', '6 Months', '1 Year', 'Custom'])

if time_frame == 'Custom':
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
else:
    if time_frame == '3 Months':
        start_date = datetime.date.today() - datetime.timedelta(days=90)
    elif time_frame == '6 Months':
        start_date = datetime.date.today() - datetime.timedelta(days=180)
    else:  # 1 Year
        start_date = datetime.date.today() - datetime.timedelta(days=365)
    end_date = datetime.date.today()

# Technical indicators options
st.sidebar.header("Technical Indicators")
show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)

if show_technical_indicators:
    indicator_options = st.sidebar.multiselect(
        "Select Technical Indicators",
        [
            "Simple Moving Average (SMA)", 
            "Exponential Moving Average (EMA)",
            "Relative Strength Index (RSI)",
            "Moving Average Convergence Divergence (MACD)",
            "Bollinger Bands",
            "Stochastic Oscillator",
            "On-Balance Volume (OBV)",
            "Average True Range (ATR)",
            "Average Directional Index (ADX)",
            "Commodity Channel Index (CCI)"
        ],
        ["Simple Moving Average (SMA)", "Relative Strength Index (RSI)"]
    )

# Start Analysis Button
if st.sidebar.button("🚀 Start Analysis", key="start_analysis", use_container_width=True, 
                    help="Click to begin stock analysis with selected parameters"):
    st.session_state.analysis_started = True
    st.session_state.page = 'analysis'

# Show home page or analysis based on navigation
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'analysis' and st.session_state.analysis_started:
    # Display analysis in progress message
    with st.spinner("🔍 Analyzing stock data... Please wait"):
        # Get historical data
        historical_df = get_stock_data(symbol, start_date, end_date)
        
        # Get real-time quote
        realtime_info = get_realtime_quote(symbol)
        
        if historical_df is None or historical_df.empty:
            st.error("No data available for the selected stock. Please try a different stock.")
        else:
            # Display stock information
            st.markdown(f'<div class="stock-card"><h2>{selected_stock} ({symbol})</h2></div>', unsafe_allow_html=True)
            
            # Calculate metrics from historical data
            current_price = historical_df['Close'][-1] if len(historical_df) > 0 else 0
            previous_close = historical_df['Close'][-2] if len(historical_df) > 1 else current_price
            price_change = current_price - previous_close
            percent_change = (price_change / previous_close) * 100 if previous_close else 0
            
            # Use real-time data if available
            if realtime_info and 'regularMarketPrice' in realtime_info:
                current_price = realtime_info.get('regularMarketPrice', current_price)
                previous_close = realtime_info.get('regularMarketPreviousClose', previous_close)
                price_change = current_price - previous_close
                percent_change = (price_change / previous_close) * 100 if previous_close else 0
            
            # Display metrics
            st.subheader("📊 Current Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}", 
                         f"₹{price_change:.2f} ({percent_change:.2f}%)")
            with col2:
                open_price = realtime_info.get('regularMarketOpen', historical_df['Open'][-1]) if realtime_info else historical_df['Open'][-1]
                st.metric("Open", f"₹{open_price:.2f}")
            with col3:
                day_high = realtime_info.get('regularMarketDayHigh', historical_df['High'][-1]) if realtime_info else historical_df['High'][-1]
                st.metric("Day's High", f"₹{day_high:.2f}")
            with col4:
                day_low = realtime_info.get('regularMarketDayLow', historical_df['Low'][-1]) if realtime_info else historical_df['Low'][-1]
                st.metric("Day's Low", f"₹{day_low:.2f}")
            
            # Display additional metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                year_high = realtime_info.get('fiftyTwoWeekHigh', historical_df['High'].max()) if realtime_info else historical_df['High'].max()
                st.metric("52 Week High", f"₹{year_high:.2f}")
            with col6:
                year_low = realtime_info.get('fiftyTwoWeekLow', historical_df['Low'].min()) if realtime_info else historical_df['Low'].min()
                st.metric("52 Week Low", f"₹{year_low:.2f}")
            with col7:
                volume = realtime_info.get('regularMarketVolume', historical_df['Volume'][-1]) if realtime_info else historical_df['Volume'][-1]
                st.metric("Volume", f"{volume:,.0f}")
            with col8:
                avg_volume = realtime_info.get('averageVolume', historical_df['Volume'].mean()) if realtime_info else historical_df['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            # Price chart
            st.subheader(f"📈 Price Chart ({time_frame})")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=historical_df.index,
                open=historical_df['Open'],
                high=historical_df['High'],
                low=historical_df['Low'],
                close=historical_df['Close'],
                name='Price'
            ))
            
            # Add selected technical indicators
            if show_technical_indicators:
                if "Simple Moving Average (SMA)" in indicator_options:
                    historical_df['SMA_20'] = calculate_sma(historical_df['Close'], 20)
                    historical_df['SMA_50'] = calculate_sma(historical_df['Close'], 50)
                    fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['SMA_20'], name='SMA 20', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['SMA_50'], name='SMA 50', line=dict(color='purple')))
                
                if "Exponential Moving Average (EMA)" in indicator_options:
                    historical_df['EMA_12'] = calculate_ema(historical_df['Close'], 12)
                    historical_df['EMA_26'] = calculate_ema(historical_df['Close'], 26)
                    fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['EMA_12'], name='EMA 12', line=dict(color='cyan')))
                    fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['EMA_26'], name='EMA 26', line=dict(color='magenta')))
                
                if "Bollinger Bands" in indicator_options:
                    upper_band, lower_band, middle_band = calculate_bollinger_bands(historical_df['Close'], 20, 2)
                    fig.add_trace(go.Scatter(x=historical_df.index, y=upper_band, name='Upper BB', line=dict(color='gray', dash='dash')))
                    fig.add_trace(go.Scatter(x=historical_df.index, y=lower_band, name='Lower BB', line=dict(color='gray', dash='dash')))
                    fig.add_trace(go.Scatter(x=historical_df.index, y=middle_band, name='SMA 20', line=dict(color='blue', dash='dot')))
            
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected_stock} Price Chart")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display historical data
            st.subheader(f"📋 Historical Data ({start_date} to {end_date})")
            
            # Create a copy for display
            display_df = historical_df.copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            
            # Format numbers
            for col in ['Open', 'High', 'Low', 'Close']:
                display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}")
            
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            
            # Display the complete table with all data
            st.dataframe(display_df, use_container_width=True)
            
            # Technical indicators in separate sections
            if show_technical_indicators:
                st.subheader("🔧 Technical Analysis")
                
                # Create tabs for different indicator groups
                tab1, tab2, tab3 = st.tabs(["Momentum Indicators", "Volatility Indicators", "Volume Indicators"])
                
                with tab1:
                    # RSI Chart
                    if "Relative Strength Index (RSI)" in indicator_options:
                        st.subheader("RSI (Relative Strength Index)")
                        historical_df['RSI'] = calculate_rsi(historical_df['Close'], 14)
                        
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=historical_df.index, y=historical_df['RSI'], name='RSI', line=dict(color='blue')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(height=300, title="RSI (14 periods)")
                        st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # Interpretation of RSI
                        if len(historical_df) > 0 and 'RSI' in historical_df and not pd.isna(historical_df['RSI'].iloc[-1]):
                            latest_rsi = historical_df['RSI'].iloc[-1]
                            if latest_rsi > 70:
                                st.warning("RSI is above 70. The stock may be overbought. Consider potential selling opportunities.")
                            elif latest_rsi < 30:
                                st.info("RSI is below 30. The stock may be oversold. Consider potential buying opportunities.")
                            else:
                                st.success("RSI is between 30 and 70. The stock is in neutral territory.")
                    
                    # Stochastic Oscillator
                    if "Stochastic Oscillator" in indicator_options:
                        st.subheader("Stochastic Oscillator")
                        stoch_k, stoch_d = calculate_stochastic(historical_df['Close'], historical_df['High'], historical_df['Low'])
                        
                        fig_stoch = go.Figure()
                        fig_stoch.add_trace(go.Scatter(x=historical_df.index, y=stoch_k, name='%K', line=dict(color='blue')))
                        fig_stoch.add_trace(go.Scatter(x=historical_df.index, y=stoch_d, name='%D', line=dict(color='red')))
                        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
                        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
                        fig_stoch.update_layout(height=300, title="Stochastic Oscillator")
                        st.plotly_chart(fig_stoch, use_container_width=True)
                
                with tab2:
                    # MACD Chart
                    if "Moving Average Convergence Divergence (MACD)" in indicator_options:
                        st.subheader("MACD (Moving Average Convergence Divergence)")
                        macd, macd_signal, macd_histogram = calculate_macd(historical_df['Close'])
                        
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=historical_df.index, y=macd, name='MACD', line=dict(color='blue')))
                        fig_macd.add_trace(go.Scatter(x=historical_df.index, y=macd_signal, name='Signal', line=dict(color='red')))
                        fig_macd.add_trace(go.Bar(x=historical_df.index, y=macd_histogram, name='Histogram', 
                                                marker_color=np.where(macd_histogram >= 0, 'green', 'red')))
                        fig_macd.update_layout(height=300, title="MACD")
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # ATR
                    if "Average True Range (ATR)" in indicator_options:
                        st.subheader("Average True Range (ATR)")
                        atr = calculate_atr(historical_df['High'], historical_df['Low'], historical_df['Close'])
                        
                        fig_atr = go.Figure()
                        fig_atr.add_trace(go.Scatter(x=historical_df.index, y=atr, name='ATR', line=dict(color='purple')))
                        fig_atr.update_layout(height=300, title="Average True Range (14 periods)")
                        st.plotly_chart(fig_atr, use_container_width=True)
                    
                    # ADX
                    if "Average Directional Index (ADX)" in indicator_options:
                        st.subheader("Average Directional Index (ADX)")
                        adx, plus_di, minus_di = calculate_adx(historical_df['High'], historical_df['Low'], historical_df['Close'])
                        
                        fig_adx = go.Figure()
                        fig_adx.add_trace(go.Scatter(x=historical_df.index, y=adx, name='ADX', line=dict(color='black')))
                        fig_adx.add_trace(go.Scatter(x=historical_df.index, y=plus_di, name='+DI', line=dict(color='green')))
                        fig_adx.add_trace(go.Scatter(x=historical_df.index, y=minus_di, name='-DI', line=dict(color='red')))
                        fig_adx.update_layout(height=300, title="Average Directional Index")
                        st.plotly_chart(fig_adx, use_container_width=True)
                
                with tab3:
                    # OBV Chart
                    if "On-Balance Volume (OBV)" in indicator_options:
                        st.subheader("On-Balance Volume (OBV)")
                        obv = calculate_obv(historical_df['Close'], historical_df['Volume'])
                        
                        fig_obv = go.Figure()
                        fig_obv.add_trace(go.Scatter(x=historical_df.index, y=obv, name='OBV', line=dict(color='blue')))
                        fig_obv.update_layout(height=300, title="On-Balance Volume")
                        st.plotly_chart(fig_obv, use_container_width=True)
                    
                    # Volume Chart
                    st.subheader("Volume Analysis")
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(x=historical_df.index, y=historical_df['Volume'], name='Volume', 
                                              marker_color=np.where(historical_df['Close'] > historical_df['Open'], 'green', 'red')))
                    fig_volume.update_layout(height=300, title="Trading Volume")
                    st.plotly_chart(fig_volume, use_container_width=True)
            
            # Performance statistics
            st.subheader("📊 Performance Statistics")
            returns = historical_df['Close'].pct_change().dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Daily Return", f"{returns.mean()*100:.4f}%" if len(returns) > 0 else "N/A")
            with col2:
                st.metric("Volatility (Std Dev)", f"{returns.std()*100:.4f}%" if len(returns) > 0 else "N/A")
            with col3:
                st.metric("Max Daily Gain", f"{returns.max()*100:.4f}%" if len(returns) > 0 else "N/A")
            with col4:
                st.metric("Max Daily Loss", f"{returns.min()*100:.4f}%" if len(returns) > 0 else "N/A")
                
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            if len(returns) > 0:
                volatility = returns.std()
                if volatility < 0.01:
                    st.success("Low volatility - Lower risk")
                elif volatility < 0.02:
                    st.warning("Moderate volatility - Medium risk")
                else:
                    st.error("High volatility - Higher risk")
            else:
                st.info("Insufficient data for risk assessment")
            
            # Download data button
            st.subheader("💾 Download Data")
            csv = historical_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Historical Data as CSV",
                data=csv,
                file_name=f"{selected_stock}_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Add a button to go back to home
            if st.button("🏠 Back to Home", use_container_width=True):
                st.session_state.page = 'home'
                st.session_state.analysis_started = False
                st.rerun()

# Footer with legal disclaimer
st.markdown("---")
st.markdown("""
<p class='info-text'>
<strong>Data Source:</strong> This application uses Yahoo Finance data through the yfinance library. 
Yahoo Finance provides legal access to stock market data for personal use.
</p>
<p class='info-text'>
<strong>Disclaimer:</strong> The information provided is for educational purposes only and should not 
be considered as financial advice. Always do your own research before making investment decisions.
</p>
""", unsafe_allow_html=True)