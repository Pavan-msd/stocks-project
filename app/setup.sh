#!/bin/bash
# setup.sh - Installation script for Streamlit Cloud

mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

# Install packages with specific versions that work on Streamlit Cloud
pip install --upgrade pip
pip install streamlit==1.22.0
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install plotly==5.13.0
pip install yfinance==0.2.18