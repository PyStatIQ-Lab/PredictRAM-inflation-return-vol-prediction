import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

# Function to process each stock and calculate predictions
def process_stock(stock_ticker, inflation_changes, portfolio_df):
    data = yf.download(stock_ticker, start="2023-01-01", end="2024-12-31", progress=False)
    stock_data = data[['Close']].reset_index()
    
    # Convert the 'Date' column to Month-Year format (same format as in inflation_df)
    stock_data['Date'] = stock_data['Date'].dt.strftime('%b-%y')

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)

    # Inflation data
    inflation_data = {
        'Date': ['Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24'],
        'Inflation': [6.155075939, 6.16, 5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87, 5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49, 5, 6, 5.5]
    }
    
    inflation_df = pd.DataFrame(inflation_data)

    # Ensure both 'Date' columns are in the same format
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y').dt.strftime('%b-%y')

    # Merge the inflation data with stock data on the 'Date' column
    merged_df = pd.merge(inflation_df, stock_data[['Date', 'Close', 'Volatility']], on='Date', how='inner')

    # Calculate inflation change (month-to-month difference)
    merged_df['Inflation_Change'] = merged_df['Inflation'].diff()

    # Drop NaN values (the first row will have NaN for Inflation_Change)
    merged_df = merged_df.dropna()

    # Prepare features (X) and target variable (y) for TCS.NS closing price prediction
    X_close = merged_df[['Inflation_Change']]  # Inflation change is the feature
    y_close = merged_df['Close']  # TCS.NS closing price is the target variable

    # Train a Linear Regression model for TCS.NS closing price prediction
    close_model = LinearRegression()
    close_model.fit(X_close, y_close)

    # Prepare features (X) and target variable (y) for volatility prediction
    X_volatility = merged_df[['Inflation_Change']]
    y_volatility = merged_df['Volatility']

    # Train a Linear Regression model for volatility prediction
    volatility_model = LinearRegression()
    volatility_model.fit(X_volatility, y_volatility)

    latest_close = stock_data['Close'].iloc[-1]
    previous_close = stock_data['Close'].iloc[-2]
    percentage_change = ((latest_close - previous_close) / previous_close) * 100

    # Get the stock quantity from the portfolio
    stock_quantity = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Quantity'].values[0]
    stock_price = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Price (INR)'].values[0]

    results = []

    for expected_inflation in inflation_changes:
        predicted_user_close = close_model.predict(np.array([[expected_inflation]]))
        predicted_volatility = volatility_model.predict(np.array([[expected_inflation]]))
        predicted_return = ((predicted_user_close[0] - latest_close) / latest_close) * 100

        results.append({
            'Stock': stock_ticker,
            'Quantity': stock_quantity,
            'Stock Price (INR)': stock_price,
            'Inflation Change (%)': expected_inflation,
            'Predicted Close': f'{predicted_user_close[0]:.2f}',
            'Predicted Return (%)': f'{predicted_return:.2f}%',
            'Predicted Volatility': f'{predicted_volatility[0]:.4f}'
        })

    return results


# Streamlit UI
st.title("Stock Prediction Dashboard")

# File uploader for the portfolio Excel file
portfolio_file = st.file_uploader("Upload your Portfolio Excel File", type=["xlsx"])

if portfolio_file:
    portfolio_df = pd.read_excel(portfolio_file)
    stocks = portfolio_df['Stock'].tolist()

    # Get expected inflation changes from user
    inflation_changes = []
    for i in range(3):
        inflation_input = st.number_input(f"Enter expected inflation change for scenario {i+1} (in %):", key=f"inflation_{i+1}")
        inflation_changes.append(inflation_input)

    all_results = []

    # Process each stock in the portfolio and get results
    for stock in stocks:
        stock_results = process_stock(stock, inflation_changes, portfolio_df)
        all_results.extend(stock_results)

    results_df = pd.DataFrame(all_results)

    # Display Prediction Results for All Stocks in Portfolio
    st.subheader("Prediction Results for All Stocks in Portfolio:")
    st.dataframe(results_df)

    # Portfolio predicted return and volatility calculations
    portfolio_results = []
    for inflation_change in inflation_changes:
        total_value = 0
        total_return = 0
        total_volatility = 0
        total_quantity = 0

        for stock_ticker in stocks:
            stock_data = results_df[(results_df['Stock'] == stock_ticker) & (results_df['Inflation Change (%)'] == inflation_change)]

            if not stock_data.empty:
                stock_quantity = stock_data['Quantity'].values[0]
                stock_price = stock_data['Stock Price (INR)'].values[0]
                predicted_return = float(stock_data['Predicted Return (%)'].values[0].replace('%', ''))
                predicted_volatility = float(stock_data['Predicted Volatility'].values[0])

                total_value += stock_quantity * stock_price
                total_return += (predicted_return * stock_quantity * stock_price) / total_value
                total_volatility += (predicted_volatility * stock_quantity * stock_price) / total_value
                total_quantity += stock_quantity

        portfolio_results.append({
            'Inflation Change (%)': inflation_change,
            'Portfolio Predicted Return (%)': f'{total_return:.2f}%',
            'Portfolio Predicted Volatility': f'{total_volatility:.4f}'
        })

    portfolio_results_df = pd.DataFrame(portfolio_results)

    # Display Portfolio Predicted Return and Volatility
    st.subheader("Portfolio Predicted Return and Volatility:")
    st.dataframe(portfolio_results_df)

    # Optionally, save the results to a new Excel file
    save_results = st.button("Save Results to Excel")
    if save_results:
        results_df.to_excel("Client1_portfolio_predictions.xlsx", index=False)
        portfolio_results_df.to_excel("Client1_portfolio_overall_predictions.xlsx", index=False)
        st.success("Results saved successfully!")
