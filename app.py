import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import streamlit as st

# Function to process each stock and calculate predictions
def process_stock(stock_ticker, inflation_changes, portfolio_df):
    # Fetch stock data from Yahoo Finance
    data = yf.download(stock_ticker, start="2023-01-01", end="2024-12-31", progress=False)

    # Extract the 'Close' prices and reset the index
    stock_data = data[['Close']].reset_index()

    # Format the 'Date' column to match the inflation data
    stock_data['Date'] = stock_data['Date'].dt.strftime('%b-%y')

    # Calculate daily returns of the stock
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()

    # Calculate rolling volatility (standard deviation of returns over a 30-day window)
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)  # Annualized volatility

    # Merge the inflation data with stock closing data on the 'Date' column
    inflation_data = {
        'Date': ['Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24'],
        'Inflation': [6.155075939, 6.16, 5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87, 5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49, 5, 6, 5.5]
    }
    inflation_df = pd.DataFrame(inflation_data)
    
    # Merge the inflation data with the stock data
    merged_df = pd.merge(inflation_df, stock_data[['Date', 'Close', 'Volatility']], on='Date')

    # Calculate inflation change (month-to-month difference)
    merged_df['Inflation_Change'] = merged_df['Inflation'].diff()

    # Drop NaN values (the first row will have NaN for Inflation_Change)
    merged_df = merged_df.dropna()

    # Prepare the features (X) and target variable (y) for TCS.NS closing price prediction
    X_close = merged_df[['Inflation_Change']]  # Inflation change is the feature
    y_close = merged_df['Close']  # TCS.NS closing price is the target variable

    # Train a Linear Regression model for TCS.NS closing price prediction
    close_model = LinearRegression()
    close_model.fit(X_close, y_close)

    # Prepare the features (X) and target variable (y) for volatility prediction
    X_volatility = merged_df[['Inflation_Change']]
    y_volatility = merged_df['Volatility']

    # Train a Linear Regression model for volatility prediction
    volatility_model = LinearRegression()
    volatility_model.fit(X_volatility, y_volatility)

    # Get the latest stock close price and calculate the percentage change from the previous close
    latest_close = stock_data['Close'].iloc[-1]
    previous_close = stock_data['Close'].iloc[-2]
    percentage_change = ((latest_close - previous_close) / previous_close) * 100

    # Get the stock quantity from the portfolio
    stock_quantity = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Quantity'].values[0]
    stock_price = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Price (INR)'].values[0]

    # Prepare a list to store the results
    results = []

    # Predict the stock closing price and volatility for each inflation change
    for expected_inflation in inflation_changes:
        # Predict the stock closing price based on inflation change
        predicted_user_close = close_model.predict(np.array([[expected_inflation]]))  # Ensure input is in correct format

        # Predict the volatility based on inflation change
        predicted_volatility = volatility_model.predict(np.array([[expected_inflation]]))  # Ensure input is in correct format

        # Calculate the predicted return (percentage change) from the latest close to predicted close
        predicted_return = ((predicted_user_close[0] - latest_close) / latest_close) * 100

        # Append results to the list
        results.append({
            'Stock': stock_ticker,
            'Quantity': stock_quantity,
            'Stock Price (INR)': stock_price,
            'Inflation Change (%)': expected_inflation,
            'Predicted Close': f'{predicted_user_close[0]:.2f}',
            'Predicted Return (%)': f'{predicted_return:.2f}%',
            'Predicted Volatility': f'{predicted_volatility[0]:.4f}'
        })

    # Return the results
    return results

# Function to calculate portfolio-level predicted return and volatility
def calculate_portfolio_results(all_results, inflation_changes, portfolio_df):
    portfolio_results = []
    
    for inflation_change in inflation_changes:
        total_value = 0
        total_return = 0
        total_volatility = 0
        total_quantity = 0

        for stock_result in all_results:
            if stock_result['Inflation Change (%)'] == inflation_change:
                stock_quantity = stock_result['Quantity']
                stock_price = stock_result['Stock Price (INR)']
                predicted_return = float(stock_result['Predicted Return (%)'].replace('%', ''))
                predicted_volatility = float(stock_result['Predicted Volatility'])

                total_value += stock_quantity * stock_price
                total_return += (predicted_return * stock_quantity * stock_price) / total_value
                total_volatility += (predicted_volatility * stock_quantity * stock_price) / total_value
                total_quantity += stock_quantity

        portfolio_results.append({
            'Inflation Change (%)': inflation_change,
            'Portfolio Predicted Return (%)': f'{total_return:.2f}%',
            'Portfolio Predicted Volatility': f'{total_volatility:.4f}'
        })

    return portfolio_results

# Streamlit application
def main():
    st.title("Stock Prediction Dashboard")

    # Read the portfolio data for multiple clients
    client_files = ['Client1_portfolio.xlsx', 'Client2_portfolio.xlsx', 'Client3_portfolio.xlsx']
    inflation_changes = [0.5, 1.0, 1.5]  # Example inflation changes (you can modify this)
    
    # Display and process each client's data
    for client_file in client_files:
        portfolio_df = pd.read_excel(client_file)
        client_name = client_file.split('_')[0]  # Extract client name (e.g., Client1)
        stocks = portfolio_df['Stock'].tolist()
        all_results = []

        for stock in stocks:
            stock_results = process_stock(stock, inflation_changes, portfolio_df)
            all_results.extend(stock_results)

        # Calculate portfolio-level results
        portfolio_results = calculate_portfolio_results(all_results, inflation_changes, portfolio_df)

        # Display results for the client
        st.subheader(f"Prediction Results for {client_name}")
        
        st.markdown("### Stock Predictions")
        st.write(pd.DataFrame(all_results))

        st.markdown("### Portfolio Predicted Return and Volatility")
        st.write(pd.DataFrame(portfolio_results))

if __name__ == '__main__':
    main()
