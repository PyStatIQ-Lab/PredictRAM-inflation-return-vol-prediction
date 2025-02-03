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

    # Convert 'Date' column to Month-Year format to match the inflation data format
    stock_data['Date'] = stock_data['Date'].dt.strftime('%b-%y').str.strip()  # Remove extra spaces

    # Calculate daily returns and volatility (rolling 30-day window)
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)  # Annualized volatility

    # Inflation data
    inflation_data = {
        'Date': ['Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24'],
        'Inflation': [6.155075939, 6.16, 5.793650794, 5.090054816, 4.418604651, 5.572755418, 7.544264819, 6.912442396, 5.02, 4.87, 5.55, 5.69, 5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49, 5, 6, 5.5]
    }
    
    inflation_df = pd.DataFrame(inflation_data)

    # Convert 'Date' column to Month-Year format in inflation_df
    inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y').dt.strftime('%b-%y').str.strip()

    # Debugging: Check unique Date values to ensure they match
    st.write("Unique Dates in Stock Data:", stock_data['Date'].unique())
    st.write("Unique Dates in Inflation Data:", inflation_df['Date'].unique())

    # Merge the inflation data with stock data on the 'Date' column
    # Using 'outer' merge first to inspect matching rows
    merged_df = pd.merge(inflation_df, stock_data[['Date', 'Close', 'Volatility']], on='Date', how='outer')

    # Check if there are unmatched rows
    unmatched_rows = merged_df[merged_df.isnull().any(axis=1)]
    if not unmatched_rows.empty:
        st.write("There are unmatched rows due to missing or unmatched dates:")
        st.write(unmatched_rows)

    # Proceed with cleaning up merged data (if there are missing values)
    merged_df = merged_df.dropna()

    # Calculate inflation change (month-to-month difference)
    merged_df['Inflation_Change'] = merged_df['Inflation'].diff()

    # Drop NaN values (the first row will have NaN for Inflation_Change)
    merged_df = merged_df.dropna()

    # Prepare features (X) and target variable (y) for stock closing price prediction
    X_close = merged_df[['Inflation_Change']]  # Inflation change is the feature
    y_close = merged_df['Close']  # Stock closing price is the target variable

    # Train a Linear Regression model for stock closing price prediction
    close_model = LinearRegression()
    close_model.fit(X_close, y_close)

    # Prepare features (X) and target variable (y) for volatility prediction
    X_volatility = merged_df[['Inflation_Change']]
    y_volatility = merged_df['Volatility']

    # Train a Linear Regression model for volatility prediction
    volatility_model = LinearRegression()
    volatility_model.fit(X_volatility, y_volatility)

    # Get the latest stock close price and calculate the percentage change
    latest_close = stock_data['Close'].iloc[-1]
    previous_close = stock_data['Close'].iloc[-2]
    percentage_change = ((latest_close - previous_close) / previous_close) * 100

    # Get the stock quantity from the portfolio
    stock_quantity = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Quantity'].values[0]
    stock_price = portfolio_df[portfolio_df['Stock'] == stock_ticker]['Price (INR)'].values[0]

    results = []

    # Predict the stock closing price and volatility for each inflation change
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
st.title('Stock Portfolio Prediction Based on Inflation Changes')

# Read the portfolio data from the user input file
portfolio_file = st.file_uploader("Upload your portfolio Excel file", type="xlsx")

if portfolio_file is not None:
    portfolio_df = pd.read_excel(portfolio_file)

    # Get the list of stocks from the portfolio
    stocks = portfolio_df['Stock'].tolist()

    # Ask for expected inflation changes (3 scenarios)
    inflation_changes = []
    for i in range(3):
        inflation_input = st.number_input(f'Enter expected inflation change for scenario {i+1} (in %): ', format="%.2f")
        inflation_changes.append(inflation_input)

    # Prepare a list to store all results
    all_results = []

    # Process each stock in the portfolio
    for stock in stocks:
        stock_results = process_stock(stock, inflation_changes, portfolio_df)
        all_results.extend(stock_results)

    # Convert results into a DataFrame for table display
    results_df = pd.DataFrame(all_results)

    # Display the results in a table format
    st.write("Prediction Results for All Stocks in Portfolio:")
    st.dataframe(results_df)

    # Calculate portfolio predicted return and volatility
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

    # Convert portfolio results to DataFrame
    portfolio_results_df = pd.DataFrame(portfolio_results)

    # Display portfolio results
    st.write("Portfolio Predicted Return and Volatility:")
    st.dataframe(portfolio_results_df)

    # Optionally, save the results to a new Excel file
    st.download_button(
        label="Download Prediction Results",
        data=results_df.to_excel(index=False),
        file_name="Client1_portfolio_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        label="Download Portfolio Results",
        data=portfolio_results_df.to_excel(index=False),
        file_name="Client1_portfolio_overall_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
