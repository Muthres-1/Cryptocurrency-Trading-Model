import requests
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Step 2: API Data Retrieval
def fetch_crypto_data(crypto_pair, start_date):
    url = f"https://api.binance.com/api/v3/klines"
    start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)  # Convert to milliseconds
    end_timestamp = int(time.time() * 1000)  # Current time in milliseconds
    all_data = []

    while start_timestamp < end_timestamp:
        params = {
            'symbol': crypto_pair,
            'interval': '1d',
            'startTime': start_timestamp,
            'endTime': end_timestamp,
            'limit': 1000  # Maximum limit of rows per request
        }
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

        data = response.json()
        
        if not data:
            print("No more data to fetch.")
            break
        
        all_data.extend(data)

        # Update the start timestamp for the next batch
        start_timestamp = data[-1][0] + 1  # Move to the timestamp after the last retrieved data point

        # To avoid hitting API rate limits, add a slight delay if necessary
        time.sleep(0.1)

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                         'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                                         'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    
    df['Date'] = pd.to_datetime(df['Open Time'], unit='ms').dt.date
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    return df

# Step 3: Calculate Metrics
def calculate_metrics(data, variable1=7, variable2=5):
    data['Date'] = pd.to_datetime(data['Date'])
    data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1).max()
    data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1).min()

    data[f'Days_Since_High_Last_{variable1}_Days'] = data.index.to_series().apply(
        lambda idx: (data.iloc[idx]['Date'] - data['Date'][data['High'][:idx].idxmax()]).days
        if idx > 0 and not data['High'][:idx].empty and data['High'][:idx].idxmax() in data.index[:idx]
        else None
    )
    
    data[f'Days_Since_Low_Last_{variable1}_Days'] = data.index.to_series().apply(
        lambda idx: (data.iloc[idx]['Date'] - data['Date'][data['Low'][:idx].idxmin()]).days
        if idx > 0 and not data['Low'][:idx].empty and data['Low'][:idx].idxmin() in data.index[:idx]
        else None
    )

    data[f'%_Diff_From_High_Last_{variable1}_Days'] = (data['Close'] - data[f'High_Last_{variable1}_Days']) / data[f'High_Last_{variable1}_Days'] * 100
    data[f'%_Diff_From_Low_Last_{variable1}_Days'] = (data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[f'Low_Last_{variable1}_Days'] * 100

    data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2).max()
    data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2).min()

    data[f'%_Diff_From_High_Next_{variable2}_Days'] = (data['Close'] - data[f'High_Next_{variable2}_Days']) / data[f'High_Next_{variable2}_Days'] * 100
    data[f'%_Diff_From_Low_Next_{variable2}_Days'] = (data['Close'] - data[f'Low_Next_{variable2}_Days']) / data[f'Low_Next_{variable2}_Days'] * 100

    return data

# Step 4: Save Data to Google Sheets
def save_to_google_sheets(data, sheet_url):
    # Authenticate with Google Sheets API
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\hp\vscode\python And LIbraries\CV\Trading Project\CryptoProject\json_key_file.json', scope)
    client = gspread.authorize(creds)

    # Extract the spreadsheet ID from the URL
    spreadsheet_id = sheet_url.split('/')[5]

    # Open the Google Sheet
    sheet = client.open_by_key(spreadsheet_id).sheet1

    # Convert DataFrame to a format that can be inserted into Google Sheets
    sheet_data = data.copy()

    # Convert Timestamp to string
    sheet_data['Date'] = sheet_data['Date'].astype(str)

    # Handle infinite and NaN values
    sheet_data.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values with 0
    sheet_data.fillna(0, inplace=True)  # Replace NaN values with 0 (or use any other value)

    # Prepare data for insertion
    rows = [sheet_data.columns.tolist()] + sheet_data.values.tolist()

    # Clear existing content before inserting new data
    sheet.clear()
    sheet.insert_rows(rows, 1)  # Insert all rows at once

# Example usage
if __name__ == "__main__":
    # Step 1: List of crypto pairs to fetch data for
    crypto_pairs = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT", "DOGEUSDT",
    "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "SHIBUSDT",
    "MATICUSDT", "TRXUSDT", "ALGOUSDT", "XLMUSDT", "VETUSDT", "FILUSDT",
    "ICPUSDT", "AAVEUSDT", "ETCUSDT", "LUNAUSDT", "SANDUSDT", "CHZUSDT",
    "MANAUSDT", "BATUSDT", "FETUSDT", "ZRXUSDT", "RAVENUSDT", "COMPUSDT",
    "DASHUSDT", "NEOUSDT", "QTUMUSDT", "ZILUSDT", "NANOUSDT", "WAVESUSDT",
    "XEMUSDT", "LTCBTC", "XRPBTC", "DOGEBTC", "BTCEUR", "BTCJPY",
    "BTCAUD", "BTCGBP", "BTCCHF", "BTCUSDC", "ETHBTC", "BNBBTC",
    "LTCBTC", "XRPETH", "ADABTC", "DOTBTC", "LINKBTC", "MATICBTC",
    "TRXBTC", "SOLBTC", "AVAXBTC", "SHIBBTC", "SANDBTC", "CHZBTC",
    "MANABTC", "AAVEBTC", "ETCBTC", "LUNABTC", "DASHBTC", "QTUMBTC",
    "FILBTC", "XLMBTC", "VETBTC", "NEOBTC", "ALGBTC", "XEMBTC"
]
    start_date = "2020-01-01"  # Start date for data retrieval

    all_metrics_data = []

    for crypto_pair in crypto_pairs:
        print(f"Fetching data for {crypto_pair}...")

        # Step 2: API Data Retrieval
        metrics_data = fetch_crypto_data(crypto_pair, start_date)

        # Step 3: Calculate Metrics if data retrieval was successful
        if metrics_data is not None:
            metrics_data = calculate_metrics(metrics_data, variable1=7, variable2=5)

            # Append the crypto pair information for distinction
            metrics_data['Crypto Pair'] = crypto_pair
            all_metrics_data.append(metrics_data)

    # Combine all data into a single DataFrame if needed
    combined_data = pd.concat(all_metrics_data, ignore_index=True)

    # Save to Google Sheets
    save_to_google_sheets(combined_data, "https://docs.google.com/spreadsheets/d/1zd39JS4JIOvQBBXoj5VbPIoR6nQ4Ei7uuQir1tAehZg/edit?gid=0#gid=0")
