import requests
import pandas as pd
import time
import joblib
import os
import logging
# Step 2: API Data Retrieval
# Set up logging
logging.basicConfig(level=logging.INFO)
def fetch_crypto_data(crypto_pair, start_date, retries=5):
    url = "https://api.binance.com/api/v3/klines"
    start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_timestamp = int(time.time() * 1000)
    all_data = []

    for attempt in range(retries):
        try:
            while start_timestamp < end_timestamp:
                params = {
                    'symbol': crypto_pair,
                    'interval': '1d',
                    'startTime': start_timestamp,
                    'endTime': end_timestamp,
                    'limit': 1000
                }
                response = requests.get(url, params=params)
                response.raise_for_status()  # Raise an error for bad responses

                data = response.json()
                if not data:
                    print("No more data to fetch.")
                    break

                all_data.extend(data)
                start_timestamp = data[-1][0] + 1
                time.sleep(0.1)  # Rate limit handling

            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[...])  # Use appropriate column names
            return df

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying

    return None  # Return None if all attempts fail


# Step 3: Calculate Metrics
def calculate_metrics(data, variable1, variable2):
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

# Step 4: Prediction using Saved Models
def predict_outcomes(model_high, model_low, feature_values):
    feature_columns = [
        'Days_Since_High_Last_7_Days',
        '%_Diff_From_High_Last_7_Days',
        'Days_Since_Low_Last_7_Days',
        '%_Diff_From_Low_Last_7_Days'
    ]
    
    features = pd.DataFrame([feature_values], columns=feature_columns)
    high_prediction = model_high.predict(features)
    low_prediction = model_low.predict(features)
    return high_prediction[0], low_prediction[0]

# Example Usage
if __name__ == "__main__":
    crypto_data = fetch_crypto_data('BTCUSDT', '2023-01-01')
    if crypto_data is not None:
        metrics_data = calculate_metrics(crypto_data, variable1=7, variable2=5)
        
        # File paths for loading saved models
        model_high_path = "model_high1.joblib"
        model_low_path = "model_low1.joblib"

        # Load pre-trained models
        if os.path.exists(model_high_path) and os.path.exists(model_low_path):
            model_high = joblib.load(model_high_path)
            model_low = joblib.load(model_low_path)
            print("Loaded saved models.")
        else:
            print("Model files not found. Please ensure the models are trained and saved as 'model_high.joblib' and 'model_low.joblib'.")

        # Define feature values (Example)
        feature_values = [1, -2.5, 3, 1.5]

        # Predict outcomes
        high_pred, low_pred = predict_outcomes(model_high, model_low, feature_values)
        
        # Display predictions
        print(f"Predicted % Difference from High: {high_pred:.2f}%")
        print(f"Predicted % Difference from Low: {low_pred:.2f}%")
