import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# Step 5: Retrieve Data from Google Sheets
def fetch_data_from_google_sheets(sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\hp\vscode\python And LIbraries\CV\Trading Project\CryptoProject\json_key_file.json', scope)
    client = gspread.authorize(creds)
    
    sheet = client.open_by_url(sheet_url).sheet1
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])  # First row as header
    df = df.apply(pd.to_numeric, errors='ignore')  # Convert numeric columns
    return df

# Step 6: Machine Learning Model
def train_model(data):
    features = [
        'Days_Since_High_Last_7_Days',
        '%_Diff_From_High_Last_7_Days',
        'Days_Since_Low_Last_7_Days',
        '%_Diff_From_Low_Last_7_Days'
    ]
    
    target_high = '%_Diff_From_High_Next_5_Days'
    target_low = '%_Diff_From_Low_Next_5_Days'
    
    # Drop rows with NaNs in any of the selected features or target columns
    data = data.dropna(subset=features + [target_high, target_low])

    X = data[features]
    y_high = data[target_high]
    y_low = data[target_low]

    # Train/test split for both high and low targets
    X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
    _, _, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

    # Train the model
    model_high = LinearRegression().fit(X_train, y_train_high)
    model_low = LinearRegression().fit(X_train, y_train_low)

    joblib.dump(model_high, 'model_high1.pkl')
    joblib.dump(model_low, 'model_low1.pkl')

    return model_high, model_low

# Example Usage for Training
if __name__ == "__main__":
    sheet_url = "https://docs.google.com/spreadsheets/d/1zd39JS4JIOvQBBXoj5VbPIoR6nQ4Ei7uuQir1tAehZg/edit?gid=0"
    
    # Fetch metrics data
    metrics_data = fetch_data_from_google_sheets(sheet_url)
    
    # Train models
    model_high, model_low = train_model(metrics_data)
    print("Models trained and saved.")
