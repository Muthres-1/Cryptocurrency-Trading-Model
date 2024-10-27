# Cryptocurrency Trading Model

## Overview

This project implements a cryptocurrency trading model that fetches historical data from the Binance API, calculates key metrics, trains linear regression models for price prediction, and saves the results to Google Sheets.

## Features

@ Data Retrieval: Fetch daily candlestick data for multiple cryptocurrency pairs.  
@ Metric Calculation: Calculate important metrics such as highs, lows, and percentage differences.  
@ Model Training: Train linear regression models to predict future price movements.  
@ Google Sheets Integration: Save calculated data and model results directly to Google Sheets for easy access and sharing.

## Prerequisites

@ Python 3.7 or later  
@ Required Python packages:  
  - `pandas`  
  - `requests`  
  - `gspread`  
  - `scikit-learn`  
  - `joblib`  

You can install the required packages using pip:

```bash
pip install pandas requests gspread scikit-learn joblib
```
## Setup Instructions
### 1. Clone the Repository
@ Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/crypto-trading-model.git
cd crypto-trading-model
```
### 2. Create Google Sheets API Credentials
@ Follow these steps to create your Google Sheets API credentials:

Go to the Google Cloud Console.
Create a new project.
Enable the Google Sheets API.
Create credentials (Service Account).
Download the JSON file containing your credentials and save it in the project directory.
### 3. Modify the Code
@ Update the DataUsingBinance.py with your desired cryptocurrency pairs and the start date for data fetching.
@ In trainModel.py, configure the paths for saving the model and loading data from Google Sheets.

### 4. Run the Scripts
@ To fetch data and calculate metrics, run the following command:

```bash
python DataUsingBinance.py
```
@ To train the models, execute:

``` bash
python trainModel.py
```
@ To make predictions (this part needs to be completed in prediction.py), run:

``` bash
python prediction.py
```
## Usage
@ DataUsingBinance.py: This script fetches data from Binance, calculates metrics, and saves the data to Google Sheets.
@ trainModel.py: This script loads the data from Google Sheets, trains linear regression models, and saves the models.
@ prediction.py: This script (to be completed) will fetch new data, load trained models, and make predictions.

## Contributing
@ Feel free to fork the repository, make changes, and submit a pull request. Contributions are welcome!

License
@ This project is licensed under the MIT License.

Acknowledgments
@ Binance API Documentation
@ Google Sheets API Documentation
