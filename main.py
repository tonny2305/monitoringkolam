import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import json

# Function to fetch data from ThingSpeak
def fetch_data(api_key, channel_id, results=100):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results={results}"
    response = requests.get(url)
    data = json.loads(response.text)
    feeds = data['feeds']
    
    # Convert data to DataFrame
    df = pd.DataFrame(feeds)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    df['field1'] = pd.to_numeric(df['field1'], errors='coerce')
    df['field2'] = pd.to_numeric(df['field2'], errors='coerce')
    
    # Drop rows with NaN values
    df.dropna(subset=['field1', 'field2'], inplace=True)
    
    # Add frequency information
    df = df.asfreq('T')  # Assuming data is in minutes. Adjust accordingly.
    
    return df[['field1', 'field2']]

# Function to apply ARIMA model
def apply_arima(data, field, order=(5, 1, 0)):
    model = ARIMA(data[field], order=order)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=10)
    
    return forecast.predicted_mean

# Fetch the data
api_key = 'FDFVA17KW5YBEPYH'
channel_id = '2590437'
data = fetch_data(api_key, channel_id)

# Apply ARIMA model
temperature_forecast = apply_arima(data, 'field1')
ph_forecast = apply_arima(data, 'field2')

# Plotting the results
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['field1'], label='Temperature')
plt.plot(temperature_forecast.index, temperature_forecast, label='Temperature Forecast', color='red')
plt.legend()
plt.title('Temperature Forecast')

plt.subplot(2, 1, 2)
plt.plot(data.index, data['field2'], label='pH')
plt.plot(ph_forecast.index, ph_forecast, label='pH Forecast', color='red')
plt.legend()
plt.title('pH Forecast')

plt.tight_layout()

# Save the figure if interactive display is not possible
plt.savefig('forecast.png')

# If running in an environment that supports GUI backends, comment the above line and uncomment the next one:
# plt.show()
