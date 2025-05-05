import pickle
import json

# Load the old pickle file
forecast_model = pickle.load(open('models/trend_forecast_model.pkl', 'rb'))

# If it's a pandas Series, convert to dictionary
if hasattr(forecast_model, 'to_dict'):
    forecast_model = forecast_model.to_dict()

# Clean and flatten structure
forecast_model_clean = {}

for k, v in forecast_model.items():
    if isinstance(v, dict):
        # If value is dict, get first element
        inner_value = list(v.values())[0]
        
        # If inner_value is still Series, get first number
        if hasattr(inner_value, 'values'):
            inner_value = inner_value.values[0]
        
        forecast_model_clean[str(k)] = int(inner_value)
    elif hasattr(v, 'values'):
        # If v itself is Series, get first number
        forecast_model_clean[str(k)] = int(v.values[0])
    else:
        forecast_model_clean[str(k)] = int(v)

# Save as JSON
with open('models/trend_forecast_model.json', 'w') as f:
    json.dump(forecast_model_clean, f)
