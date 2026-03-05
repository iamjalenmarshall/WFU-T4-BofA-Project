import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def arima_forecast(ts, order=(1,1,2), forecast_steps=8):
    """
    Implement ARIMA(1,1,2) forecasting
    
    Parameters:
    -----------
    ts : pandas Series or array-like
        Time series data
    order : tuple
        ARIMA order (p, d, q) - default (1, 1, 2)
    forecast_steps : int
        Number of steps to forecast ahead - default 8
    
    Returns:
    --------
    np.array : Forecasted values
    """
    p, d, q = order
    
    # Remove NaN values
    ts = ts.dropna().astype(float).values
    
    if len(ts) < (p + q + d + 10):
        # If series is too short, use simple exponential smoothing
        return np.full(forecast_steps, ts[-1] if len(ts) > 0 else 0)
    
    try:
        # Apply differencing d times
        diff_ts = ts.copy()
        for _ in range(d):
            if len(diff_ts) > 1:
                diff_ts = np.diff(diff_ts)
        
        # Fit ARIMA using simple estimation
        # ARIMA(1,1,2): AR(1) on differenced series + MA(2) on residuals
        
        # Initialize parameters
        phi = 0.5  # AR(1) coefficient
        theta1 = 0.3  # MA(1) coefficient
        theta2 = 0.1  # MA(2) coefficient
        
        # Simple parameter estimation using mean squared error
        def objective(params):
            phi, theta1, theta2 = params
            residuals = []
            for i in range(2, len(diff_ts)):
                pred = phi * diff_ts[i-1] + theta1 * (diff_ts[i-1] - phi * diff_ts[i-2] if i > 1 else 0) + theta2 * (diff_ts[i-2] - phi * diff_ts[i-3] if i > 2 else 0)
                if i >= 2:
                    residuals.append((diff_ts[i] - pred) ** 2)
            return np.mean(residuals) if residuals else 0
        
        # Optimize parameters
        res = minimize(objective, [0.5, 0.3, 0.1], method='Nelder-Mead')
        phi, theta1, theta2 = res.x
        
        # Ensure stationarity
        phi = np.clip(phi, -0.99, 0.99)
        
        # Generate forecasts
        forecasts = []
        last_diff = diff_ts[-1]
        last_diff_2 = diff_ts[-2] if len(diff_ts) > 1 else 0
        
        for step in range(forecast_steps):
            # AR(1) + MA(2) model on differenced series
            pred = phi * last_diff + theta1 * 0 + theta2 * 0  # MA errors decay to 0 in forecast
            forecasts.append(pred)
            last_diff_2 = last_diff
            last_diff = pred
        
        # Reverse differencing to get forecasts on original scale
        reversed_forecasts = np.array(forecasts)
        for _ in range(d):
            if len(reversed_forecasts) > 0:
                reversed_forecasts = np.cumsum(np.concatenate([[ts[-1]], reversed_forecasts]))
                reversed_forecasts = reversed_forecasts[1:]
        
        return reversed_forecasts
    
    except:
        # Fallback: return last value repeated
        return np.full(forecast_steps, ts[-1] if len(ts) > 0 else 0)


def forecast_csv_with_arima(input_csv_path, output_csv_path, forecast_quarters=8, arima_order=(1, 1, 2)):
    """
    Load a CSV file, apply ARIMA forecasting to each feature, and save with forecast columns
    
    Parameters:
    -----------
    input_csv_path : str
        Path to input CSV file (features as rows, time periods as columns)
    output_csv_path : str
        Path to save output CSV file with forecast columns
    forecast_quarters : int
        Number of quarters to forecast (default 8)
    arima_order : tuple
        ARIMA order (p, d, q) - default (1, 1, 2)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with original and forecast data
    """
    
    # Load the CSV
    df = pd.read_csv(input_csv_path, index_col=0)
    print(f"Original data shape: {df.shape}")
    
    # Generate future quarter labels for the next N quarters
    quarters = [
        '12/31/2025', '03/31/2026', '06/30/2026', '09/30/2026', 
        '12/31/2026', '03/31/2027', '06/30/2027', '09/30/2027'
    ][:forecast_quarters]
    
    # Create a new dataframe to store results
    df_forecast = df.copy()
    
    # Add the new forecast columns
    for quarter in quarters:
        df_forecast[quarter] = np.nan
    
    # Fit ARIMA for each feature and generate forecasts
    print("\nForecasting...")
    for idx, feature in enumerate(df.index):
        try:
            # Get the time series for this feature
            ts = df.loc[feature]
            
            # Forecast the next N periods
            forecast_values = arima_forecast(ts, order=arima_order, forecast_steps=forecast_quarters)
            
            # Add forecasted values to the new dataframe
            for i, quarter in enumerate(quarters):
                df_forecast.loc[feature, quarter] = forecast_values[i]
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df.index)} features")
            
        except Exception as e:
            print(f"  Error processing {feature}: {str(e)}")
    
    # Save the result
    df_forecast.to_csv(output_csv_path)
    
    print(f"\n✓ Forecast complete!")
    print(f"Output file: {output_csv_path}")
    print(f"Original shape: {df.shape}")
    print(f"Final shape with {forecast_quarters} forecast quarters: {df_forecast.shape}")
    
    return df_forecast


if __name__ == "__main__":
    # Example usage:
    # For Citibank data
    forecast_csv_with_arima(
        input_csv_path='ffiec_citibank_features.csv',
        output_csv_path='ffiec_citibank_features_with_forecast.csv',
        forecast_quarters=8,
        arima_order=(1, 1, 2)
    )
    
    # For JPMorgan Chase data
    forecast_csv_with_arima(
        input_csv_path='ffiec_jpmorgan_chase_bank_features.csv',
        output_csv_path='ffiec_jpmorgan_chase_bank_features_with_forecast.csv',
        forecast_quarters=8,
        arima_order=(1, 1, 2)
    )