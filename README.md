# MacroRisk Forecaster

A comprehensive tool for forecasting macroeconomic risk factors and analyzing their impact on financial markets. This package provides an integrated workflow from data acquisition to risk analysis and visualization.

## Features
- **Data Acquisition**: Fetch economic data from FRED (Federal Reserve Economic Database) with built-in support for common indicators
- **Time Series Analysis**: Implement ARIMA, exponential smoothing, and other forecasting models
- **Risk Assessment**: Calculate Value at Risk (VaR) using multiple methods (historical, parametric, EWMA)
- **Statistical Tools**: Stationarity tests, data transformation, and outlier detection
- **Visualization**: Plotting utilities for forecasts and risk metrics

## Installation
```bash
git clone https://github.com/yourusername/macrorisk-forecaster.git
cd macrorisk-forecaster
pip install -e .
```

## Configuration
Create a `.env` file in the root directory with your FRED API key:
```
FRED_API_KEY=your_api_key_here
```
You can obtain a free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html).

## Usage

### Data Acquisition
```python
from src.data.fred_connector import FREDConnector

# Initialize connector
fred = FREDConnector()

# Get common indicators
inflation = fred.get_common_indicator('inflation', frequency='m')
gdp = fred.get_common_indicator('gdp', frequency='q')

# Get multiple indicators at once
macro_data = fred.get_multiple_indicators(['gdp', 'unemployment', 'fed_funds'])
```

### Time Series Forecasting
```python
from src.models.time_series import arima_forecast, plot_forecast

# Create and plot a forecast
forecast_results = arima_forecast(gdp, forecast_periods=8)
plot_forecast(forecast_results, title="GDP Forecast")
```

### Risk Analysis
```python
from src.risk.var_calculator import calculate_var, plot_var

# Calculate Value at Risk
var = calculate_var(returns, confidence=0.95, method='historical')

# Visualize VaR
plot_var(returns, confidence=0.95)
```

## Project Structure
```
macrorisk-forecaster/
│
├── src/
│   ├── data/            # Data acquisition modules
│   ├── models/          # Forecasting models
│   ├── risk/            # Risk calculation tools
│   └── visualization/   # Visualization utilities
│
├── notebooks/          # Example Jupyter notebooks
├── tests/              # Unit tests
└── data/               # Data storage directory
```

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
