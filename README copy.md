# MacroRisk Forecaster

A comprehensive tool for forecasting macroeconomic risk factors and analyzing their impact on financial markets.

## Features

- Data acquisition from various sources (FRED, World Bank, market data)
- Time series forecasting models for macroeconomic indicators
- Machine learning models for risk prediction
- Risk calculations including Value at Risk (VaR) and Monte Carlo simulations
- Interactive dashboard for visualizing forecasts and risk metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/macrorisk-forecaster.git
cd macrorisk-forecaster

# Install the package and dependencies
pip install -e .
```

## Usage

```python
# Example usage
from src.data import fred_connector
from src.models import time_series
from src.risk import var_calculator

# Get economic data
gdp_data = fred_connector.get_series('GDP')

# Create a forecast
forecast = time_series.arima_forecast(gdp_data)

# Calculate risk metrics
var = var_calculator.calculate_var(forecast, confidence=0.95)
```

## Dashboard

Run the dashboard with:

```bash
python -m src.visualization.dashboard
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
