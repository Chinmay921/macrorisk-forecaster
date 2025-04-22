"""
Time series forecasting module.

This module provides various time series models and forecasting techniques 
for macroeconomic indicators and financial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_stationarity(series, window=12):
    """
    Check if a time series is stationary using rolling statistics and ADF test.
    
    Parameters
    ----------
    series : pandas.Series
        The time series to check for stationarity.
    window : int, optional
        The window size for rolling statistics.
        
    Returns
    -------
    bool
        True if the series is stationary, False otherwise.
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Check input
    if not isinstance(series, pd.Series):
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 1:
                series = series.iloc[:, 0]
            else:
                logger.error("Input must be a pandas Series or single-column DataFrame")
                return False
        else:
            logger.error("Input must be a pandas Series or DataFrame")
            return False
    
    # Perform ADF test
    result = adfuller(series.dropna())
    adf_pvalue = result[1]
    
    # Get rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Plot rolling statistics
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series, label='Original')
    ax.plot(rolling_mean, label='Rolling Mean')
    ax.plot(rolling_std, label='Rolling Std')
    ax.legend()
    ax.set_title(f'Rolling Mean & Standard Deviation (p-value: {adf_pvalue:.4f})')
    
    # Log results
    logger.info(f"ADF Test p-value: {adf_pvalue}")
    logger.info("ADF Test Result:")
    logger.info(f"Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")
    logger.info(f"Critical Values: {result[4]}")
    
    # Return stationarity result (p-value < 0.05 indicates stationarity)
    return adf_pvalue < 0.05


def make_stationary(series, method='diff', order=1):
    """
    Transform a time series to make it stationary.
    
    Parameters
    ----------
    series : pandas.Series
        The time series to transform.
    method : str, optional
        Transformation method: 'diff' for differencing, 'log' for log transform,
        'log_diff' for log transform followed by differencing.
    order : int, optional
        Order of differencing if using 'diff' or 'log_diff'.
        
    Returns
    -------
    pandas.Series
        The transformed stationary series.
    dict
        Transformation parameters used (for later inversion).
    """
    if not isinstance(series, pd.Series) and not isinstance(series, pd.DataFrame):
        logger.error("Input must be a pandas Series or DataFrame")
        return None, None
    
    transform_params = {'method': method, 'order': order}
    transformed = None
    
    if method == 'diff':
        transformed = series.diff(order).dropna()
    elif method == 'log':
        # Ensure all values are positive for log transform
        if (series <= 0).any().any() if isinstance(series, pd.DataFrame) else (series <= 0).any():
            logger.warning("Series contains non-positive values. Adding minimum value offset.")
            min_val = series.min().min() if isinstance(series, pd.DataFrame) else series.min()
            offset = abs(min_val) + 1 if min_val <= 0 else 0
            transform_params['offset'] = offset
            transformed = np.log(series + offset)
        else:
            transformed = np.log(series)
    elif method == 'log_diff':
        # Log transform first
        if (series <= 0).any().any() if isinstance(series, pd.DataFrame) else (series <= 0).any():
            logger.warning("Series contains non-positive values. Adding minimum value offset.")
            min_val = series.min().min() if isinstance(series, pd.DataFrame) else series.min()
            offset = abs(min_val) + 1 if min_val <= 0 else 0
            transform_params['offset'] = offset
            log_series = np.log(series + offset)
        else:
            log_series = np.log(series)
        # Then difference
        transformed = log_series.diff(order).dropna()
    else:
        logger.error(f"Unknown transformation method: {method}")
        return None, None
    
    # Check if the transformed series is stationary
    is_stationary = check_stationarity(transformed) if isinstance(transformed, pd.Series) else False
    transform_params['is_stationary'] = is_stationary
    
    if not is_stationary:
        logger.warning("Series is still non-stationary after transformation.")
    
    return transformed, transform_params


def inverse_transform(forecasted_series, transform_params, original_series=None):
    """
    Invert the transformation applied to make a series stationary.
    
    Parameters
    ----------
    forecasted_series : pandas.Series
        The forecasted stationary series.
    transform_params : dict
        Transformation parameters from make_stationary function.
    original_series : pandas.Series, optional
        The original series (needed for some transformations).
        
    Returns
    -------
    pandas.Series
        The inverse-transformed forecasted series.
    """
    method = transform_params.get('method')
    order = transform_params.get('order', 1)
    offset = transform_params.get('offset', 0)
    
    if method == 'diff':
        if original_series is None:
            logger.error("Original series is required for inverse differencing")
            return None
        
        last_values = original_series.iloc[-order:].values if order > 1 else original_series.iloc[-1]
        # Implement cumulative sum for the forecast
        inv_forecast = pd.Series(last_values).append(forecasted_series).cumsum().iloc[1:]
        return inv_forecast
        
    elif method == 'log':
        # Inverse of log is exp, then subtract the offset if any
        return np.exp(forecasted_series) - offset
        
    elif method == 'log_diff':
        if original_series is None:
            logger.error("Original series is required for inverse log differencing")
            return None
            
        # First get log of original
        log_original = np.log(original_series + offset)
        
        # Get last values for differencing
        last_values = log_original.iloc[-order:].values if order > 1 else log_original.iloc[-1]
        
        # Undo differencing (cumsum)
        inv_diff = pd.Series(last_values).append(forecasted_series).cumsum().iloc[1:]
        
        # Undo log transform
        return np.exp(inv_diff) - offset
    
    else:
        logger.error(f"Unknown transformation method: {method}")
        return None


def arima_forecast(series, forecast_periods=12, seasonal=False, 
                   auto=True, order=None, seasonal_order=None):
    """
    Create an ARIMA or SARIMA forecast for a time series.
    
    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame
        The time series to forecast.
    forecast_periods : int, optional
        Number of periods to forecast.
    seasonal : bool, optional
        Whether to use seasonal ARIMA (SARIMA).
    auto : bool, optional
        Whether to automatically select the best order using auto_arima.
    order : tuple, optional
        The (p,d,q) order of the ARIMA model if auto=False.
    seasonal_order : tuple, optional
        The (P,D,Q,s) seasonal order of the SARIMA model if seasonal=True and auto=False.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original series, the fitted values, and the forecast.
    statsmodels.tsa.arima.model.ARIMAResults or statsmodels.tsa.statespace.sarimax.SARIMAXResults
        The fitted model object.
    """
    # Handle input
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            logger.error("Input must be a pandas Series or single-column DataFrame")
            return None, None
    
    # Check for missing values
    if series.isna().any():
        logger.warning("Series contains missing values. Interpolating...")
        series = series.interpolate(method='linear')
    
    # Auto-detect best parameters if requested
    if auto:
        logger.info("Auto-selecting best ARIMA parameters...")
        
        # Use auto_arima to find best parameters
        auto_model = auto_arima(
            series,
            seasonal=seasonal,
            m=12 if seasonal else None,  # Assuming monthly data if seasonal
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=None,
            trace=True
        )
        
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order if seasonal else None
        
        logger.info(f"Best ARIMA order: {order}")
        if seasonal:
            logger.info(f"Best seasonal order: {seasonal_order}")
    
    # Default orders if not provided and not auto-detected
    if not order:
        order = (1, 1, 1)
        logger.info(f"Using default ARIMA order: {order}")
    
    if seasonal and not seasonal_order:
        seasonal_order = (1, 1, 1, 12)  # Assuming monthly data
        logger.info(f"Using default seasonal order: {seasonal_order}")
    
    # Fit the model
    try:
        if seasonal:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            logger.info("Fitting SARIMA model...")
            fitted_model = model.fit(disp=False)
            
        else:
            model = ARIMA(series, order=order)
            
            logger.info("Fitting ARIMA model...")
            fitted_model = model.fit()
        
        # Make forecast
        logger.info(f"Forecasting {forecast_periods} periods ahead...")
        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'original': series,
            'fitted': fitted_model.fittedvalues,
        })
        
        # Add forecast
        forecast_index = pd.date_range(start=series.index[-1], periods=forecast_periods+1, freq='MS')[1:]
        forecast_series = pd.Series(forecast_mean.values, index=forecast_index)
        results_df = results_df.append(pd.DataFrame({'forecast': forecast_series}))
        
        # Add confidence intervals
        results_df['lower_ci'] = pd.Series(forecast_ci.iloc[:, 0].values, index=forecast_index)
        results_df['upper_ci'] = pd.Series(forecast_ci.iloc[:, 1].values, index=forecast_index)
        
        return results_df, fitted_model
        
    except Exception as e:
        logger.error(f"Error in ARIMA forecasting: {str(e)}")
        return None, None


def plot_forecast(results_df, title="Time Series Forecast", figsize=(12, 6)):
    """
    Plot the original series, fitted values, and forecast with confidence intervals.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        The DataFrame returned by arima_forecast function.
    title : str, optional
        The title of the plot.
    figsize : tuple, optional
        The size of the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    if results_df is None:
        logger.error("No results to plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original series
    results_df['original'].plot(ax=ax, label='Original', color='blue')
    
    # Plot fitted values
    results_df['fitted'].plot(ax=ax, label='Fitted', color='green', alpha=0.7)
    
    # Plot forecast
    forecast_start = results_df['forecast'].first_valid_index()
    results_df['forecast'].plot(ax=ax, label='Forecast', color='red')
    
    # Plot confidence intervals
    if 'lower_ci' in results_df.columns and 'upper_ci' in results_df.columns:
        ax.fill_between(
            results_df.index,
            results_df['lower_ci'],
            results_df['upper_ci'],
            color='red',
            alpha=0.2
        )
    
    # Add vertical line at forecast start
    if forecast_start:
        ax.axvline(forecast_start, color='black', linestyle='--', alpha=0.5)
    
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    return fig


def exponential_smoothing(series, forecast_periods=12, seasonal=True, seasonal_periods=12):
    """
    Apply exponential smoothing (ETS) for forecasting.
    
    Parameters
    ----------
    series : pandas.Series
        The time series to forecast.
    forecast_periods : int, optional
        Number of periods to forecast.
    seasonal : bool, optional
        Whether to use seasonal decomposition.
    seasonal_periods : int, optional
        Number of periods in seasonal cycle.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original series, the fitted values, and the forecast.
    statsmodels.tsa.holtwinters.HoltWintersResults
        The fitted model object.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Handle input
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            logger.error("Input must be a pandas Series or single-column DataFrame")
            return None, None
    
    # Check for missing values
    if series.isna().any():
        logger.warning("Series contains missing values. Interpolating...")
        series = series.interpolate(method='linear')
    
    try:
        # Initialize model
        if seasonal:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            )
        else:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None
            )
        
        # Fit model
        logger.info("Fitting Exponential Smoothing model...")
        fitted_model = model.fit()
        
        # Generate forecast
        logger.info(f"Forecasting {forecast_periods} periods ahead...")
        forecast = fitted_model.forecast(forecast_periods)
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'original': series,
            'fitted': fitted_model.fittedvalues,
        })
        
        # Add forecast
        forecast_index = pd.date_range(
            start=series.index[-1], 
            periods=forecast_periods+1, 
            freq=pd.infer_freq(series.index)
        )[1:]
        forecast_series = pd.Series(forecast.values, index=forecast_index)
        results_df = results_df.append(pd.DataFrame({'forecast': forecast_series}))
        
        return results_df, fitted_model
        
    except Exception as e:
        logger.error(f"Error in Exponential Smoothing: {str(e)}")
        return None, None
