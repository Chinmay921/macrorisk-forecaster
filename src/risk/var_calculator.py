"""
Value at Risk (VaR) calculator module.

This module provides functions to calculate Value at Risk (VaR) for financial 
and macroeconomic data using various approaches.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_var(returns, confidence=0.95, method='historical', window=None, clean_outliers=False):
    """
    Calculate Value at Risk (VaR) for a given return series.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns (percentage changes).
    confidence : float, optional
        Confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
    method : str, optional
        Method to use for VaR calculation: 'historical', 'parametric', or 'ewma'.
    window : int, optional
        Window size for EWMA method.
    clean_outliers : bool, optional
        Whether to remove outliers before calculation.
        
    Returns
    -------
    float or pandas.Series
        Calculated VaR value(s).
    """
    # Ensure input is correct
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        logger.error("Input must be a pandas Series or DataFrame")
        return None
    
    # Convert to returns if the data is not already in return format
    # (check if all values are between -1 and 1)
    if (abs(returns) > 1).any().any() if isinstance(returns, pd.DataFrame) else (abs(returns) > 1).any():
        logger.warning("Data appears to be in price format. Converting to returns.")
        returns = returns.pct_change().dropna()
    
    # Clean outliers if requested
    if clean_outliers:
        returns = _clean_outliers(returns)
    
    # Calculate VaR using the requested method
    if method == 'historical':
        return _historical_var(returns, confidence)
    elif method == 'parametric':
        return _parametric_var(returns, confidence)
    elif method == 'ewma':
        return _ewma_var(returns, confidence, window)
    else:
        logger.error(f"Unknown VaR method: {method}")
        return None


def _clean_outliers(data, threshold=3.0):
    """
    Remove outliers from the data.
    
    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Data to clean.
    threshold : float, optional
        Z-score threshold to identify outliers.
        
    Returns
    -------
    pandas.Series or pandas.DataFrame
        Cleaned data.
    """
    if isinstance(data, pd.DataFrame):
        # Clean each column
        cleaned_data = data.copy()
        for col in cleaned_data.columns:
            series = cleaned_data[col]
            z_scores = np.abs((series - series.mean()) / series.std())
            cleaned_data.loc[z_scores > threshold, col] = np.nan
        
        # Interpolate missing values
        cleaned_data = cleaned_data.interpolate(method='linear')
        
        # Log how many outliers were removed
        outliers_count = (data.count() - cleaned_data.count()).sum()
        logger.info(f"Removed {outliers_count} outliers from data.")
        
        return cleaned_data
        
    else:  # Series
        series = data.copy()
        z_scores = np.abs((series - series.mean()) / series.std())
        series[z_scores > threshold] = np.nan
        
        # Interpolate missing values
        series = series.interpolate(method='linear')
        
        # Log how many outliers were removed
        outliers_count = len(data) - series.count()
        logger.info(f"Removed {outliers_count} outliers from data.")
        
        return series


def _historical_var(returns, confidence):
    """
    Calculate historical VaR.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns.
    confidence : float
        Confidence level for VaR.
        
    Returns
    -------
    float or pandas.Series
        Calculated VaR value(s).
    """
    # Calculate percentile
    percentile = 1 - confidence
    
    if isinstance(returns, pd.DataFrame):
        # For each column, calculate VaR
        var_values = {}
        for col in returns.columns:
            var_values[col] = np.percentile(returns[col].dropna(), percentile * 100)
        
        return pd.Series(var_values)
    else:
        # For a series, return the percentile
        return np.percentile(returns.dropna(), percentile * 100)


def _parametric_var(returns, confidence):
    """
    Calculate parametric VaR assuming normal distribution.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns.
    confidence : float
        Confidence level for VaR.
        
    Returns
    -------
    float or pandas.Series
        Calculated VaR value(s).
    """
    # Get z-score for confidence level
    z_score = stats.norm.ppf(1 - confidence)
    
    if isinstance(returns, pd.DataFrame):
        # For each column, calculate VaR
        var_values = {}
        for col in returns.columns:
            series = returns[col].dropna()
            var_values[col] = series.mean() - z_score * series.std()
        
        return pd.Series(var_values)
    else:
        # For a series, calculate VaR
        returns = returns.dropna()
        return returns.mean() - z_score * returns.std()


def _ewma_var(returns, confidence, window=None):
    """
    Calculate VaR using Exponentially Weighted Moving Average for volatility.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns.
    confidence : float
        Confidence level for VaR.
    window : int, optional
        Window size for EWMA. If None, use RiskMetrics decay factor.
        
    Returns
    -------
    float or pandas.Series
        Calculated VaR value(s).
    """
    # Get z-score for confidence level
    z_score = stats.norm.ppf(1 - confidence)
    
    # RiskMetrics decay factor
    decay_factor = 0.94
    
    if isinstance(returns, pd.DataFrame):
        # For each column, calculate VaR
        var_values = {}
        for col in returns.columns:
            series = returns[col].dropna()
            
            if window:
                # Use rolling window
                variance = series.ewm(span=window).var()
            else:
                # Use RiskMetrics approach
                variance = series.ewm(alpha=1-decay_factor).var()
            
            # Get most recent variance
            latest_variance = variance.iloc[-1]
            
            # Calculate VaR
            var_values[col] = series.mean() - z_score * np.sqrt(latest_variance)
        
        return pd.Series(var_values)
    else:
        # For a series, calculate VaR
        returns = returns.dropna()
        
        if window:
            # Use rolling window
            variance = returns.ewm(span=window).var()
        else:
            # Use RiskMetrics approach
            variance = returns.ewm(alpha=1-decay_factor).var()
        
        # Get most recent variance
        latest_variance = variance.iloc[-1]
        
        # Calculate VaR
        return returns.mean() - z_score * np.sqrt(latest_variance)


def calculate_cvar(returns, confidence=0.95, method='historical', window=None, clean_outliers=False):
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns.
    confidence : float, optional
        Confidence level for CVaR calculation.
    method : str, optional
        Method to use for initial VaR calculation.
    window : int, optional
        Window size for EWMA method.
    clean_outliers : bool, optional
        Whether to remove outliers before calculation.
        
    Returns
    -------
    float or pandas.Series
        Calculated CVaR value(s).
    """
    # First, calculate VaR
    var = calculate_var(returns, confidence, method, window, clean_outliers)
    
    if var is None:
        return None
    
    # For historical method, calculate the mean of all returns worse than VaR
    if method == 'historical':
        if isinstance(returns, pd.DataFrame):
            # For each column, calculate CVaR
            cvar_values = {}
            for col in returns.columns:
                series = returns[col].dropna()
                var_value = var[col]
                
                # Get all returns worse than VaR
                worse_returns = series[series < var_value]
                
                # Calculate CVaR as the mean of worse returns
                cvar_values[col] = worse_returns.mean() if len(worse_returns) > 0 else var_value
            
            return pd.Series(cvar_values)
        else:
            # For a series, calculate CVaR
            returns = returns.dropna()
            
            # Get all returns worse than VaR
            worse_returns = returns[returns < var]
            
            # Calculate CVaR as the mean of worse returns
            return worse_returns.mean() if len(worse_returns) > 0 else var
    
    # For parametric method, calculate CVaR analytically
    elif method == 'parametric':
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # Calculate probability density at z-score
        pdf_value = stats.norm.pdf(z_score)
        
        # Calculate CVaR adjustment
        adjustment = pdf_value / (1 - confidence)
        
        if isinstance(returns, pd.DataFrame):
            # For each column, calculate CVaR
            cvar_values = {}
            for col in returns.columns:
                series = returns[col].dropna()
                var_value = var[col]
                
                # Adjust VaR to get CVaR
                cvar_values[col] = var_value - adjustment * series.std()
            
            return pd.Series(cvar_values)
        else:
            # For a series, calculate CVaR
            returns = returns.dropna()
            
            # Adjust VaR to get CVaR
            return var - adjustment * returns.std()
    
    # For EWMA method, similar approach as parametric
    elif method == 'ewma':
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # Calculate probability density at z-score
        pdf_value = stats.norm.pdf(z_score)
        
        # Calculate CVaR adjustment
        adjustment = pdf_value / (1 - confidence)
        
        # RiskMetrics decay factor
        decay_factor = 0.94
        
        if isinstance(returns, pd.DataFrame):
            # For each column, calculate CVaR
            cvar_values = {}
            for col in returns.columns:
                series = returns[col].dropna()
                var_value = var[col]
                
                if window:
                    # Use rolling window
                    variance = series.ewm(span=window).var()
                else:
                    # Use RiskMetrics approach
                    variance = series.ewm(alpha=1-decay_factor).var()
                
                # Get most recent volatility
                latest_volatility = np.sqrt(variance.iloc[-1])
                
                # Adjust VaR to get CVaR
                cvar_values[col] = var_value - adjustment * latest_volatility
            
            return pd.Series(cvar_values)
        else:
            # For a series, calculate CVaR
            returns = returns.dropna()
            
            if window:
                # Use rolling window
                variance = returns.ewm(span=window).var()
            else:
                # Use RiskMetrics approach
                variance = returns.ewm(alpha=1-decay_factor).var()
            
            # Get most recent volatility
            latest_volatility = np.sqrt(variance.iloc[-1])
            
            # Adjust VaR to get CVaR
            return var - adjustment * latest_volatility
    
    else:
        logger.error(f"Unknown CVaR method: {method}")
        return None


def plot_var(returns, confidence=0.95, method='historical', window=None, figsize=(12, 6)):
    """
    Plot histogram of returns with VaR and CVaR.
    
    Parameters
    ----------
    returns : pandas.Series
        Series of returns.
    confidence : float, optional
        Confidence level for VaR and CVaR calculation.
    method : str, optional
        Method to use for VaR and CVaR calculation.
    window : int, optional
        Window size for EWMA method.
    figsize : tuple, optional
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        Plot of returns with VaR and CVaR.
    """
    if not isinstance(returns, pd.Series):
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            logger.error("Input must be a pandas Series or single-column DataFrame")
            return None
    
    # Calculate VaR and CVaR
    var = calculate_var(returns, confidence, method, window)
    cvar = calculate_cvar(returns, confidence, method, window)
    
    if var is None or cvar is None:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    returns.hist(ax=ax, bins=50, alpha=0.5, density=True)
    
    # Plot kernel density estimate
    returns.plot(kind='kde', ax=ax, color='blue')
    
    # Plot VaR and CVaR
    ax.axvline(var, color='red', linestyle='--', label=f'VaR ({confidence:.0%})')
    ax.axvline(cvar, color='darkred', linestyle='--', label=f'CVaR ({confidence:.0%})')
    
    # Add text annotations
    var_text = f'VaR ({confidence:.0%}): {var:.2%}'
    cvar_text = f'CVaR ({confidence:.0%}): {cvar:.2%}'
    
    ax.text(0.02, 0.95, var_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.02, 0.90, cvar_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Returns')
    ax.set_ylabel('Density')
    ax.set_title(f'Value at Risk ({method.capitalize()} Method)')
    ax.legend()
    
    plt.tight_layout()
    return fig


def rolling_var(returns, window=252, confidence=0.95, method='historical', plot=True, figsize=(12, 6)):
    """
    Calculate rolling VaR over a specified window.
    
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
        Series or DataFrame of returns.
    window : int, optional
        Window size for rolling calculation.
    confidence : float, optional
        Confidence level for VaR calculation.
    method : str, optional
        Method to use for VaR calculation.
    plot : bool, optional
        Whether to create a plot of rolling VaR.
    figsize : tuple, optional
        Figure size for plot.
        
    Returns
    -------
    pandas.Series or pandas.DataFrame
        Rolling VaR values.
    matplotlib.figure.Figure, optional
        Plot of rolling VaR if plot=True.
    """
    # Calculate rolling VaR
    if isinstance(returns, pd.DataFrame):
        # For each column, calculate rolling VaR
        rolling_var_values = pd.DataFrame(index=returns.index)
        
        for col in returns.columns:
            series = returns[col]
            
            # Calculate rolling VaR
            rolling_var_col = pd.Series(index=series.index)
            
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i-window:i]
                var_value = calculate_var(window_data, confidence, method)
                rolling_var_col.iloc[i-1] = var_value
            
            rolling_var_values[col] = rolling_var_col
        
        # Create plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            rolling_var_values.plot(ax=ax)
            ax.set_title(f'Rolling {confidence:.0%} VaR ({method.capitalize()} Method)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value at Risk')
            plt.tight_layout()
            return rolling_var_values, fig
        else:
            return rolling_var_values
    
    else:  # Series
        # Calculate rolling VaR
        rolling_var_values = pd.Series(index=returns.index)
        
        for i in range(window, len(returns) + 1):
            window_data = returns.iloc[i-window:i]
            var_value = calculate_var(window_data, confidence, method)
            rolling_var_values.iloc[i-1] = var_value
        
        # Create plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            rolling_var_values.plot(ax=ax)
            ax.set_title(f'Rolling {confidence:.0%} VaR ({method.capitalize()} Method)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value at Risk')
            
            # Plot returns
            ax2 = ax.twinx()
            returns.plot(ax=ax2, alpha=0.3, color='gray')
            ax2.set_ylabel('Returns')
            
            plt.tight_layout()
            return rolling_var_values, fig
        else:
            return rolling_var_values
