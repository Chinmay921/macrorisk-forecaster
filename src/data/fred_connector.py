"""
FRED Data Connector Module

This module handles data retrieval from the Federal Reserve Economic Database (FRED)
using their API. It provides functions to fetch, clean, and preprocess economic indicators.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fredapi import Fred
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FREDConnector:
    """
    A class to handle connections to the FRED API and retrieve economic data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the FRED API connector.
        
        Parameters:
        -----------
        api_key : str, optional
            The API key for FRED. If not provided, it will be read from FRED_API_KEY
            environment variable.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not provided. Set the FRED_API_KEY environment variable.")
        
        self.fred = Fred(api_key=self.api_key)
        logger.info("FRED connector initialized successfully")
        
        # Common economic indicators and their FRED series IDs
        self.common_indicators = {
            'gdp': 'GDP',                      # Gross Domestic Product
            'inflation': 'CPIAUCSL',           # Consumer Price Index for All Urban Consumers
            'unemployment': 'UNRATE',          # Unemployment Rate
            'fed_funds': 'FEDFUNDS',           # Federal Funds Effective Rate
            'treasury_10y': 'GS10',            # 10-Year Treasury Constant Maturity Rate
            'treasury_2y': 'GS2',              # 2-Year Treasury Constant Maturity Rate
            'industrial_production': 'INDPRO', # Industrial Production Index
            'retail_sales': 'RSAFS',           # Retail Sales
            'house_price_index': 'CSUSHPISA',  # S&P/Case-Shiller U.S. National Home Price Index
            'consumer_sentiment': 'UMCSENT'    # University of Michigan: Consumer Sentiment
        }
    
    def get_series(self, series_id, start_date=None, end_date=None, frequency=None, transform=None):
        """
        Retrieve a time series from FRED.
        
        Parameters:
        -----------
        series_id : str
            The FRED series ID to retrieve.
        start_date : str, optional
            The start date in 'YYYY-MM-DD' format. Default is 10 years ago.
        end_date : str, optional
            The end date in 'YYYY-MM-DD' format. Default is today.
        frequency : str, optional
            Data frequency ('d', 'w', 'm', 'q', 'a'). Default is None (use original frequency).
        transform : str, optional
            Transformation to apply ('lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log').
            
        Returns:
        --------
        pd.Series
            Time series of the requested data.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Retrieving FRED series {series_id} from {start_date} to {end_date}")
        
        try:
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date,
                frequency=frequency,
                transform=transform
            )
            
            # Handle missing values
            if data.isna().any():
                logger.warning(f"Series {series_id} contains {data.isna().sum()} missing values")
                
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving series {series_id}: {str(e)}")
            raise
    
    def get_common_indicator(self, indicator, **kwargs):
        """
        Retrieve one of the common economic indicators by name.
        
        Parameters:
        -----------
        indicator : str
            The name of the indicator (e.g., 'gdp', 'inflation', 'unemployment').
        **kwargs : 
            Additional arguments to pass to get_series().
            
        Returns:
        --------
        pd.Series
            Time series of the requested indicator.
        """
        if indicator not in self.common_indicators:
            raise ValueError(f"Unknown indicator: {indicator}. Available indicators: {', '.join(self.common_indicators.keys())}")
        
        series_id = self.common_indicators[indicator]
        return self.get_series(series_id, **kwargs)
    
    def get_multiple_indicators(self, indicators, **kwargs):
        """
        Retrieve multiple indicators and combine them into a DataFrame.
        
        Parameters:
        -----------
        indicators : list
            List of indicator names to retrieve.
        **kwargs : 
            Additional arguments to pass to get_series().
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing all requested indicators.
        """
        result = pd.DataFrame()
        
        for indicator in indicators:
            series = self.get_common_indicator(indicator, **kwargs)
            result[indicator] = series
            
        return result
    
    def search_series(self, search_text, limit=10):
        """
        Search for series in FRED database.
        
        Parameters:
        -----------
        search_text : str
            Text to search for.
        limit : int, optional
            Maximum number of results to return.
            
        Returns:
        --------
        pd.DataFrame
            Information about matching series.
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching for '{search_text}': {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    connector = FREDConnector()
    
    # Get single indicator
    inflation = connector.get_common_indicator('inflation', frequency='m')
    print(f"Inflation data shape: {inflation.shape}")
    print(inflation.tail())
    
    # Get multiple indicators
    macro_data = connector.get_multiple_indicators(['gdp', 'unemployment', 'fed_funds'], frequency='q')
    print(f"Macro data shape: {macro_data.shape}")
    print(macro_data.tail())