#!/usr/bin/env python3
"""Simple test script for FRED connector"""

import sys
import os
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append('.')

from src.data.fred_connector import FREDConnector

# Load environment variables
load_dotenv()

print("Testing FRED Connector")
print("=====================")

# Initialize FRED connector
fred = FREDConnector()

# Get GDP data
print("Fetching GDP data...")
gdp = fred.get_common_indicator('gdp', frequency='q')
print(f"GDP data shape: {gdp.shape}")
print(gdp.tail())

print("\nTest completed successfully!") 