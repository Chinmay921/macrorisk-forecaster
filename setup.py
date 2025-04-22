from setuptools import setup, find_packages

setup(
    name="macrorisk-forecaster",
    version="0.1.0",
    description="A tool for forecasting macroeconomic risk factors",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/macrorisk-forecaster",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "fredapi>=0.5.0",
        "worldbank>=0.3.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "tensorflow>=2.7.0",
        "pmdarima>=2.0.0"  # For ARIMA models
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "macrorisk=src.main:main",
        ],
    },
)
