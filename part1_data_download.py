"""
Part 1: Data Download and Preparation
FINM_32500 - End-to-End Trading System
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataDownloader:
    """
    Downloads and prepares intraday market data for trading strategy development.
    Supports both equity and cryptocurrency data.
    """

    def __init__(self, ticker: str, data_source: str = 'yfinance'):
        """
        Initialize the data downloader.

        Args:
            ticker: Stock symbol (e.g., 'AAPL') or crypto pair
            data_source: 'yfinance' for stocks, 'binance' for crypto
        """
        self.ticker = ticker
        self.data_source = data_source
        self.raw_data = None
        self.clean_data = None

    def download_equity_data(self, period: str = '7d', interval: str = '1m') -> pd.DataFrame:
        """
        Download intraday equity data using yfinance.

        Args:
            period: Time period (e.g., '1d', '5d', '7d', '1mo')
            interval: Data interval (e.g., '1m', '5m', '15m', '1h')

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading {self.ticker} data for period={period}, interval={interval}...")

        try:
            data = yf.download(
                tickers=self.ticker,
                period=period,
                interval=interval,
                progress=False
            )

            if data.empty:
                raise ValueError(f"No data downloaded for {self.ticker}")

            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            self.raw_data = data
            print(f"Downloaded {len(data)} rows of data")
            return data

        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def download_crypto_data(self, symbol: str = 'BTCUSDT', interval: str = '1m',
                            limit: int = 1000) -> pd.DataFrame:
        """
        Download cryptocurrency data from Binance API.

        Args:
            symbol: Crypto trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of candlesticks to retrieve (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        import requests

        print(f"Downloading {symbol} crypto data from Binance...")

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base',
                'Taker_buy_quote', 'Ignore'
            ])

            # Convert timestamp to datetime
            df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')

            # Select and rename columns to match equity format
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

            self.raw_data = df
            print(f"Downloaded {len(df)} rows of crypto data")
            return df

        except Exception as e:
            print(f"Error downloading crypto data: {e}")
            raise

    def clean_data(self, add_features: bool = True) -> pd.DataFrame:
        """
        Clean and organize the downloaded data.

        Args:
            add_features: Whether to add derived features

        Returns:
            Cleaned DataFrame ready for analysis
        """
        if self.raw_data is None:
            raise ValueError("No data to clean. Download data first.")

        print("Cleaning data...")
        df = self.raw_data.copy()

        # Reset index if datetime is not already the index
        if 'Datetime' in df.columns:
            df.set_index('Datetime', inplace=True)

        # Remove duplicates
        initial_rows = len(df)
        df = df[~df.index.duplicated(keep='first')]
        print(f"Removed {initial_rows - len(df)} duplicate rows")

        # Remove missing values
        initial_rows = len(df)
        df.dropna(inplace=True)
        print(f"Removed {initial_rows - len(df)} rows with missing values")

        # Sort chronologically
        df.sort_index(inplace=True)

        # Add derived features if requested
        if add_features:
            df = self._add_features(df)

        self.clean_data = df
        print(f"Data cleaning complete. Final dataset: {len(df)} rows")
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for strategy development.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        print("Adding derived features...")

        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential moving averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Volatility (rolling standard deviation)
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()

        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

        # Volume features
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

        # RSI (Relative Strength Index)
        df['RSI_14'] = self._calculate_rsi(df['Close'], period=14)

        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)

        print(f"Added {df.shape[1] - 6} derived features")
        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def save_data(self, filename: str, use_clean: bool = True):
        """
        Save data to CSV file.

        Args:
            filename: Output filename
            use_clean: Use cleaned data if True, raw data if False
        """
        data = self.clean_data if use_clean and self.clean_data is not None else self.raw_data

        if data is None:
            raise ValueError("No data to save")

        data.to_csv(filename)
        print(f"Data saved to {filename}")


def main():
    """
    Example usage: Download and prepare market data.
    """
    # Example 1: Download equity data
    print("=" * 60)
    print("Example 1: Downloading AAPL equity data")
    print("=" * 60)

    equity_downloader = DataDownloader('AAPL', data_source='yfinance')
    equity_data = equity_downloader.download_equity_data(period='7d', interval='1m')
    clean_equity_data = equity_downloader.clean_data(add_features=True)
    equity_downloader.save_data('market_data_aapl.csv')

    print("\nSample of cleaned equity data:")
    print(clean_equity_data.head())
    print(f"\nColumns: {list(clean_equity_data.columns)}")

    # Example 2: Download crypto data
    print("\n" + "=" * 60)
    print("Example 2: Downloading BTC crypto data")
    print("=" * 60)

    crypto_downloader = DataDownloader('BTC', data_source='binance')
    crypto_data = crypto_downloader.download_crypto_data(symbol='BTCUSDT', interval='1m', limit=1000)
    clean_crypto_data = crypto_downloader.clean_data(add_features=True)
    crypto_downloader.save_data('market_data_btc.csv')

    print("\nSample of cleaned crypto data:")
    print(clean_crypto_data.head())

    print("\n" + "=" * 60)
    print("Data download and preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
