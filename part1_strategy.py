"""
Part 1 Step 3: Trading Strategy Implementation
FINM_32500 - End-to-End Trading System

Implements multiple trading strategies with signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Signal(Enum):
    """Trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy:
    """
    Base class for trading strategies.
    All strategies should inherit from this class and implement generate_signals().
    """

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.signals = None
        self.positions = None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.

        Args:
            data: DataFrame with market data and features

        Returns:
            DataFrame with added 'Signal' column
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def calculate_positions(self, signals: pd.DataFrame, position_size: float = 1.0) -> pd.DataFrame:
        """
        Calculate positions from signals.

        Args:
            signals: DataFrame with Signal column
            position_size: Size of each position

        Returns:
            DataFrame with Position column
        """
        df = signals.copy()
        df['Position'] = df['Signal'].replace({
            Signal.BUY.value: position_size,
            Signal.SELL.value: -position_size,
            Signal.HOLD.value: 0
        })
        return df


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    Buys when price momentum is positive, sells when negative.
    """

    def __init__(self, lookback_period: int = 10, threshold: float = 0.0):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Period for momentum calculation
            threshold: Minimum momentum for signal generation
        """
        super().__init__(name=f"Momentum_{lookback_period}")
        self.lookback_period = lookback_period
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on price momentum.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with Signal column
        """
        df = data.copy()

        # Calculate momentum if not already present
        momentum_col = f'Momentum_{self.lookback_period}'
        if momentum_col not in df.columns:
            df[momentum_col] = df['Close'] - df['Close'].shift(self.lookback_period)

        # Generate signals
        df['Signal'] = Signal.HOLD.value
        df.loc[df[momentum_col] > self.threshold, 'Signal'] = Signal.BUY.value
        df.loc[df[momentum_col] < -self.threshold, 'Signal'] = Signal.SELL.value

        self.signals = df
        return df


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy.
    Generates buy signal when short MA crosses above long MA,
    sell signal when short MA crosses below long MA.
    """

    def __init__(self, short_window: int = 5, long_window: int = 20, ma_type: str = 'SMA'):
        """
        Initialize MA crossover strategy.

        Args:
            short_window: Short moving average period
            long_window: Long moving average period
            ma_type: 'SMA' or 'EMA'
        """
        super().__init__(name=f"MA_Cross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MA crossover.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with Signal column
        """
        df = data.copy()

        # Calculate moving averages
        short_ma_col = f'{self.ma_type}_{self.short_window}'
        long_ma_col = f'{self.ma_type}_{self.long_window}'

        if short_ma_col not in df.columns:
            if self.ma_type == 'SMA':
                df[short_ma_col] = df['Close'].rolling(window=self.short_window).mean()
            else:  # EMA
                df[short_ma_col] = df['Close'].ewm(span=self.short_window, adjust=False).mean()

        if long_ma_col not in df.columns:
            if self.ma_type == 'SMA':
                df[long_ma_col] = df['Close'].rolling(window=self.long_window).mean()
            else:  # EMA
                df[long_ma_col] = df['Close'].ewm(span=self.long_window, adjust=False).mean()

        # Generate signals based on crossover
        df['MA_Diff'] = df[short_ma_col] - df[long_ma_col]
        df['MA_Diff_Prev'] = df['MA_Diff'].shift(1)

        df['Signal'] = Signal.HOLD.value

        # Buy when short MA crosses above long MA
        df.loc[(df['MA_Diff'] > 0) & (df['MA_Diff_Prev'] <= 0), 'Signal'] = Signal.BUY.value

        # Sell when short MA crosses below long MA
        df.loc[(df['MA_Diff'] < 0) & (df['MA_Diff_Prev'] >= 0), 'Signal'] = Signal.SELL.value

        self.signals = df
        return df


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    Buys when RSI is oversold, sells when overbought.
    """

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI strategy.

        Args:
            rsi_period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        super().__init__(name=f"RSI_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on RSI.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with Signal column
        """
        df = data.copy()

        # Calculate RSI if not present
        rsi_col = f'RSI_{self.rsi_period}'
        if rsi_col not in df.columns:
            df[rsi_col] = self._calculate_rsi(df['Close'], self.rsi_period)

        # Generate signals
        df['Signal'] = Signal.HOLD.value
        df.loc[df[rsi_col] < self.oversold, 'Signal'] = Signal.BUY.value
        df.loc[df[rsi_col] > self.overbought, 'Signal'] = Signal.SELL.value

        self.signals = df
        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy.
    Buys when price touches lower band, sells when it touches upper band.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Bands strategy.

        Args:
            period: Moving average period
            num_std: Number of standard deviations for bands
        """
        super().__init__(name=f"BB_{period}")
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Bollinger Bands.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with Signal column
        """
        df = data.copy()

        # Calculate Bollinger Bands if not present
        if 'BB_Middle' not in df.columns:
            df['BB_Middle'] = df['Close'].rolling(window=self.period).mean()
            bb_std = df['Close'].rolling(window=self.period).std()
            df['BB_Upper'] = df['BB_Middle'] + (self.num_std * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (self.num_std * bb_std)

        # Generate signals
        df['Signal'] = Signal.HOLD.value

        # Buy when price touches or crosses below lower band
        df.loc[df['Close'] <= df['BB_Lower'], 'Signal'] = Signal.BUY.value

        # Sell when price touches or crosses above upper band
        df.loc[df['Close'] >= df['BB_Upper'], 'Signal'] = Signal.SELL.value

        self.signals = df
        return df


class CombinedStrategy(BaseStrategy):
    """
    Combined strategy that uses multiple indicators.
    Generates signals when multiple conditions align.
    """

    def __init__(self,
                 ma_short: int = 5,
                 ma_long: int = 20,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """
        Initialize combined strategy.

        Args:
            ma_short: Short moving average period
            ma_long: Long moving average period
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        super().__init__(name="Combined_Strategy")
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on multiple indicators.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with Signal column
        """
        df = data.copy()

        # Calculate indicators if not present
        short_ma = f'SMA_{self.ma_short}'
        long_ma = f'SMA_{self.ma_long}'
        rsi_col = f'RSI_{self.rsi_period}'

        if short_ma not in df.columns:
            df[short_ma] = df['Close'].rolling(window=self.ma_short).mean()
        if long_ma not in df.columns:
            df[long_ma] = df['Close'].rolling(window=self.ma_long).mean()
        if rsi_col not in df.columns:
            df[rsi_col] = self._calculate_rsi(df['Close'], self.rsi_period)

        # Initialize signal
        df['Signal'] = Signal.HOLD.value

        # Buy signals: MA trending up AND RSI oversold
        buy_condition = (df[short_ma] > df[long_ma]) & (df[rsi_col] < self.rsi_oversold)
        df.loc[buy_condition, 'Signal'] = Signal.BUY.value

        # Sell signals: MA trending down AND RSI overbought
        sell_condition = (df[short_ma] < df[long_ma]) & (df[rsi_col] > self.rsi_overbought)
        df.loc[sell_condition, 'Signal'] = Signal.SELL.value

        self.signals = df
        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def main():
    """
    Example usage: Test strategies on sample data.
    """
    # Load sample data
    try:
        data = pd.read_csv('market_data_aapl.csv', index_col='Datetime', parse_dates=True)
        print(f"Loaded data with {len(data)} rows")
    except FileNotFoundError:
        print("Sample data not found. Run part1_data_download.py first.")
        return

    print("\n" + "=" * 60)
    print("Testing Trading Strategies")
    print("=" * 60)

    # Test each strategy
    strategies = [
        MomentumStrategy(lookback_period=10),
        MovingAverageCrossoverStrategy(short_window=5, long_window=20),
        RSIStrategy(rsi_period=14),
        BollingerBandsStrategy(period=20),
        CombinedStrategy()
    ]

    for strategy in strategies:
        print(f"\n{strategy.name}:")
        signals = strategy.generate_signals(data)

        # Count signals
        buy_signals = (signals['Signal'] == Signal.BUY.value).sum()
        sell_signals = (signals['Signal'] == Signal.SELL.value).sum()
        hold_signals = (signals['Signal'] == Signal.HOLD.value).sum()

        print(f"  Buy signals: {buy_signals}")
        print(f"  Sell signals: {sell_signals}")
        print(f"  Hold signals: {hold_signals}")

        # Show first few signals
        signal_rows = signals[signals['Signal'] != Signal.HOLD.value].head()
        if not signal_rows.empty:
            print(f"  First signals:")
            for idx, row in signal_rows.iterrows():
                signal_type = "BUY" if row['Signal'] == Signal.BUY.value else "SELL"
                print(f"    {idx}: {signal_type} at ${row['Close']:.2f}")

    print("\n" + "=" * 60)
    print("Strategy testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
