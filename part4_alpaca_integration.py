"""
Part 4: Alpaca Trading Integration
FINM_32500 - End-to-End Trading System

Integration with Alpaca Paper Trading API.
IMPORTANT: Only use paper trading keys! Do not add real money!
"""

import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")


class AlpacaConfig:
    """
    Configuration for Alpaca API.
    Store your keys in environment variables or a separate config file.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 base_url: str = 'https://paper-api.alpaca.markets'):
        """
        Initialize Alpaca configuration.

        Args:
            api_key: Alpaca API Key ID (or set ALPACA_API_KEY env var)
            api_secret: Alpaca API Secret Key (or set ALPACA_SECRET_KEY env var)
            base_url: API base URL (default: paper trading)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url

        if not self.api_key or not self.api_secret:
            print("\nWARNING: Alpaca API keys not found!")
            print("Set environment variables or pass to constructor:")
            print("  export ALPACA_API_KEY='your_key_id'")
            print("  export ALPACA_SECRET_KEY='your_secret_key'")

    def is_configured(self) -> bool:
        """Check if API keys are configured"""
        return bool(self.api_key and self.api_secret)


class AlpacaDataFetcher:
    """
    Fetches market data from Alpaca API.
    """

    def __init__(self, config: AlpacaConfig):
        """
        Initialize data fetcher.

        Args:
            config: AlpacaConfig instance
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not installed")

        if not config.is_configured():
            raise ValueError("Alpaca API keys not configured")

        self.config = config
        self.api = tradeapi.REST(
            config.api_key,
            config.api_secret,
            config.base_url,
            api_version='v2'
        )

    def fetch_bars(self,
                   symbol: str,
                   timeframe: str = '1Min',
                   start: Optional[str] = None,
                   end: Optional[str] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        Fetch bar data from Alpaca.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Bar timeframe (e.g., '1Min', '5Min', '1Hour', '1Day')
            start: Start date (ISO format or datetime)
            end: End date (ISO format or datetime)
            limit: Maximum number of bars

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {symbol} data from Alpaca...")

        try:
            # Get bars
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit
            ).df

            if bars.empty:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Rename columns to match our format
            bars = bars.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Reset index to make timestamp a column
            bars = bars.reset_index()
            bars = bars.rename(columns={'timestamp': 'Datetime'})

            print(f"Fetched {len(bars)} bars")
            return bars

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_latest_quote(self, symbol: str) -> Dict:
        """
        Fetch latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with quote data
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bp,
                'ask': quote.ap,
                'bid_size': quote.bs,
                'ask_size': quote.as_,
                'timestamp': quote.t
            }
        except Exception as e:
            print(f"Error fetching quote: {e}")
            return {}

    def fetch_latest_trade(self, symbol: str) -> Dict:
        """
        Fetch latest trade for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with trade data
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            return {
                'symbol': symbol,
                'price': trade.p,
                'size': trade.s,
                'timestamp': trade.t
            }
        except Exception as e:
            print(f"Error fetching trade: {e}")
            return {}


class AlpacaDataStorage:
    """
    Handles storage of market data fetched from Alpaca.
    Supports CSV and Pickle formats.
    """

    @staticmethod
    def save_to_csv(data: pd.DataFrame, filename: str):
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            filename: Output filename
        """
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    @staticmethod
    def save_to_pickle(data: pd.DataFrame, filename: str):
        """
        Save data to Pickle file.

        Args:
            data: DataFrame to save
            filename: Output filename
        """
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")

    @staticmethod
    def load_from_csv(filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Input filename

        Returns:
            DataFrame
        """
        return pd.read_csv(filename, parse_dates=['Datetime'])

    @staticmethod
    def load_from_pickle(filename: str) -> pd.DataFrame:
        """
        Load data from Pickle file.

        Args:
            filename: Input filename

        Returns:
            DataFrame
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AlpacaTradingClient:
    """
    Handles live trading with Alpaca Paper Trading.
    Integrates with strategies from Part 1.
    """

    def __init__(self, config: AlpacaConfig, strategy):
        """
        Initialize trading client.

        Args:
            config: AlpacaConfig instance
            strategy: Strategy instance from part1_strategy
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not installed")

        if not config.is_configured():
            raise ValueError("Alpaca API keys not configured")

        self.config = config
        self.strategy = strategy
        self.api = tradeapi.REST(
            config.api_key,
            config.api_secret,
            config.base_url,
            api_version='v2'
        )

        self.positions = {}
        self.pending_orders = {}

    def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Dictionary with account info
        """
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            print(f"Error getting account: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': p.symbol,
                'quantity': int(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def submit_order(self,
                    symbol: str,
                    qty: int,
                    side: str,
                    order_type: str = 'market',
                    limit_price: Optional[float] = None) -> Optional[str]:
        """
        Submit an order to Alpaca.

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            limit_price: Limit price (for limit orders)

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=limit_price
            )

            print(f"Order submitted: {order.id} - {side.upper()} {qty} {symbol}")
            return order.id

        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            self.api.cancel_order(order_id)
            print(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False

    def get_orders(self, status: str = 'open') -> List[Dict]:
        """
        Get orders.

        Args:
            status: Order status ('open', 'closed', 'all')

        Returns:
            List of order dictionaries
        """
        try:
            orders = self.api.list_orders(status=status)
            return [{
                'id': o.id,
                'symbol': o.symbol,
                'qty': int(o.qty),
                'filled_qty': int(o.filled_qty),
                'side': o.side,
                'type': o.type,
                'status': o.status,
                'limit_price': float(o.limit_price) if o.limit_price else None,
                'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None
            } for o in orders]
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []

    def run_live_trading(self,
                        symbol: str,
                        check_interval: int = 60,
                        position_size: int = 10):
        """
        Run live trading with the configured strategy.

        Args:
            symbol: Symbol to trade
            check_interval: Seconds between checks
            position_size: Position size per trade

        WARNING: This runs indefinitely. Use Ctrl+C to stop.
        """
        print(f"\nStarting live trading for {symbol}")
        print(f"Strategy: {self.strategy.name}")
        print(f"Check interval: {check_interval} seconds")
        print(f"Position size: {position_size}")
        print("\nPress Ctrl+C to stop\n")

        try:
            while True:
                # Fetch latest data
                fetcher = AlpacaDataFetcher(self.config)
                data = fetcher.fetch_bars(
                    symbol=symbol,
                    timeframe='1Min',
                    limit=100
                )

                if data.empty:
                    print("No data available")
                    time.sleep(check_interval)
                    continue

                # Generate signals
                signals = self.strategy.generate_signals(data)
                latest_signal = signals.iloc[-1]['Signal']

                # Get current position
                positions = self.get_positions()
                current_qty = 0
                for pos in positions:
                    if pos['symbol'] == symbol:
                        current_qty = pos['quantity']
                        break

                # Execute trades based on signal
                from part1_strategy import Signal

                if latest_signal == Signal.BUY.value and current_qty == 0:
                    print(f"\nBUY signal detected at {datetime.now()}")
                    self.submit_order(symbol, position_size, 'buy', 'market')

                elif latest_signal == Signal.SELL.value and current_qty > 0:
                    print(f"\nSELL signal detected at {datetime.now()}")
                    self.submit_order(symbol, current_qty, 'sell', 'market')

                # Display status
                account = self.get_account()
                print(f"\n[{datetime.now()}] Status:")
                print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
                print(f"  Cash: ${account.get('cash', 0):,.2f}")
                print(f"  Position: {current_qty} shares")
                print(f"  Latest Signal: {latest_signal}")

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nStopping live trading...")
            print("Final account status:")
            account = self.get_account()
            for key, value in account.items():
                print(f"  {key}: {value}")


def main():
    """
    Example usage: Alpaca integration.
    """
    print("=" * 60)
    print("Alpaca Trading Integration")
    print("=" * 60)

    # Initialize configuration
    config = AlpacaConfig()

    if not config.is_configured():
        print("\nTo use Alpaca integration:")
        print("1. Sign up at https://alpaca.markets")
        print("2. Get your Paper Trading API keys")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY='your_key_id'")
        print("   export ALPACA_SECRET_KEY='your_secret_key'")
        print("\nOr create a config file:")
        print("   config = AlpacaConfig(api_key='...', api_secret='...')")
        return

    if not ALPACA_AVAILABLE:
        print("\nPlease install alpaca-trade-api:")
        print("  pip install alpaca-trade-api")
        return

    # Example 1: Fetch market data
    print("\nExample 1: Fetching market data")
    print("-" * 60)

    fetcher = AlpacaDataFetcher(config)

    # Fetch AAPL data
    data = fetcher.fetch_bars(
        symbol='AAPL',
        timeframe='1Min',
        limit=100
    )

    if not data.empty:
        print(f"\nFetched {len(data)} bars")
        print(data.head())

        # Save data
        storage = AlpacaDataStorage()
        storage.save_to_csv(data, 'alpaca_aapl_data.csv')

    # Example 2: Get account info
    print("\nExample 2: Account Information")
    print("-" * 60)

    from part1_strategy import MovingAverageCrossoverStrategy

    strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
    client = AlpacaTradingClient(config, strategy)

    account = client.get_account()
    print("\nAccount Details:")
    for key, value in account.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value}")

    # Example 3: Get positions
    print("\nExample 3: Current Positions")
    print("-" * 60)

    positions = client.get_positions()
    if positions:
        for pos in positions:
            print(f"\n{pos['symbol']}:")
            print(f"  Quantity: {pos['quantity']}")
            print(f"  Avg Entry: ${pos['avg_entry_price']:.2f}")
            print(f"  Market Value: ${pos['market_value']:.2f}")
            print(f"  Unrealized P/L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2%})")
    else:
        print("No open positions")

    # Example 4: Live trading (commented out for safety)
    print("\nExample 4: Live Trading")
    print("-" * 60)
    print("To run live trading, uncomment the following line:")
    print("# client.run_live_trading(symbol='AAPL', check_interval=60, position_size=1)")
    print("\nWARNING: Only use with Paper Trading!")

    print("\n" + "=" * 60)
    print("Alpaca integration examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
