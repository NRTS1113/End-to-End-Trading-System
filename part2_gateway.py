"""
Part 2 Step 1: Gateway for Data Ingestion
FINM_32500 - End-to-End Trading System

Simulates live market data feed from historical files.
"""

import pandas as pd
from typing import Optional, Iterator, Dict
from datetime import datetime


class MarketDataGateway:
    """
    Gateway for streaming market data.
    Reads cleaned CSV files and streams data row-by-row to simulate real-time updates.
    """

    def __init__(self, data_file: str):
        """
        Initialize the market data gateway.

        Args:
            data_file: Path to CSV file with historical market data
        """
        self.data_file = data_file
        self.data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self.total_rows = 0

    def load_data(self):
        """
        Load market data from CSV file.
        """
        print(f"Loading market data from {self.data_file}...")
        self.data = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
        self.total_rows = len(self.data)
        self.current_index = 0
        print(f"Loaded {self.total_rows} rows of market data")

    def get_next_tick(self) -> Optional[Dict]:
        """
        Get the next market data tick.

        Returns:
            Dictionary with market data for current timestamp, or None if no more data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.current_index >= self.total_rows:
            return None

        row = self.data.iloc[self.current_index]
        tick = {
            'timestamp': self.data.index[self.current_index],
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }

        # Add any additional columns (indicators, signals, etc.)
        for col in self.data.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                tick[col.lower()] = row[col]

        self.current_index += 1
        return tick

    def stream_data(self) -> Iterator[Dict]:
        """
        Stream market data as an iterator.

        Yields:
            Market data tick dictionaries
        """
        if self.data is None:
            self.load_data()

        while True:
            tick = self.get_next_tick()
            if tick is None:
                break
            yield tick

    def reset(self):
        """
        Reset the gateway to start streaming from the beginning.
        """
        self.current_index = 0
        print("Gateway reset to beginning")

    def has_more_data(self) -> bool:
        """
        Check if there is more data to stream.

        Returns:
            True if more data available, False otherwise
        """
        return self.current_index < self.total_rows

    def get_progress(self) -> float:
        """
        Get current progress through the data.

        Returns:
            Progress as percentage (0-100)
        """
        if self.total_rows == 0:
            return 0.0
        return (self.current_index / self.total_rows) * 100


class OrderGateway:
    """
    Gateway for order logging and auditing.
    Writes all order events to a file for analysis.
    """

    def __init__(self, log_file: str = 'order_log.csv'):
        """
        Initialize the order gateway.

        Args:
            log_file: Path to order log file
        """
        self.log_file = log_file
        self.orders = []
        self._initialize_log()

    def _initialize_log(self):
        """
        Initialize the order log file with headers.
        """
        headers = [
            'timestamp', 'order_id', 'event_type', 'symbol', 'side',
            'quantity', 'price', 'order_type', 'status', 'filled_quantity',
            'remaining_quantity', 'fill_price', 'reason'
        ]

        with open(self.log_file, 'w') as f:
            f.write(','.join(headers) + '\n')

        print(f"Initialized order log: {self.log_file}")

    def log_order_sent(self, order: Dict):
        """
        Log when an order is sent.

        Args:
            order: Order dictionary
        """
        self._log_event(
            order_id=order['order_id'],
            event_type='SENT',
            symbol=order['symbol'],
            side=order['side'],
            quantity=order['quantity'],
            price=order.get('price', 0.0),
            order_type=order.get('order_type', 'MARKET'),
            status='PENDING'
        )

    def log_order_modified(self, order_id: str, old_order: Dict, new_order: Dict):
        """
        Log when an order is modified.

        Args:
            order_id: Order ID
            old_order: Original order
            new_order: Modified order
        """
        self._log_event(
            order_id=order_id,
            event_type='MODIFIED',
            symbol=new_order['symbol'],
            side=new_order['side'],
            quantity=new_order['quantity'],
            price=new_order.get('price', 0.0),
            order_type=new_order.get('order_type', 'MARKET'),
            status='PENDING'
        )

    def log_order_cancelled(self, order_id: str, order: Dict, reason: str = ''):
        """
        Log when an order is cancelled.

        Args:
            order_id: Order ID
            order: Order dictionary
            reason: Cancellation reason
        """
        self._log_event(
            order_id=order_id,
            event_type='CANCELLED',
            symbol=order['symbol'],
            side=order['side'],
            quantity=order['quantity'],
            price=order.get('price', 0.0),
            order_type=order.get('order_type', 'MARKET'),
            status='CANCELLED',
            reason=reason
        )

    def log_order_filled(self, order_id: str, order: Dict, filled_qty: int,
                        fill_price: float, is_partial: bool = False):
        """
        Log when an order is filled (fully or partially).

        Args:
            order_id: Order ID
            order: Order dictionary
            filled_qty: Filled quantity
            fill_price: Fill price
            is_partial: Whether this is a partial fill
        """
        status = 'PARTIAL_FILL' if is_partial else 'FILLED'
        remaining = order['quantity'] - filled_qty

        self._log_event(
            order_id=order_id,
            event_type='FILLED',
            symbol=order['symbol'],
            side=order['side'],
            quantity=order['quantity'],
            price=order.get('price', 0.0),
            order_type=order.get('order_type', 'MARKET'),
            status=status,
            filled_quantity=filled_qty,
            remaining_quantity=remaining,
            fill_price=fill_price
        )

    def log_order_rejected(self, order_id: str, order: Dict, reason: str):
        """
        Log when an order is rejected.

        Args:
            order_id: Order ID
            order: Order dictionary
            reason: Rejection reason
        """
        self._log_event(
            order_id=order_id,
            event_type='REJECTED',
            symbol=order['symbol'],
            side=order['side'],
            quantity=order['quantity'],
            price=order.get('price', 0.0),
            order_type=order.get('order_type', 'MARKET'),
            status='REJECTED',
            reason=reason
        )

    def _log_event(self, order_id: str, event_type: str, symbol: str,
                   side: str, quantity: int, price: float, order_type: str,
                   status: str, filled_quantity: int = 0,
                   remaining_quantity: int = 0, fill_price: float = 0.0,
                   reason: str = ''):
        """
        Write order event to log file.
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'order_id': order_id,
            'event_type': event_type,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'status': status,
            'filled_quantity': filled_quantity,
            'remaining_quantity': remaining_quantity,
            'fill_price': fill_price,
            'reason': reason
        }

        self.orders.append(log_entry)

        # Append to file
        with open(self.log_file, 'a') as f:
            values = [str(log_entry[k]) for k in [
                'timestamp', 'order_id', 'event_type', 'symbol', 'side',
                'quantity', 'price', 'order_type', 'status', 'filled_quantity',
                'remaining_quantity', 'fill_price', 'reason'
            ]]
            f.write(','.join(values) + '\n')

    def get_order_history(self, order_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get order history as DataFrame.

        Args:
            order_id: Optional order ID to filter by

        Returns:
            DataFrame with order history
        """
        df = pd.DataFrame(self.orders)

        if order_id is not None:
            df = df[df['order_id'] == order_id]

        return df


def main():
    """
    Example usage: Test gateway functionality.
    """
    print("=" * 60)
    print("Testing Market Data Gateway")
    print("=" * 60)

    # Test market data gateway
    try:
        gateway = MarketDataGateway('market_data_aapl.csv')
        gateway.load_data()

        print("\nStreaming first 10 ticks:")
        for i, tick in enumerate(gateway.stream_data()):
            if i >= 10:
                break
            print(f"Tick {i}: {tick['timestamp']} - Close: ${tick['close']:.2f}")

        print(f"\nProgress: {gateway.get_progress():.2f}%")

    except FileNotFoundError:
        print("Market data file not found. Run part1_data_download.py first.")

    print("\n" + "=" * 60)
    print("Testing Order Gateway")
    print("=" * 60)

    # Test order gateway
    order_gateway = OrderGateway('test_order_log.csv')

    # Simulate order lifecycle
    test_order = {
        'order_id': 'ORD001',
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 100,
        'price': 150.0,
        'order_type': 'LIMIT'
    }

    order_gateway.log_order_sent(test_order)
    order_gateway.log_order_filled('ORD001', test_order, 100, 149.95)

    print("\nOrder log created. Sample entries:")
    history = order_gateway.get_order_history()
    print(history)

    print("\n" + "=" * 60)
    print("Gateway testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
