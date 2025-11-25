"""
Part 2 Step 2: Order Book Implementation
FINM_32500 - End-to-End Trading System

Manages and matches bid/ask orders using efficient data structures.
"""

import heapq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass(order=True)
class Order:
    """
    Represents a single order in the order book.
    Uses dataclass with ordering for heap operations.
    """
    priority: Tuple[float, float] = field(compare=True)  # (price, timestamp)
    order_id: str = field(compare=False)
    symbol: str = field(compare=False)
    side: str = field(compare=False)  # 'BUY' or 'SELL'
    quantity: int = field(compare=False)
    price: float = field(compare=False)
    order_type: str = field(compare=False)  # 'LIMIT' or 'MARKET'
    timestamp: float = field(compare=False)
    filled_quantity: int = field(default=0, compare=False)
    status: str = field(default='ACTIVE', compare=False)  # ACTIVE, FILLED, CANCELLED

    @property
    def remaining_quantity(self) -> int:
        """Get remaining unfilled quantity"""
        return self.quantity - self.filled_quantity

    def is_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.filled_quantity >= self.quantity

    def fill(self, quantity: int) -> int:
        """
        Fill the order with specified quantity.

        Args:
            quantity: Quantity to fill

        Returns:
            Actual quantity filled
        """
        fillable = min(quantity, self.remaining_quantity)
        self.filled_quantity += fillable

        if self.is_filled():
            self.status = 'FILLED'

        return fillable


class OrderBook:
    """
    Order book implementation using heaps for efficient price-time priority matching.
    Maintains separate bid and ask heaps.
    """

    def __init__(self, symbol: str):
        """
        Initialize order book for a symbol.

        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol

        # Use heaps for price-time priority
        # Bids: max heap (negative prices for max behavior)
        # Asks: min heap (positive prices for min behavior)
        self.bids: List[Order] = []  # Max heap (highest price first)
        self.asks: List[Order] = []  # Min heap (lowest price first)

        # Track orders by ID for quick lookup
        self.orders: Dict[str, Order] = {}

        # Track order history
        self.trade_history: List[Dict] = []

    def add_order(self, order_id: str, side: str, quantity: int,
                  price: Optional[float] = None, order_type: str = 'LIMIT') -> Order:
        """
        Add a new order to the order book.

        Args:
            order_id: Unique order identifier
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Limit price (None for market orders)
            order_type: 'LIMIT' or 'MARKET'

        Returns:
            Created Order object
        """
        if order_id in self.orders:
            raise ValueError(f"Order {order_id} already exists")

        timestamp = datetime.now().timestamp()

        # For market orders, use extreme prices
        if order_type == 'MARKET':
            if side == 'BUY':
                price = float('inf')  # Buy at any price
            else:
                price = 0.0  # Sell at any price

        # Create priority tuple (price, timestamp) for heap ordering
        if side == 'BUY':
            # For buy orders: higher price = higher priority (use negative for max heap)
            priority = (-price, timestamp)
        else:
            # For sell orders: lower price = higher priority
            priority = (price, timestamp)

        order = Order(
            priority=priority,
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            timestamp=timestamp
        )

        # Add to appropriate heap
        if side == 'BUY':
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

        # Track in orders dict
        self.orders[order_id] = order

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        order.status = 'CANCELLED'

        # Note: We don't remove from heap immediately (lazy deletion)
        # Will be cleaned up during matching or when order is accessed

        return True

    def modify_order(self, order_id: str, new_quantity: Optional[int] = None,
                    new_price: Optional[float] = None) -> bool:
        """
        Modify an existing order.
        Implemented as cancel + add new order.

        Args:
            order_id: Order ID to modify
            new_quantity: New quantity (None to keep current)
            new_price: New price (None to keep current)

        Returns:
            True if modified, False if not found
        """
        if order_id not in self.orders:
            return False

        old_order = self.orders[order_id]

        if old_order.status != 'ACTIVE':
            return False

        # Cancel old order
        self.cancel_order(order_id)

        # Add new order with updated parameters
        quantity = new_quantity if new_quantity is not None else old_order.quantity
        price = new_price if new_price is not None else old_order.price

        self.add_order(
            order_id=f"{order_id}_MOD",
            side=old_order.side,
            quantity=quantity,
            price=price,
            order_type=old_order.order_type
        )

        return True

    def match_orders(self) -> List[Dict]:
        """
        Match buy and sell orders based on price-time priority.

        Returns:
            List of trade dictionaries
        """
        trades = []

        while self.bids and self.asks:
            # Get best bid and ask (without removing yet)
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            # Skip cancelled or filled orders
            if best_bid.status != 'ACTIVE' or best_bid.is_filled():
                heapq.heappop(self.bids)
                continue

            if best_ask.status != 'ACTIVE' or best_ask.is_filled():
                heapq.heappop(self.asks)
                continue

            # Check if orders can match
            # For buy: bid price >= ask price (remember bid price is stored as negative)
            bid_price = -best_bid.priority[0]  # Convert back to positive
            ask_price = best_ask.priority[0]

            if bid_price < ask_price:
                # No match possible
                break

            # Match orders
            match_quantity = min(best_bid.remaining_quantity, best_ask.remaining_quantity)

            # Determine trade price (use ask price by convention)
            trade_price = ask_price if ask_price != 0 else bid_price

            # Fill orders
            best_bid.fill(match_quantity)
            best_ask.fill(match_quantity)

            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'buy_order_id': best_bid.order_id,
                'sell_order_id': best_ask.order_id,
                'quantity': match_quantity,
                'price': trade_price
            }

            trades.append(trade)
            self.trade_history.append(trade)

            # Remove filled orders from heaps
            if best_bid.is_filled():
                heapq.heappop(self.bids)

            if best_ask.is_filled():
                heapq.heappop(self.asks)

        return trades

    def get_best_bid(self) -> Optional[float]:
        """
        Get best (highest) bid price.

        Returns:
            Best bid price or None if no bids
        """
        while self.bids:
            best = self.bids[0]
            if best.status == 'ACTIVE' and not best.is_filled():
                return -best.priority[0]  # Convert back to positive
            heapq.heappop(self.bids)

        return None

    def get_best_ask(self) -> Optional[float]:
        """
        Get best (lowest) ask price.

        Returns:
            Best ask price or None if no asks
        """
        while self.asks:
            best = self.asks[0]
            if best.status == 'ACTIVE' and not best.is_filled():
                return best.priority[0]
            heapq.heappop(self.asks)

        return None

    def get_spread(self) -> Optional[float]:
        """
        Get bid-ask spread.

        Returns:
            Spread or None if no bids or asks
        """
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid is not None and ask is not None:
            return ask - bid

        return None

    def get_depth(self, levels: int = 5) -> Dict:
        """
        Get order book depth (top N levels).

        Args:
            levels: Number of price levels to return

        Returns:
            Dictionary with bid and ask depth
        """
        bid_depth = []
        ask_depth = []

        # Get bid depth
        bid_prices = {}
        for order in self.bids:
            if order.status == 'ACTIVE' and not order.is_filled():
                price = -order.priority[0]
                if price not in bid_prices:
                    bid_prices[price] = 0
                bid_prices[price] += order.remaining_quantity

        for price in sorted(bid_prices.keys(), reverse=True)[:levels]:
            bid_depth.append({'price': price, 'quantity': bid_prices[price]})

        # Get ask depth
        ask_prices = {}
        for order in self.asks:
            if order.status == 'ACTIVE' and not order.is_filled():
                price = order.priority[0]
                if price not in ask_prices:
                    ask_prices[price] = 0
                ask_prices[price] += order.remaining_quantity

        for price in sorted(ask_prices.keys())[:levels]:
            ask_depth.append({'price': price, 'quantity': ask_prices[price]})

        return {
            'bids': bid_depth,
            'asks': ask_depth
        }

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get status of an order.

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary or None if not found
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'price': order.price,
            'status': order.status
        }


def main():
    """
    Example usage: Test order book functionality.
    """
    print("=" * 60)
    print("Testing Order Book")
    print("=" * 60)

    # Create order book
    book = OrderBook('AAPL')

    # Add some orders
    print("\nAdding orders to book...")
    book.add_order('BUY1', 'BUY', 100, 150.00)
    book.add_order('BUY2', 'BUY', 200, 149.50)
    book.add_order('BUY3', 'BUY', 150, 149.75)

    book.add_order('SELL1', 'SELL', 100, 150.50)
    book.add_order('SELL2', 'SELL', 200, 151.00)
    book.add_order('SELL3', 'SELL', 150, 150.25)

    # Display book depth
    print("\nOrder book depth:")
    depth = book.get_depth(levels=3)
    print("Bids:")
    for level in depth['bids']:
        print(f"  {level['quantity']} @ ${level['price']:.2f}")

    print("Asks:")
    for level in depth['asks']:
        print(f"  {level['quantity']} @ ${level['price']:.2f}")

    # Best bid/ask
    print(f"\nBest bid: ${book.get_best_bid():.2f}")
    print(f"Best ask: ${book.get_best_ask():.2f}")
    print(f"Spread: ${book.get_spread():.2f}")

    # Add crossing order to trigger match
    print("\nAdding buy order that crosses the spread...")
    book.add_order('BUY4', 'BUY', 250, 151.00)

    # Match orders
    print("\nMatching orders...")
    trades = book.match_orders()

    print(f"\nExecuted {len(trades)} trades:")
    for trade in trades:
        print(f"  {trade['quantity']} shares @ ${trade['price']:.2f}")
        print(f"    Buy order: {trade['buy_order_id']}, Sell order: {trade['sell_order_id']}")

    # Check order status
    print("\nOrder status:")
    for order_id in ['BUY4', 'SELL3', 'SELL1']:
        status = book.get_order_status(order_id)
        if status:
            print(f"  {order_id}: {status['filled_quantity']}/{status['quantity']} filled - {status['status']}")

    print("\n" + "=" * 60)
    print("Order book testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
