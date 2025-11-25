"""
Part 2 Step 3: Order Manager & Gateway
FINM_32500 - End-to-End Trading System

Validates and records orders before execution.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid


class OrderManager:
    """
    Manages order validation and risk checks before execution.
    Implements capital sufficiency and risk limit checks.
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 max_orders_per_minute: int = 100,
                 max_position_pct: float = 0.3):
        """
        Initialize order manager.

        Args:
            initial_capital: Starting capital
            max_orders_per_minute: Maximum orders per minute (rate limit)
            max_position_pct: Maximum position as percentage of capital (0.3 = 30%)
        """
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.max_orders_per_minute = max_orders_per_minute
        self.max_position_pct = max_position_pct

        # Track positions by symbol
        self.positions: Dict[str, int] = defaultdict(int)  # Positive = long, negative = short

        # Track pending orders
        self.pending_orders: Dict[str, Dict] = {}

        # Track order timestamps for rate limiting
        self.order_timestamps: deque = deque()

        # Track capital committed to pending orders
        self.committed_capital: float = 0.0

        # Statistics
        self.total_orders = 0
        self.rejected_orders = 0
        self.rejection_reasons = defaultdict(int)

    def validate_order(self, order: Dict) -> Tuple[bool, str]:
        """
        Validate an order against all risk checks.

        Args:
            order: Order dictionary with keys: symbol, side, quantity, price, order_type

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order.get('price', 0.0)
        order_type = order.get('order_type', 'MARKET')

        # Check 1: Rate limiting (orders per minute)
        if not self._check_rate_limit():
            reason = f"Rate limit exceeded: max {self.max_orders_per_minute} orders/minute"
            self.rejection_reasons['rate_limit'] += 1
            return False, reason

        # Check 2: Capital sufficiency
        if not self._check_capital_sufficiency(order):
            reason = "Insufficient capital"
            self.rejection_reasons['insufficient_capital'] += 1
            return False, reason

        # Check 3: Position limits
        if not self._check_position_limits(order):
            reason = "Position limit exceeded"
            self.rejection_reasons['position_limit'] += 1
            return False, reason

        # Check 4: Order parameters
        if quantity <= 0:
            reason = "Invalid quantity"
            self.rejection_reasons['invalid_params'] += 1
            return False, reason

        if order_type == 'LIMIT' and price <= 0:
            reason = "Invalid limit price"
            self.rejection_reasons['invalid_params'] += 1
            return False, reason

        return True, ""

    def submit_order(self, order: Dict, gateway) -> Tuple[bool, str, Optional[str]]:
        """
        Submit an order after validation.

        Args:
            order: Order dictionary
            gateway: OrderGateway instance for logging

        Returns:
            Tuple of (success, message, order_id)
        """
        self.total_orders += 1

        # Validate order
        is_valid, reason = self.validate_order(order)

        if not is_valid:
            self.rejected_orders += 1

            # Generate order ID for rejected order
            order_id = f"ORD_{uuid.uuid4().hex[:8]}"
            order['order_id'] = order_id

            # Log rejection
            gateway.log_order_rejected(order_id, order, reason)

            return False, reason, None

        # Generate order ID
        order_id = f"ORD_{uuid.uuid4().hex[:8]}"
        order['order_id'] = order_id

        # Reserve capital
        self._reserve_capital(order)

        # Track pending order
        self.pending_orders[order_id] = order

        # Record timestamp for rate limiting
        self.order_timestamps.append(datetime.now())

        # Log order sent
        gateway.log_order_sent(order)

        return True, "Order submitted successfully", order_id

    def cancel_order(self, order_id: str, gateway) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel
            gateway: OrderGateway for logging

        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.pending_orders:
            return False

        order = self.pending_orders[order_id]

        # Release reserved capital
        self._release_capital(order)

        # Remove from pending
        del self.pending_orders[order_id]

        # Log cancellation
        gateway.log_order_cancelled(order_id, order, "User cancelled")

        return True

    def order_filled(self, order_id: str, filled_qty: int, fill_price: float, gateway):
        """
        Handle order fill notification.

        Args:
            order_id: Filled order ID
            filled_qty: Filled quantity
            fill_price: Fill price
            gateway: OrderGateway for logging
        """
        if order_id not in self.pending_orders:
            return

        order = self.pending_orders[order_id]
        symbol = order['symbol']
        side = order['side']

        # Update position
        if side == 'BUY':
            self.positions[symbol] += filled_qty
            # Deduct capital
            cost = filled_qty * fill_price
            self.available_capital -= cost
        else:  # SELL
            self.positions[symbol] -= filled_qty
            # Add capital
            proceeds = filled_qty * fill_price
            self.available_capital += proceeds

        # Release reserved capital
        self._release_capital(order)

        # Check if partial or full fill
        is_partial = filled_qty < order['quantity']

        # Log fill
        gateway.log_order_filled(order_id, order, filled_qty, fill_price, is_partial)

        # Remove from pending if fully filled
        if not is_partial:
            del self.pending_orders[order_id]
        else:
            # Update remaining quantity
            order['quantity'] -= filled_qty
            # Re-reserve capital for remaining quantity
            self._reserve_capital(order)

    def _check_rate_limit(self) -> bool:
        """
        Check if order rate limit is exceeded.

        Returns:
            True if within limit, False otherwise
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove timestamps older than 1 minute
        while self.order_timestamps and self.order_timestamps[0] < cutoff:
            self.order_timestamps.popleft()

        # Check if we've hit the limit
        return len(self.order_timestamps) < self.max_orders_per_minute

    def _check_capital_sufficiency(self, order: Dict) -> bool:
        """
        Check if sufficient capital exists to execute the order.

        Args:
            order: Order dictionary

        Returns:
            True if sufficient capital, False otherwise
        """
        if order['side'] != 'BUY':
            # No capital needed for sell orders
            return True

        # Estimate required capital
        quantity = order['quantity']
        price = order.get('price', 0.0)

        if order.get('order_type') == 'MARKET':
            # For market orders, use a conservative estimate
            # Assume price could move 5% against us
            estimated_price = price * 1.05 if price > 0 else 0
        else:
            estimated_price = price

        required_capital = quantity * estimated_price

        # Check if we have enough available capital
        available = self.available_capital - self.committed_capital

        return available >= required_capital

    def _check_position_limits(self, order: Dict) -> bool:
        """
        Check if executing the order would exceed position limits.

        Args:
            order: Order dictionary

        Returns:
            True if within limits, False otherwise
        """
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order.get('price', 0.0)

        # Calculate new position
        current_position = self.positions[symbol]

        if side == 'BUY':
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity

        # Calculate position value (use price or estimate)
        if price > 0:
            position_value = abs(new_position) * price
        else:
            # Conservative estimate if no price
            return True

        # Check against maximum position size
        max_position_value = self.initial_capital * self.max_position_pct

        return position_value <= max_position_value

    def _reserve_capital(self, order: Dict):
        """
        Reserve capital for a pending order.

        Args:
            order: Order dictionary
        """
        if order['side'] == 'BUY':
            quantity = order['quantity']
            price = order.get('price', 0.0)

            if price > 0:
                reserved = quantity * price
                self.committed_capital += reserved

    def _release_capital(self, order: Dict):
        """
        Release reserved capital for an order.

        Args:
            order: Order dictionary
        """
        if order['side'] == 'BUY':
            quantity = order['quantity']
            price = order.get('price', 0.0)

            if price > 0:
                released = quantity * price
                self.committed_capital = max(0, self.committed_capital - released)

    def get_statistics(self) -> Dict:
        """
        Get order manager statistics.

        Returns:
            Dictionary with statistics
        """
        acceptance_rate = (
            (self.total_orders - self.rejected_orders) / self.total_orders * 100
            if self.total_orders > 0 else 0
        )

        return {
            'total_orders': self.total_orders,
            'accepted_orders': self.total_orders - self.rejected_orders,
            'rejected_orders': self.rejected_orders,
            'acceptance_rate': acceptance_rate,
            'available_capital': self.available_capital,
            'committed_capital': self.committed_capital,
            'positions': dict(self.positions),
            'pending_orders': len(self.pending_orders),
            'rejection_reasons': dict(self.rejection_reasons)
        }

    def reset(self):
        """Reset order manager to initial state"""
        self.available_capital = self.initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.order_timestamps.clear()
        self.committed_capital = 0.0
        self.total_orders = 0
        self.rejected_orders = 0
        self.rejection_reasons.clear()


def main():
    """
    Example usage: Test order manager.
    """
    from part2_gateway import OrderGateway

    print("=" * 60)
    print("Testing Order Manager")
    print("=" * 60)

    # Create order manager and gateway
    manager = OrderManager(
        initial_capital=100000.0,
        max_orders_per_minute=10,
        max_position_pct=0.3
    )

    gateway = OrderGateway('test_order_manager_log.csv')

    print(f"\nInitial capital: ${manager.available_capital:,.2f}")

    # Test 1: Valid order
    print("\nTest 1: Submitting valid order...")
    order1 = {
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 100,
        'price': 150.0,
        'order_type': 'LIMIT'
    }

    success, msg, order_id = manager.submit_order(order1, gateway)
    print(f"  Result: {success}, Message: {msg}, Order ID: {order_id}")

    # Test 2: Insufficient capital
    print("\nTest 2: Order with insufficient capital...")
    order2 = {
        'symbol': 'GOOGL',
        'side': 'BUY',
        'quantity': 10000,
        'price': 150.0,
        'order_type': 'LIMIT'
    }

    success, msg, order_id = manager.submit_order(order2, gateway)
    print(f"  Result: {success}, Message: {msg}")

    # Test 3: Rate limiting
    print("\nTest 3: Testing rate limit...")
    for i in range(12):
        order = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 10,
            'price': 150.0,
            'order_type': 'LIMIT'
        }
        success, msg, order_id = manager.submit_order(order, gateway)
        if not success:
            print(f"  Order {i+1}: Rejected - {msg}")

    # Test 4: Order fill
    print("\nTest 4: Simulating order fill...")
    if manager.pending_orders:
        first_order_id = list(manager.pending_orders.keys())[0]
        first_order = manager.pending_orders[first_order_id]
        manager.order_filled(first_order_id, first_order['quantity'], 149.50, gateway)
        print(f"  Filled order {first_order_id}")
        print(f"  New capital: ${manager.available_capital:,.2f}")
        print(f"  Position in AAPL: {manager.positions['AAPL']}")

    # Show statistics
    print("\nOrder Manager Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Order manager testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
