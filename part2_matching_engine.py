"""
Part 2 Step 4: Matching Engine Simulator
FINM_32500 - End-to-End Trading System

Simulates realistic order execution outcomes.
"""

import random
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ExecutionOutcome(Enum):
    """Possible execution outcomes"""
    FILLED = "FILLED"
    PARTIAL_FILL = "PARTIAL_FILL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class MatchingEngine:
    """
    Simulates order matching and execution with realistic outcomes.
    Randomly determines whether orders are filled, partially filled, or cancelled.
    """

    def __init__(self,
                 fill_probability: float = 0.85,
                 partial_fill_probability: float = 0.10,
                 cancel_probability: float = 0.05,
                 slippage_range: Tuple[float, float] = (-0.001, 0.001)):
        """
        Initialize matching engine.

        Args:
            fill_probability: Probability of full fill (0-1)
            partial_fill_probability: Probability of partial fill (0-1)
            cancel_probability: Probability of cancellation (0-1)
            slippage_range: Range of price slippage as fraction (min, max)
        """
        # Normalize probabilities to sum to 1
        total = fill_probability + partial_fill_probability + cancel_probability
        self.fill_prob = fill_probability / total
        self.partial_prob = partial_fill_probability / total
        self.cancel_prob = cancel_probability / total

        self.slippage_range = slippage_range

        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.partial_fills = 0
        self.cancelled_orders = 0
        self.total_slippage = 0.0

    def execute_order(self, order: Dict, market_price: float) -> Dict:
        """
        Simulate order execution.

        Args:
            order: Order dictionary with keys: order_id, symbol, side, quantity, price, order_type
            market_price: Current market price

        Returns:
            Execution result dictionary
        """
        self.total_orders += 1

        order_id = order['order_id']
        order_type = order.get('order_type', 'MARKET')
        side = order['side']
        quantity = order['quantity']
        limit_price = order.get('price', 0.0)

        # Determine execution outcome
        outcome = self._determine_outcome(order, market_price)

        # Generate execution result based on outcome
        if outcome == ExecutionOutcome.FILLED:
            return self._execute_full_fill(order, market_price)

        elif outcome == ExecutionOutcome.PARTIAL_FILL:
            return self._execute_partial_fill(order, market_price)

        elif outcome == ExecutionOutcome.CANCELLED:
            return self._execute_cancellation(order)

        else:  # REJECTED
            return self._execute_rejection(order)

    def _determine_outcome(self, order: Dict, market_price: float) -> ExecutionOutcome:
        """
        Determine execution outcome based on probabilities and market conditions.

        Args:
            order: Order dictionary
            market_price: Current market price

        Returns:
            ExecutionOutcome
        """
        order_type = order.get('order_type', 'MARKET')
        side = order['side']
        limit_price = order.get('price', 0.0)

        # Market orders have higher fill probability
        if order_type == 'MARKET':
            # 95% filled, 4% partial, 1% cancelled
            rand = random.random()
            if rand < 0.95:
                return ExecutionOutcome.FILLED
            elif rand < 0.99:
                return ExecutionOutcome.PARTIAL_FILL
            else:
                return ExecutionOutcome.CANCELLED

        # Limit orders depend on price
        else:
            # Check if limit price is marketable
            is_marketable = False
            if side == 'BUY' and limit_price >= market_price:
                is_marketable = True
            elif side == 'SELL' and limit_price <= market_price:
                is_marketable = True

            if is_marketable:
                # Marketable limit order - higher fill probability
                rand = random.random()
                if rand < self.fill_prob * 1.2:  # Boost fill probability
                    return ExecutionOutcome.FILLED
                elif rand < (self.fill_prob * 1.2 + self.partial_prob):
                    return ExecutionOutcome.PARTIAL_FILL
                else:
                    return ExecutionOutcome.CANCELLED
            else:
                # Non-marketable limit order - lower fill probability
                rand = random.random()
                if rand < self.fill_prob * 0.5:  # Reduce fill probability
                    return ExecutionOutcome.FILLED
                elif rand < (self.fill_prob * 0.5 + self.partial_prob * 1.5):
                    return ExecutionOutcome.PARTIAL_FILL
                else:
                    return ExecutionOutcome.CANCELLED

    def _execute_full_fill(self, order: Dict, market_price: float) -> Dict:
        """
        Execute a full fill.

        Args:
            order: Order dictionary
            market_price: Current market price

        Returns:
            Execution result
        """
        self.filled_orders += 1

        # Calculate execution price with slippage
        exec_price = self._calculate_execution_price(order, market_price)

        return {
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'outcome': ExecutionOutcome.FILLED.value,
            'filled_quantity': order['quantity'],
            'remaining_quantity': 0,
            'execution_price': exec_price,
            'total_value': order['quantity'] * exec_price,
            'message': 'Order fully filled'
        }

    def _execute_partial_fill(self, order: Dict, market_price: float) -> Dict:
        """
        Execute a partial fill.

        Args:
            order: Order dictionary
            market_price: Current market price

        Returns:
            Execution result
        """
        self.partial_fills += 1

        # Random fill percentage (20% to 80%)
        fill_pct = random.uniform(0.2, 0.8)
        filled_qty = int(order['quantity'] * fill_pct)
        filled_qty = max(1, filled_qty)  # At least 1 share

        remaining_qty = order['quantity'] - filled_qty

        # Calculate execution price with slippage
        exec_price = self._calculate_execution_price(order, market_price)

        return {
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'outcome': ExecutionOutcome.PARTIAL_FILL.value,
            'filled_quantity': filled_qty,
            'remaining_quantity': remaining_qty,
            'execution_price': exec_price,
            'total_value': filled_qty * exec_price,
            'message': f'Order partially filled: {filled_qty}/{order["quantity"]}'
        }

    def _execute_cancellation(self, order: Dict) -> Dict:
        """
        Execute a cancellation.

        Args:
            order: Order dictionary

        Returns:
            Execution result
        """
        self.cancelled_orders += 1

        # Random cancellation reasons
        reasons = [
            'Insufficient liquidity',
            'Market volatility',
            'Price moved away',
            'Timeout',
            'Risk limit'
        ]

        return {
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'outcome': ExecutionOutcome.CANCELLED.value,
            'filled_quantity': 0,
            'remaining_quantity': order['quantity'],
            'execution_price': 0.0,
            'total_value': 0.0,
            'message': f'Order cancelled: {random.choice(reasons)}'
        }

    def _execute_rejection(self, order: Dict) -> Dict:
        """
        Execute a rejection.

        Args:
            order: Order dictionary

        Returns:
            Execution result
        """
        reasons = [
            'Invalid price',
            'Exchange rejected',
            'Symbol halted',
            'Outside market hours'
        ]

        return {
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'outcome': ExecutionOutcome.REJECTED.value,
            'filled_quantity': 0,
            'remaining_quantity': order['quantity'],
            'execution_price': 0.0,
            'total_value': 0.0,
            'message': f'Order rejected: {random.choice(reasons)}'
        }

    def _calculate_execution_price(self, order: Dict, market_price: float) -> float:
        """
        Calculate execution price with slippage.

        Args:
            order: Order dictionary
            market_price: Current market price

        Returns:
            Execution price
        """
        order_type = order.get('order_type', 'MARKET')
        side = order['side']
        limit_price = order.get('price', 0.0)

        # Base price
        if order_type == 'LIMIT':
            # For limit orders, may get price improvement
            if side == 'BUY':
                # Might get filled at better price than limit
                base_price = random.uniform(market_price * 0.999, limit_price)
            else:
                # Might get filled at better price than limit
                base_price = random.uniform(limit_price, market_price * 1.001)
        else:
            base_price = market_price

        # Add slippage
        slippage = random.uniform(*self.slippage_range)

        # Slippage direction depends on side
        if side == 'BUY':
            # Buyers experience positive slippage (pay more)
            exec_price = base_price * (1 + abs(slippage))
        else:
            # Sellers experience negative slippage (receive less)
            exec_price = base_price * (1 - abs(slippage))

        self.total_slippage += slippage

        return round(exec_price, 2)

    def get_statistics(self) -> Dict:
        """
        Get matching engine statistics.

        Returns:
            Dictionary with statistics
        """
        if self.total_orders == 0:
            return {
                'total_orders': 0,
                'fill_rate': 0.0,
                'partial_fill_rate': 0.0,
                'cancel_rate': 0.0,
                'avg_slippage': 0.0
            }

        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'partial_fills': self.partial_fills,
            'cancelled_orders': self.cancelled_orders,
            'fill_rate': self.filled_orders / self.total_orders * 100,
            'partial_fill_rate': self.partial_fills / self.total_orders * 100,
            'cancel_rate': self.cancelled_orders / self.total_orders * 100,
            'avg_slippage': self.total_slippage / self.total_orders
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.total_orders = 0
        self.filled_orders = 0
        self.partial_fills = 0
        self.cancelled_orders = 0
        self.total_slippage = 0.0


def main():
    """
    Example usage: Test matching engine.
    """
    print("=" * 60)
    print("Testing Matching Engine")
    print("=" * 60)

    # Create matching engine
    engine = MatchingEngine(
        fill_probability=0.80,
        partial_fill_probability=0.15,
        cancel_probability=0.05
    )

    market_price = 150.0

    # Test various order types
    print(f"\nMarket price: ${market_price:.2f}\n")

    test_orders = [
        {
            'order_id': 'TEST001',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'order_type': 'MARKET'
        },
        {
            'order_id': 'TEST002',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 200,
            'price': 150.50,
            'order_type': 'LIMIT'
        },
        {
            'order_id': 'TEST003',
            'symbol': 'AAPL',
            'side': 'SELL',
            'quantity': 150,
            'price': 149.50,
            'order_type': 'LIMIT'
        },
        {
            'order_id': 'TEST004',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 300,
            'price': 148.00,
            'order_type': 'LIMIT'
        },
    ]

    print("Executing test orders:")
    for order in test_orders:
        result = engine.execute_order(order, market_price)
        print(f"\nOrder {result['order_id']}:")
        print(f"  Outcome: {result['outcome']}")
        print(f"  Filled: {result['filled_quantity']}/{result['filled_quantity'] + result['remaining_quantity']}")
        if result['filled_quantity'] > 0:
            print(f"  Execution Price: ${result['execution_price']:.2f}")
            print(f"  Total Value: ${result['total_value']:.2f}")
        print(f"  Message: {result['message']}")

    # Run simulation with many orders
    print("\n" + "=" * 60)
    print("Running simulation with 100 orders...")
    print("=" * 60)

    engine.reset_statistics()

    for i in range(100):
        order = {
            'order_id': f'SIM{i:03d}',
            'symbol': 'AAPL',
            'side': random.choice(['BUY', 'SELL']),
            'quantity': random.randint(10, 500),
            'price': market_price + random.uniform(-2, 2),
            'order_type': random.choice(['MARKET', 'LIMIT', 'LIMIT'])
        }
        engine.execute_order(order, market_price)

    # Show statistics
    print("\nMatching Engine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}{'%' if 'rate' in key else ''}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Matching engine testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
