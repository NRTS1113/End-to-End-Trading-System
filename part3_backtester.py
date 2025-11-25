"""
Part 3: Strategy Backtesting Framework
FINM_32500 - End-to-End Trading System

Integrates all components to backtest trading strategies with performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import json

from part1_strategy import BaseStrategy, Signal
from part2_gateway import MarketDataGateway, OrderGateway
from part2_order_manager import OrderManager
from part2_matching_engine import MatchingEngine


class Backtester:
    """
    Complete backtesting framework that integrates:
    - Market data gateway
    - Trading strategy
    - Order manager
    - Matching engine
    - Performance analytics
    """

    def __init__(self,
                 strategy: BaseStrategy,
                 data_file: str,
                 initial_capital: float = 100000.0,
                 position_size: int = 100,
                 order_log_file: str = 'backtest_orders.csv'):
        """
        Initialize backtester.

        Args:
            strategy: Trading strategy instance
            data_file: Path to market data CSV
            initial_capital: Starting capital
            position_size: Default position size per trade
            order_log_file: Path to order log file
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size = position_size

        # Initialize components
        self.market_gateway = MarketDataGateway(data_file)
        self.order_gateway = OrderGateway(order_log_file)
        self.order_manager = OrderManager(
            initial_capital=initial_capital,
            max_orders_per_minute=100,
            max_position_pct=0.5
        )
        self.matching_engine = MatchingEngine(
            fill_probability=0.85,
            partial_fill_probability=0.10,
            cancel_probability=0.05
        )

        # Track state
        self.current_position = 0
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.timestamps: List = []

        # Performance metrics
        self.metrics: Dict = {}

    def run(self) -> pd.DataFrame:
        """
        Run the backtest.

        Returns:
            DataFrame with backtest results
        """
        print("=" * 60)
        print(f"Running Backtest: {self.strategy.name}")
        print("=" * 60)

        # Load market data
        self.market_gateway.load_data()

        # Generate strategy signals
        print("\nGenerating strategy signals...")
        signals = self.strategy.generate_signals(self.market_gateway.data)

        # Initialize tracking
        capital = self.initial_capital
        position = 0
        pending_order_id = None

        # Stream through data
        print("\nSimulating trading...")
        for tick in self.market_gateway.stream_data():
            timestamp = tick['timestamp']
            price = tick['close']
            signal = tick.get('signal', Signal.HOLD.value)

            # Update equity curve
            equity = self.order_manager.available_capital + (position * price)
            self.equity_curve.append(equity)
            self.timestamps.append(timestamp)

            # Process signal
            if signal == Signal.BUY.value and position <= 0:
                # Buy signal - enter or reverse position
                order = {
                    'symbol': 'SYMBOL',
                    'side': 'BUY',
                    'quantity': self.position_size,
                    'price': price,
                    'order_type': 'MARKET'
                }

                # Submit order
                success, msg, order_id = self.order_manager.submit_order(order, self.order_gateway)

                if success:
                    # Execute order
                    result = self.matching_engine.execute_order(order, price)

                    if result['filled_quantity'] > 0:
                        # Order filled
                        self.order_manager.order_filled(
                            order_id,
                            result['filled_quantity'],
                            result['execution_price'],
                            self.order_gateway
                        )

                        position += result['filled_quantity']

                        # Record trade
                        self.trades.append({
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'quantity': result['filled_quantity'],
                            'price': result['execution_price'],
                            'value': result['total_value']
                        })

            elif signal == Signal.SELL.value and position >= 0:
                # Sell signal - exit or reverse position
                quantity = max(self.position_size, position)

                order = {
                    'symbol': 'SYMBOL',
                    'side': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'order_type': 'MARKET'
                }

                # Submit order
                success, msg, order_id = self.order_manager.submit_order(order, self.order_gateway)

                if success:
                    # Execute order
                    result = self.matching_engine.execute_order(order, price)

                    if result['filled_quantity'] > 0:
                        # Order filled
                        self.order_manager.order_filled(
                            order_id,
                            result['filled_quantity'],
                            result['execution_price'],
                            self.order_gateway
                        )

                        position -= result['filled_quantity']

                        # Record trade
                        self.trades.append({
                            'timestamp': timestamp,
                            'action': 'SELL',
                            'quantity': result['filled_quantity'],
                            'price': result['execution_price'],
                            'value': result['total_value']
                        })

        # Calculate final equity
        final_equity = self.order_manager.available_capital + (position * price)

        print(f"\nBacktest complete!")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Final equity: ${final_equity:,.2f}")
        print(f"Total return: {(final_equity - self.initial_capital) / self.initial_capital * 100:.2f}%")
        print(f"Total trades: {len(self.trades)}")

        # Calculate performance metrics
        self._calculate_metrics()

        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve
        })

        return results

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Basic metrics
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        self.metrics['total_return'] = total_return * 100

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252 * 390) * returns.mean() / returns.std()  # 390 minutes per day
            self.metrics['sharpe_ratio'] = sharpe
        else:
            self.metrics['sharpe_ratio'] = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        self.metrics['max_drawdown'] = drawdown.min() * 100

        # Win rate
        if len(self.trades) >= 2:
            winning_trades = 0
            for i in range(1, len(self.trades)):
                if self.trades[i]['action'] == 'SELL':
                    # Find corresponding buy
                    for j in range(i - 1, -1, -1):
                        if self.trades[j]['action'] == 'BUY':
                            pnl = (self.trades[i]['price'] - self.trades[j]['price']) * self.trades[i]['quantity']
                            if pnl > 0:
                                winning_trades += 1
                            break

            total_round_trips = len([t for t in self.trades if t['action'] == 'SELL'])
            self.metrics['win_rate'] = (winning_trades / total_round_trips * 100) if total_round_trips > 0 else 0
        else:
            self.metrics['win_rate'] = 0.0

        # Trade statistics
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['buy_trades'] = len([t for t in self.trades if t['action'] == 'BUY'])
        self.metrics['sell_trades'] = len([t for t in self.trades if t['action'] == 'SELL'])

        # Order manager stats
        om_stats = self.order_manager.get_statistics()
        self.metrics['order_acceptance_rate'] = om_stats['acceptance_rate']
        self.metrics['rejected_orders'] = om_stats['rejected_orders']

        # Matching engine stats
        me_stats = self.matching_engine.get_statistics()
        self.metrics['fill_rate'] = me_stats['fill_rate']
        self.metrics['avg_slippage'] = me_stats['avg_slippage']

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.

        Args:
            save_path: Path to save plot (optional)
        """
        if not self.equity_curve:
            print("No results to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Equity curve
        axes[0].plot(self.timestamps, self.equity_curve, label='Equity', linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title(f'Equity Curve - {self.strategy.name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Returns distribution
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        axes[1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[1].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Returns')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self, filename: str = 'backtest_report.txt'):
        """
        Generate a text report of backtest results.

        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"BACKTEST REPORT - {self.strategy.name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 60 + "\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    f.write(f"{key:.<40} {value:.2f}\n")
                else:
                    f.write(f"{key:.<40} {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("TRADE SUMMARY\n")
            f.write("=" * 60 + "\n")

            if self.trades:
                f.write(f"\nFirst 10 trades:\n")
                for i, trade in enumerate(self.trades[:10]):
                    f.write(f"\n{i+1}. {trade['timestamp']}\n")
                    f.write(f"   Action: {trade['action']}\n")
                    f.write(f"   Quantity: {trade['quantity']}\n")
                    f.write(f"   Price: ${trade['price']:.2f}\n")
                    f.write(f"   Value: ${trade['value']:.2f}\n")

        print(f"Report saved to {filename}")

    def save_results(self, filename: str = 'backtest_results.json'):
        """
        Save backtest results to JSON.

        Args:
            filename: Output filename
        """
        results = {
            'strategy': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1] if self.equity_curve else 0,
            'metrics': self.metrics,
            'trades': self.trades
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filename}")


def main():
    """
    Example usage: Run backtests with different strategies.
    """
    from part1_strategy import (
        MomentumStrategy,
        MovingAverageCrossoverStrategy,
        RSIStrategy,
        BollingerBandsStrategy,
        CombinedStrategy
    )

    # Check if data exists
    try:
        # Test strategies
        strategies = [
            MovingAverageCrossoverStrategy(short_window=5, long_window=20),
            MomentumStrategy(lookback_period=10),
            RSIStrategy(rsi_period=14),
            BollingerBandsStrategy(period=20),
            CombinedStrategy()
        ]

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Backtesting: {strategy.name}")
            print(f"{'='*60}")

            # Create backtester
            backtester = Backtester(
                strategy=strategy,
                data_file='market_data_aapl.csv',
                initial_capital=100000.0,
                position_size=100,
                order_log_file=f'backtest_{strategy.name}_orders.csv'
            )

            # Run backtest
            results = backtester.run()

            # Display metrics
            print(f"\nPerformance Metrics:")
            for key, value in backtester.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            # Generate report
            backtester.generate_report(f'backtest_{strategy.name}_report.txt')

            # Save results
            backtester.save_results(f'backtest_{strategy.name}_results.json')

            # Plot results
            backtester.plot_results(f'backtest_{strategy.name}_plot.png')

        print("\n" + "=" * 60)
        print("All backtests complete!")
        print("=" * 60)

    except FileNotFoundError:
        print("Market data not found. Run part1_data_download.py first.")


if __name__ == "__main__":
    main()
