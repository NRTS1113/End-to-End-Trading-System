# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the System (In Order)

### Step 1: Download Data
```bash
python part1_data_download.py
```
**Output**: `market_data_aapl.csv`, `market_data_btc.csv`

### Step 2: Test Strategies
```bash
python part1_strategy.py
```
**Output**: Signal counts and strategy performance preview

### Step 3: Run Backtests
```bash
python part3_backtester.py
```
**Output**: Multiple files per strategy:
- `backtest_*_orders.csv` - Order logs
- `backtest_*_report.txt` - Performance report
- `backtest_*_results.json` - Results data
- `backtest_*_plot.png` - Equity curves

### Step 4 (Optional): Alpaca Integration
```bash
export ALPACA_API_KEY='your_paper_key'
export ALPACA_SECRET_KEY='your_paper_secret'
python part4_alpaca_integration.py
```

## Quick Examples

### Use a Specific Strategy

```python
from part3_backtester import Backtester
from part1_strategy import RSIStrategy

strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)

backtester = Backtester(
    strategy=strategy,
    data_file='market_data_aapl.csv',
    initial_capital=100000.0,
    position_size=100
)

results = backtester.run()
backtester.plot_results()
```

### Test Order Book

```python
from part2_orderbook import OrderBook

book = OrderBook('TEST')
book.add_order('BUY1', 'BUY', 100, 150.00)
book.add_order('SELL1', 'SELL', 100, 150.50)

print(f"Best Bid: ${book.get_best_bid():.2f}")
print(f"Best Ask: ${book.get_best_ask():.2f}")
print(f"Spread: ${book.get_spread():.2f}")

trades = book.match_orders()
print(f"Trades: {trades}")
```

### Create Custom Strategy

```python
from part1_strategy import BaseStrategy, Signal
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['Signal'] = Signal.HOLD.value

        # Your logic here
        df.loc[df['Close'] > df['SMA_20'], 'Signal'] = Signal.BUY.value
        df.loc[df['Close'] < df['SMA_20'], 'Signal'] = Signal.SELL.value

        return df
```

## File Reference

| File | What It Does |
|------|--------------|
| `part1_data_download.py` | Downloads and cleans market data |
| `part1_strategy.py` | 5 trading strategies |
| `part2_gateway.py` | Data streaming & order logging |
| `part2_orderbook.py` | Order matching engine |
| `part2_order_manager.py` | Order validation & risk |
| `part2_matching_engine.py` | Execution simulation |
| `part3_backtester.py` | Complete backtester |
| `part4_alpaca_integration.py` | Alpaca API integration |

## Common Issues

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "No data downloaded"
Try shorter period:
```python
downloader.download_equity_data(period='1d', interval='5m')
```

### "Alpaca API keys not configured"
```bash
export ALPACA_API_KEY='PK...'
export ALPACA_SECRET_KEY='...'
```

### No trades in backtest
Check signals:
```python
signals = strategy.generate_signals(data)
print((signals['Signal'] != 0).sum())
```

## Key Classes

- `DataDownloader`: Download market data
- `MomentumStrategy`, `RSIStrategy`, etc.: Trading strategies
- `MarketDataGateway`: Stream data
- `OrderBook`: Manage orders
- `OrderManager`: Validate orders
- `MatchingEngine`: Execute orders
- `Backtester`: Run backtests
- `AlpacaTradingClient`: Live trading

## Performance Metrics

After running a backtest, check `backtester.metrics`:

- `total_return`: Overall return (%)
- `sharpe_ratio`: Risk-adjusted return
- `max_drawdown`: Worst decline (%)
- `win_rate`: % of winning trades
- `total_trades`: Number of trades
- `fill_rate`: % of orders filled

## Safety Checklist

- [ ] Using paper trading only
- [ ] API keys in environment variables
- [ ] Not committing keys to git
- [ ] Testing with small position sizes
- [ ] Understanding the code before running

## Support

See `README.md` for full documentation.
See `PROJECT_SUMMARY.md` for implementation details.
