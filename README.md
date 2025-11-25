# End-to-End Trading System

**FINM_32500 - Financial Mathematics**
**University of Chicago**

A complete algorithmic trading system implementation featuring data acquisition, strategy development, backtesting framework, and live trading integration with Alpaca.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Part 1: Data Download and Preparation](#part-1-data-download-and-preparation)
6. [Part 2: Backtester Framework](#part-2-backtester-framework)
7. [Part 3: Strategy Backtesting](#part-3-strategy-backtesting)
8. [Part 4: Alpaca Trading Integration](#part-4-alpaca-trading-integration)
9. [File Structure](#file-structure)
10. [Examples](#examples)
11. [Performance Metrics](#performance-metrics)
12. [Safety and Best Practices](#safety-and-best-practices)

---

## Project Overview

This project implements a complete end-to-end algorithmic trading system with the following capabilities:

- **Market Data Acquisition**: Download and prepare intraday data from Yahoo Finance and Binance
- **Strategy Development**: Multiple trading strategies (Momentum, MA Crossover, RSI, Bollinger Bands, Combined)
- **Order Book Management**: Efficient order book using heaps with price-time priority
- **Risk Management**: Capital sufficiency checks, position limits, and rate limiting
- **Backtesting Engine**: Complete simulation with realistic order execution
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, drawdown, win rate
- **Live Trading**: Integration with Alpaca Paper Trading API

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRADING SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │   Part 1     │     │   Part 2     │     │   Part 3     ││
│  │     Data     │────▶│  Backtester  │────▶│ Performance  ││
│  │ Acquisition  │     │  Framework   │     │  Analytics   ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│         │                     │                     │        │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Trading Strategies                      │  │
│  │  • Momentum  • MA Crossover  • RSI  • BB  • Combined│  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────┐                         │
│                    │   Part 4     │                         │
│                    │    Alpaca    │                         │
│                    │   Live API   │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **MarketDataGateway**: Streams historical data to simulate live feed
2. **OrderBook**: Manages bid/ask orders with heap-based priority queues
3. **OrderManager**: Validates orders against capital and risk limits
4. **MatchingEngine**: Simulates realistic order execution outcomes
5. **Strategy Classes**: Pluggable trading strategy implementations
6. **Backtester**: Integrates all components for performance evaluation

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd end_to_end_trading_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import yfinance, pandas, numpy, matplotlib; print('All packages installed!')"
   ```

---

## Quick Start Guide

### Step 1: Download Market Data

```bash
python part1_data_download.py
```

This will:
- Download 7 days of 1-minute AAPL data
- Download 1000 candles of BTC/USDT data
- Clean and add technical indicators
- Save to `market_data_aapl.csv` and `market_data_btc.csv`

### Step 2: Test Strategies

```bash
python part1_strategy.py
```

This will:
- Load the downloaded data
- Test all 5 trading strategies
- Display signal counts and sample signals

### Step 3: Run Backtests

```bash
python part3_backtester.py
```

This will:
- Backtest all strategies on historical data
- Generate performance reports
- Create equity curve plots
- Save results to JSON files

### Step 4: (Optional) Alpaca Integration

```bash
# Set up API keys first
export ALPACA_API_KEY='your_paper_trading_key'
export ALPACA_SECRET_KEY='your_paper_trading_secret'

python part4_alpaca_integration.py
```

---

## Part 1: Data Download and Preparation

### Overview

Part 1 handles data acquisition and feature engineering.

### Files

- `part1_data_download.py`: Data download and cleaning
- `part1_strategy.py`: Trading strategy implementations

### DataDownloader Class

```python
from part1_data_download import DataDownloader

# Download equity data
downloader = DataDownloader('AAPL', data_source='yfinance')
data = downloader.download_equity_data(period='7d', interval='1m')
clean_data = downloader.clean_data(add_features=True)
downloader.save_data('my_data.csv')

# Download crypto data
crypto_downloader = DataDownloader('BTC', data_source='binance')
crypto_data = crypto_downloader.download_crypto_data(symbol='BTCUSDT', limit=1000)
crypto_downloader.clean_data(add_features=True)
```

### Features Added

The system automatically calculates:

- **Returns**: Simple and log returns
- **Moving Averages**: SMA (5, 10, 20, 50) and EMA (5, 10, 20)
- **Volatility**: Rolling standard deviation (10, 20 periods)
- **Momentum**: Price momentum (5, 10 periods)
- **RSI**: Relative Strength Index (14 periods)
- **VWAP**: Volume Weighted Average Price
- **Bollinger Bands**: 20-period with 2 standard deviations

### Trading Strategies

#### 1. Momentum Strategy

Trades based on price momentum over a lookback period.

```python
from part1_strategy import MomentumStrategy

strategy = MomentumStrategy(lookback_period=10, threshold=0.0)
signals = strategy.generate_signals(data)
```

#### 2. Moving Average Crossover

Generates signals when short MA crosses long MA.

```python
from part1_strategy import MovingAverageCrossoverStrategy

strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20, ma_type='SMA')
signals = strategy.generate_signals(data)
```

#### 3. RSI Strategy

Mean reversion based on RSI oversold/overbought levels.

```python
from part1_strategy import RSIStrategy

strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
signals = strategy.generate_signals(data)
```

#### 4. Bollinger Bands Strategy

Trades bounces off Bollinger Bands.

```python
from part1_strategy import BollingerBandsStrategy

strategy = BollingerBandsStrategy(period=20, num_std=2.0)
signals = strategy.generate_signals(data)
```

#### 5. Combined Strategy

Combines MA crossover with RSI confirmation.

```python
from part1_strategy import CombinedStrategy

strategy = CombinedStrategy(ma_short=5, ma_long=20, rsi_period=14)
signals = strategy.generate_signals(data)
```

---

## Part 2: Backtester Framework

### Overview

Part 2 implements the core backtesting infrastructure.

### Files

- `part2_gateway.py`: Market data and order gateways
- `part2_orderbook.py`: Order book with heap-based matching
- `part2_order_manager.py`: Order validation and risk management
- `part2_matching_engine.py`: Order execution simulator

### Example Usage

See individual Python files for detailed examples and test code.

---

## Part 3: Strategy Backtesting

### Overview

Part 3 integrates all components to run complete backtests with performance analysis.

### File

- `part3_backtester.py`: Complete backtesting framework

### Running a Backtest

```python
from part3_backtester import Backtester
from part1_strategy import MovingAverageCrossoverStrategy

# Create strategy
strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)

# Create backtester
backtester = Backtester(
    strategy=strategy,
    data_file='market_data_aapl.csv',
    initial_capital=100000.0,
    position_size=100,
    order_log_file='backtest_orders.csv'
)

# Run backtest
results = backtester.run()

# Generate reports
backtester.generate_report('backtest_report.txt')
backtester.save_results('backtest_results.json')
backtester.plot_results('backtest_plot.png')
```

---

## Part 4: Alpaca Trading Integration

### Overview

Part 4 integrates with Alpaca's Paper Trading API for live trading.

### File

- `part4_alpaca_integration.py`: Alpaca API integration

### Setup

1. **Create Alpaca Account**: Sign up at [alpaca.markets](https://alpaca.markets)
2. **Get API Keys**: Navigate to Paper Trading → API Keys
3. **Configure Environment**:
   ```bash
   export ALPACA_API_KEY='your_paper_key_id'
   export ALPACA_SECRET_KEY='your_paper_secret_key'
   ```

**WARNING**: Only use paper trading! Never add real money to your account for this project.

---

## File Structure

```
end_to_end_trading_system/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── part1_data_download.py            # Data acquisition and cleaning
├── part1_strategy.py                 # Trading strategies
│
├── part2_gateway.py                  # Market data and order gateways
├── part2_orderbook.py                # Order book implementation
├── part2_order_manager.py            # Order validation and risk management
├── part2_matching_engine.py          # Order execution simulator
│
├── part3_backtester.py               # Complete backtesting framework
│
└── part4_alpaca_integration.py       # Alpaca API integration
```

---

## Examples

Each Python file contains a `main()` function with working examples. Run any file directly to see it in action:

```bash
python part1_data_download.py
python part1_strategy.py
python part2_gateway.py
python part2_orderbook.py
python part2_order_manager.py
python part2_matching_engine.py
python part3_backtester.py
python part4_alpaca_integration.py
```

---

## Performance Metrics

The backtester calculates:

- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Fill Rate**: Percentage of orders filled
- **Average Slippage**: Average execution slippage

---

## Safety and Best Practices

### Paper Trading Only

- **NEVER add real money** to your Alpaca account for this project
- Always use paper trading keys (`https://paper-api.alpaca.markets`)

### API Key Security

- **NEVER commit API keys** to version control
- Use environment variables
- Rotate keys if accidentally exposed

### Risk Management

The system includes built-in risk controls:
- Capital sufficiency checks
- Position size limits (default: 30% of capital)
- Order rate limiting (default: 100 orders/minute)

---

## Additional Resources

- **Alpaca Documentation**: [https://alpaca.markets/docs/](https://alpaca.markets/docs/)
- **yfinance Documentation**: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
- **Binance API**: [https://binance-docs.github.io/apidocs/](https://binance-docs.github.io/apidocs/)

---

## License

This project is for educational purposes as part of FINM_32500 at the University of Chicago.

**Remember**: This is a learning project. Always use paper trading and never risk real capital!
