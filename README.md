# ExtendBot - Extended Exchange Trading Bot

Automated trading bot for [Extended Exchange](https://extended.exchange) built using the official [x10xchange Python SDK](https://github.com/x10xchange/python_sdk).

## Features

- **Real-time Market Data**: WebSocket subscriptions for orderbooks, trades, and candles
- **Order Management**: Market/limit orders with TP/SL using Stark signatures
- **Multiple Strategies**: Swing trading with adaptive risk management
- **Risk Controls**: Kill switch, drawdown monitor, Kelly criterion position sizing
- **Multi-Asset Trading**: Trade multiple markets simultaneously
- **Paper Trading**: Test strategies without real capital
- **Telegram Notifications**: Real-time alerts for trades and signals
- **Trade Analytics**: PostgreSQL database logging and analytics

## Prerequisites

- Python 3.10+
- Extended Exchange account with API credentials
- (Optional) PostgreSQL database
- (Optional) Telegram bot token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/extendbot.git
cd extendbot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Required Credentials

Get these from your Extended Exchange account:

| Variable | Description |
|----------|-------------|
| `API_KEY` | API key for REST calls |
| `PRIVATE_KEY` | Stark private key for signing orders |
| `PUBLIC_KEY` | Stark public key |
| `VAULT_ID` | Your account vault ID |

### Trading Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `SOL-USD` | Primary trading pair |
| `TIMEFRAME` | `1m` | Candle timeframe (1m, 5m, 15m, 1h, 4h) |
| `MAX_LEVERAGE` | `5` | Maximum leverage (1-20x) |
| `MAX_POSITIONS` | `3` | Max concurrent positions |
| `BASE_POSITION_SIZE_PCT` | `25` | Position size as % of account |

### Risk Management

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DAILY_LOSS_PCT` | `5.0` | Daily loss limit (kill switch) |
| `MAX_DRAWDOWN_PCT` | `10.0` | Max drawdown limit |
| `DEFAULT_TP_PCT` | `3.0` | Default take profit % |
| `DEFAULT_SL_PCT` | `1.5` | Default stop loss % |

## Usage

### Start the Bot

```bash
python -m app.bot
```

### Paper Trading Mode

Set `PAPER_TRADING=true` in `.env` to run without real trades:

```bash
PAPER_TRADING=true python -m app.bot
```

### Multi-Asset Mode

Enable trading multiple assets:

```env
MULTI_ASSET_MODE=true
MULTI_ASSETS=BTC-USD,ETH-USD,SOL-USD
MAX_POSITIONS=3
```

## Project Structure

```
extendbot/
├── app/
│   ├── __init__.py
│   ├── bot.py              # Main bot controller
│   ├── config.py           # Configuration validation
│   ├── ex/                 # Extended Exchange integration
│   │   ├── ex_client.py    # REST API client
│   │   ├── ex_websocket.py # WebSocket streams
│   │   └── ex_order_manager.py  # Order execution
│   ├── strategies/         # Trading strategies
│   │   ├── strategy_manager.py
│   │   ├── rule_based/     # Rule-based strategies
│   │   └── adaptive/       # Adaptive indicators
│   ├── risk/              # Risk management
│   │   ├── risk_engine.py
│   │   ├── kill_switch.py
│   │   ├── drawdown_monitor.py
│   │   └── kelly_criterion.py
│   ├── portfolio/         # Position management
│   ├── utils/             # Utilities
│   ├── tg_bot/            # Telegram integration
│   └── database/          # Database management
├── data/
│   └── trades/            # Trade logs (JSONL)
├── logs/                  # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

## API Reference

### Extended Exchange SDK

This bot uses the official x10xchange Python SDK:
- **GitHub**: https://github.com/x10xchange/python_sdk
- **API Docs**: https://api.docs.extended.exchange

### Key SDK Components

- `PerpetualTradingClient`: REST API client for trading operations
- `StarkPerpetualAccount`: Account with Stark key for signing
- `PerpetualStreamClient`: WebSocket client for real-time data

## Supported Markets

Extended Exchange supports perpetual futures on:
- BTC-USD
- ETH-USD
- SOL-USD
- (More markets available on exchange)

## Risk Disclaimer

**⚠️ TRADING CRYPTOCURRENCY INVOLVES SUBSTANTIAL RISK OF LOSS**

- This bot is provided for educational purposes only
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- Always start with paper trading mode
- The authors are not responsible for any financial losses

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md before submitting PRs.

## Support

- Issues: GitHub Issues
- Extended Exchange: https://extended.exchange
- SDK Docs: https://api.docs.extended.exchange
