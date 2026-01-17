# Polymarket Trading Bot

A sophisticated trading bot that analyzes Polymarket sentiment and underlying probabilities to identify and execute trading opportunities.

## Features

- **Real-time Market Data**: Fetches live market data from Polymarket API
- **Multiple Trading Strategies**:
  - Mean Reversion: Trades against extreme probabilities
  - Arbitrage: Exploits probability sum deviations in binary markets
  - Value: Compares market prices to fundamental probability estimates
- **Risk Management**:
  - Position sizing based on signal confidence
  - Stop loss and take profit automation
  - Portfolio exposure limits
  - Maximum position limits
- **Performance Tracking**: Track PnL, win rate, and other performance metrics
- **Dry Run Mode**: Test strategies without risking real capital

## Architecture

```
PolymarketTradingBot/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── market_data.py        # Polymarket API data fetcher
│   ├── strategy.py           # Trading strategies
│   ├── risk_manager.py       # Position and risk management
│   └── bot.py                # Main bot orchestrator
├── config/
│   └── config.yaml           # Configuration file
├── notebooks/
│   └── trading_bot_demo.ipynb # Interactive demo
├── tests/                    # Unit tests (optional)
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd PolymarketTradingBot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (optional for read-only data access):
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Configure the bot**:
   Edit `config/config.yaml` to customize:
   - Risk management parameters
   - Trading strategies
   - Execution settings

## Configuration

### Risk Management Settings

```yaml
risk_management:
  total_capital: 10000          # Total capital in USD
  max_position_size_pct: 0.1    # Max 10% per position
  max_portfolio_risk_pct: 0.2   # Max 20% total exposure
  max_positions: 10             # Max concurrent positions
  stop_loss_pct: 0.15           # Stop loss at -15%
  take_profit_pct: 0.30         # Take profit at +30%
```

### Trading Strategies

Enable/disable strategies and configure parameters:

```yaml
strategies:
  mean_reversion:
    enabled: true
    extreme_low: 0.05   # Buy when probability < 5%
    extreme_high: 0.95  # Sell when probability > 95%

  arbitrage:
    enabled: true
    min_prob_sum: 0.95  # Flag when probabilities sum < 95%
    max_prob_sum: 1.05  # Flag when probabilities sum > 105%

  value:
    enabled: false      # Requires fundamental probability estimates
```

## Usage

### Command Line

**Run the bot continuously**:
```bash
cd src
python bot.py
```

**Run a single cycle** (for testing):
```bash
python bot.py --once
```

**Use custom configuration**:
```bash
python bot.py --config /path/to/config.yaml
```

### Jupyter Notebook

For interactive exploration and testing:

```bash
jupyter notebook notebooks/trading_bot_demo.ipynb
```

The notebook includes:
- Market data fetching examples
- Strategy signal generation
- Risk management demonstrations
- Portfolio analysis

### Programmatic Usage

```python
from bot import PolymarketTradingBot

# Initialize bot
bot = PolymarketTradingBot()

# Run single cycle
bot.run_single_cycle()

# Or run continuously
bot.run()
```

## How It Works

### 1. Market Data Collection
The bot fetches market data from Polymarket's public API:
- Active markets and their details
- Current prices and probabilities
- Order book data

### 2. Signal Generation
Three strategies analyze the data:

**Mean Reversion**: Identifies markets with extreme probabilities (very high or very low) that may revert to the mean.

**Arbitrage**: Finds binary markets where probabilities don't sum to ~100%, indicating potential arbitrage opportunities.

**Value**: Compares market-implied probabilities to fundamental estimates (requires external data).

### 3. Risk Management
Before executing trades:
- Validates position limits
- Calculates position size based on signal confidence
- Checks available capital
- Ensures portfolio risk limits

### 4. Execution
In dry-run mode (default), the bot logs what it would trade. When live trading is enabled, it executes orders through the Polymarket API.

### 5. Position Monitoring
The bot continuously:
- Updates position values
- Checks stop loss/take profit levels
- Rebalances portfolio
- Tracks performance metrics

## Trading Strategies Explained

### Mean Reversion Strategy

**Thesis**: Markets tend to overreact to news and events, creating opportunities when probabilities become extreme.

**Entry Signals**:
- BUY when probability < 5% (expecting reversion upward)
- SELL when probability > 95% (expecting reversion downward)

**Example**: If a candidate's election probability drops to 2% after bad news, the strategy might buy, expecting some recovery.

### Arbitrage Strategy

**Thesis**: In binary markets, probabilities should sum to ~100%. Deviations indicate mispricing.

**Entry Signals**:
- BUY both outcomes when probabilities sum to < 95%
- SELL both outcomes when probabilities sum to > 105%

**Example**: If outcome A shows 48% and outcome B shows 48% (sum = 96%), buy both for potential arbitrage profit.

### Value Strategy

**Thesis**: Market probabilities can deviate from true underlying probabilities estimated through fundamental analysis.

**Entry Signals**:
- BUY when market probability < fundamental probability
- SELL when market probability > fundamental probability

**Note**: Requires you to provide fundamental probability estimates based on your own analysis or models.

## Risk Management

The bot implements comprehensive risk controls:

1. **Position Sizing**: Based on signal confidence and edge
2. **Portfolio Limits**: Maximum total exposure as % of capital
3. **Stop Loss**: Automatically close losing positions at -15%
4. **Take Profit**: Automatically close winning positions at +30%
5. **Position Limits**: Maximum number of concurrent positions

## Performance Metrics

The bot tracks:
- Total PnL (realized + unrealized)
- Win rate
- Average win/loss
- Profit factor
- Return on capital
- Number of trades

## Dry Run vs Live Trading

**Dry Run Mode** (default):
- Fetches real market data
- Generates real signals
- Simulates position opening/closing
- Safe for testing and learning
- No real money at risk

**Live Trading Mode**:
- Requires Polymarket API credentials
- Executes real trades
- Uses real capital
- Set `dry_run: false` in config

⚠️ **Warning**: Live trading involves real financial risk. Only enable after thorough testing in dry-run mode.

## API Rate Limits

Polymarket's free tier typically allows ~100 requests per minute. The bot respects this with a 0.6-second delay between calls (configurable).

## Dependencies

- `requests`: HTTP library for API calls
- `pyyaml`: Configuration file parsing
- `python-dotenv`: Environment variable management
- `pandas`, `numpy`: Data analysis (optional)

See `requirements.txt` for full list.

## Limitations and Disclaimers

1. **Market Risk**: Trading involves substantial risk of loss
2. **No Guarantees**: Past performance doesn't guarantee future results
3. **Data Accuracy**: Bot relies on Polymarket API data accuracy
4. **Execution Risk**: Slippage and failed orders can occur
5. **Regulatory**: Ensure compliance with local regulations

## Future Enhancements

Potential improvements:
- Machine learning for probability estimation
- Sentiment analysis from news/social media
- Multi-market correlation analysis
- Advanced portfolio optimization
- Webhook notifications
- Web dashboard for monitoring

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Check the documentation
- Review example notebook
- Open an issue on GitHub

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Acknowledgments

Built using:
- [Polymarket API](https://docs.polymarket.com/)
- Market data from Polymarket's public endpoints

---

**Disclaimer**: This bot is for educational purposes. Trading prediction markets involves risk. Always do your own research and never risk more than you can afford to lose.
