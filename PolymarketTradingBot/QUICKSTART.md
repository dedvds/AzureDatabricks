# Polymarket Trading Bot - Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
cd PolymarketTradingBot
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
# With mock data (works offline)
python demo_with_mock_data.py

# With live Polymarket data (requires internet)
python demo.py
```

### 3. Customize and Run
Edit `config/config.yaml` to adjust strategies and risk parameters, then:
```bash
# Run a single trading cycle
python src/bot.py --once

# Run continuously
python src/bot.py
```

## 📊 What You Just Saw

The demo showed the bot:
1. ✅ **Analyzing market probabilities** from Polymarket
2. ✅ **Detecting arbitrage opportunities** (probabilities not summing to 100%)
3. ✅ **Identifying mean reversion signals** (extreme probabilities)
4. ✅ **Calculating optimal position sizes** based on confidence
5. ✅ **Managing risk** with stop loss and take profit
6. ✅ **Tracking portfolio performance** in real-time

## 🎯 Key Features

### Three Trading Strategies

1. **Arbitrage Strategy** ⚡
   - Finds binary markets where probabilities don't sum to ~100%
   - Example: If Yes=48% and No=46% (sum=94%), buy both outcomes

2. **Mean Reversion Strategy** 📈
   - Trades against extreme probabilities
   - Example: If probability drops to 3%, expect some reversion

3. **Value Strategy** 💎
   - Compares market prices to fundamental estimates
   - Requires you to provide probability estimates

### Risk Management 🛡️

- **Position Sizing**: Automatically calculated based on signal confidence
- **Portfolio Limits**: Maximum 20% exposure by default
- **Stop Loss**: Auto-close at -15% loss
- **Take Profit**: Auto-close at +30% profit
- **Position Limits**: Maximum 10 concurrent positions

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
risk_management:
  total_capital: 10000        # Your trading capital
  max_position_size_pct: 0.1  # Max 10% per trade
  stop_loss_pct: 0.15         # -15% stop loss
  take_profit_pct: 0.30       # +30% take profit

strategies:
  mean_reversion:
    enabled: true
    extreme_low: 0.05   # Buy when prob < 5%
    extreme_high: 0.95  # Sell when prob > 95%

  arbitrage:
    enabled: true
    min_prob_sum: 0.95  # Flag when sum < 95%
```

## 📱 Ways to Use the Bot

### 1. Command Line (Recommended for Production)
```bash
python src/bot.py
```

### 2. Jupyter Notebook (Recommended for Learning)
```bash
jupyter notebook notebooks/trading_bot_demo.ipynb
```

### 3. Python Script (Recommended for Integration)
```python
from bot import PolymarketTradingBot

bot = PolymarketTradingBot()
bot.run_single_cycle()  # Run once
# or
bot.run()  # Run continuously
```

## 🔒 Safety Features

- **DRY RUN MODE** is enabled by default
- No real trades are executed until you set `dry_run: false` in config
- All trades are logged for review
- Stop loss and take profit protect your capital

## 📈 Live Trading (When Ready)

To enable live trading:

1. Get Polymarket API credentials (wallet, private key)
2. Add credentials to `.env` file
3. Set `dry_run: false` in `config/config.yaml`
4. Start small with limited capital
5. Monitor performance regularly

⚠️ **Warning**: Live trading involves real financial risk!

## 🎓 Learn More

- **Full Documentation**: See README.md
- **Strategy Details**: Review src/strategy.py
- **Risk Management**: Check src/risk_manager.py
- **Interactive Examples**: Open notebooks/trading_bot_demo.ipynb

## 🤝 Support

Questions? Check:
- README.md for detailed documentation
- config/config.yaml for all configuration options
- demo scripts for working examples

---

**Happy Trading! Remember: Only risk what you can afford to lose.**
