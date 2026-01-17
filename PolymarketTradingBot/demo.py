#!/usr/bin/env python3
"""
Quick Demo Script for Polymarket Trading Bot

This script demonstrates the bot's capabilities without requiring full setup.
Run with: python demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from market_data import PolymarketDataFetcher
from strategy import StrategyManager, MeanReversionStrategy, ArbitrageStrategy
from risk_manager import RiskManager
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def main():
    """Run the demo"""
    print_header("POLYMARKET TRADING BOT DEMO")

    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("Initializing bot components...")
    fetcher = PolymarketDataFetcher(rate_limit_delay=0.6)
    strategy_manager = StrategyManager(config)
    risk_manager = RiskManager(config.get('risk_management', {}))
    print("✓ Bot components initialized\n")

    # Step 1: Fetch markets
    print_header("STEP 1: Fetching Polymarket Markets")
    print("Fetching active markets from Polymarket API...")

    markets = fetcher.get_all_markets(limit=10, active=True)
    print(f"✓ Successfully fetched {len(markets)} active markets\n")

    if markets:
        print("Sample markets:")
        for i, market in enumerate(markets[:5], 1):
            question = market.get('question', 'N/A')
            if len(question) > 70:
                question = question[:67] + "..."
            print(f"  {i}. {question}")
    else:
        print("⚠ No markets fetched. Please check your internet connection.")
        return

    # Step 2: Analyze market sentiment
    print_header("STEP 2: Analyzing Market Sentiment")

    sample_market = markets[0]
    market_id = sample_market.get('id')
    print(f"Analyzing market: {sample_market.get('question', 'N/A')[:60]}...\n")

    sentiment = fetcher.get_market_sentiment(market_id)

    if sentiment:
        print("Market Outcomes and Probabilities:")
        for outcome in sentiment.get('outcomes', []):
            outcome_name = outcome['outcome']
            prob = outcome['implied_probability']
            price = outcome['price']
            print(f"  • {outcome_name}: {prob:.2%} (Price: ${price:.4f})")
    else:
        print("⚠ Could not fetch sentiment data")
        return

    # Step 3: Generate trading signals
    print_header("STEP 3: Generating Trading Signals")

    all_signals = strategy_manager.generate_all_signals(sentiment)

    if all_signals:
        print(f"Generated signals from {len(all_signals)} strategies:\n")
        for strategy_signal in all_signals:
            strategy_name = strategy_signal['strategy']
            signals = strategy_signal['signals']

            if signals:
                print(f"Strategy: {strategy_name.upper()}")
                for signal in signals:
                    print(f"  • {signal['signal']}: {signal['outcome']}")
                    print(f"    Confidence: {signal['confidence']:.1%} | Edge: {signal['edge']:.1%}")
                    print(f"    Reasoning: {signal['reasoning']}")
                print()
    else:
        print("No trading signals generated for this market.")

    # Step 4: Risk management and position sizing
    print_header("STEP 4: Risk Management & Position Sizing")

    best_signal = strategy_manager.get_best_signal(all_signals) if all_signals else None

    if best_signal:
        print("Best Trading Opportunity:")
        print(f"  Signal: {best_signal['signal']}")
        print(f"  Outcome: {best_signal['outcome']}")
        print(f"  Strategy: {best_signal['strategy']}")
        print(f"  Confidence: {best_signal['confidence']:.1%}")
        print(f"  Edge: {best_signal['edge']:.1%}")
        print(f"\n  {best_signal['reasoning']}")

        # Calculate position size
        position_size = risk_manager.calculate_position_size(best_signal)
        print(f"\nRecommended Position Size: ${position_size:,.2f}")
        print(f"  (Based on {config['risk_management']['total_capital']:,} total capital)")
    else:
        print("No high-confidence signals to trade at this time.")

    # Step 5: Portfolio summary
    print_header("STEP 5: Portfolio Status")

    summary = risk_manager.get_portfolio_summary()
    print("Current Portfolio:")
    print(f"  Total Capital:      ${summary['total_capital']:,.2f}")
    print(f"  Available Capital:  ${summary['available_capital']:,.2f}")
    print(f"  Active Positions:   {summary['active_positions']}")
    print(f"  Total Exposure:     ${summary['total_exposure']:,.2f}")
    print(f"  Total PnL:          ${summary['total_pnl']:,.2f}")
    print(f"  Return:             {summary['return_pct']:.2f}%")

    # Step 6: Find more opportunities
    print_header("STEP 6: Scanning for Trading Opportunities")

    print("Analyzing markets for arbitrage and mispricing opportunities...\n")
    opportunities = fetcher.find_trading_opportunities(markets, threshold=0.05)

    if opportunities:
        print(f"Found {len(opportunities)} potential opportunities:\n")
        for i, opp in enumerate(opportunities[:3], 1):
            question = opp['question']
            if len(question) > 60:
                question = question[:57] + "..."
            print(f"{i}. {question}")
            print(f"   Type: {opp.get('opportunity_type', 'N/A')}")

            if 'deviation' in opp:
                print(f"   Probability sum: {opp['probability_sum']:.1%} "
                      f"(Deviation: {opp['deviation']:.1%})")
            elif 'probability' in opp:
                print(f"   {opp.get('outcome', 'N/A')}: {opp['probability']:.1%}")
            print()
    else:
        print("No significant opportunities found at this time.")

    # Final summary
    print_header("DEMO COMPLETE")
    print("The trading bot successfully demonstrated:")
    print("  ✓ Market data fetching from Polymarket API")
    print("  ✓ Probability and sentiment analysis")
    print("  ✓ Multi-strategy signal generation")
    print("  ✓ Risk management and position sizing")
    print("  ✓ Opportunity scanning and analysis")
    print("\nTo run the bot continuously, use:")
    print("  python src/bot.py")
    print("\nOr explore interactively with:")
    print("  jupyter notebook notebooks/trading_bot_demo.ipynb")
    print("\nNote: Currently in DRY RUN mode - no real trades are executed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print(f"\n⚠ Error: {e}")
        print("Please ensure you have internet connectivity to access Polymarket API")
