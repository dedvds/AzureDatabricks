#!/usr/bin/env python3
"""
Mock Data Demo for Polymarket Trading Bot

Demonstrates the bot's capabilities using sample data.
Run with: python demo_with_mock_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
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


def create_mock_market_data():
    """Create sample market data for demonstration"""
    return [
        {
            'market_id': 'market_001',
            'question': 'Will Bitcoin reach $150,000 by end of 2026?',
            'outcomes': [
                {'outcome': 'Yes', 'token_id': 'token_001_yes', 'price': 0.35, 'implied_probability': 0.35},
                {'outcome': 'No', 'token_id': 'token_001_no', 'price': 0.70, 'implied_probability': 0.70}
            ]
        },
        {
            'market_id': 'market_002',
            'question': 'Will SpaceX successfully land on Mars in 2026?',
            'outcomes': [
                {'outcome': 'Yes', 'token_id': 'token_002_yes', 'price': 0.03, 'implied_probability': 0.03},
                {'outcome': 'No', 'token_id': 'token_002_no', 'price': 0.96, 'implied_probability': 0.96}
            ]
        },
        {
            'market_id': 'market_003',
            'question': 'Will unemployment rate be below 4% in Q4 2026?',
            'outcomes': [
                {'outcome': 'Yes', 'token_id': 'token_003_yes', 'price': 0.48, 'implied_probability': 0.48},
                {'outcome': 'No', 'token_id': 'token_003_no', 'price': 0.46, 'implied_probability': 0.46}
            ]
        },
        {
            'market_id': 'market_004',
            'question': 'Will AI achieve AGI by 2026?',
            'outcomes': [
                {'outcome': 'Yes', 'token_id': 'token_004_yes', 'price': 0.98, 'implied_probability': 0.98},
                {'outcome': 'No', 'token_id': 'token_004_no', 'price': 0.05, 'implied_probability': 0.05}
            ]
        }
    ]


def main():
    """Run the demo with mock data"""
    print_header("POLYMARKET TRADING BOT DEMO (Mock Data)")
    print("Note: Using sample data since Polymarket API is currently unavailable\n")

    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("Initializing bot components...")
    strategy_manager = StrategyManager(config)
    risk_manager = RiskManager(config.get('risk_management', {}))
    print("✓ Bot components initialized\n")

    # Get mock market data
    print_header("STEP 1: Sample Polymarket Markets")

    markets = create_mock_market_data()
    print(f"Analyzing {len(markets)} sample markets:\n")

    for i, market in enumerate(markets, 1):
        question = market['question']
        print(f"{i}. {question}")

    # Analyze each market
    print_header("STEP 2: Analyzing Markets & Generating Signals")

    all_opportunities = []

    for market in markets:
        print(f"\n📊 Market: {market['question']}")
        print("─" * 80)

        # Show probabilities
        print("\nCurrent Probabilities:")
        for outcome in market['outcomes']:
            print(f"  • {outcome['outcome']}: {outcome['implied_probability']:.1%} (${outcome['price']:.4f})")

        # Check probability sum
        prob_sum = sum(o['implied_probability'] for o in market['outcomes'])
        print(f"\nProbability Sum: {prob_sum:.1%}", end="")
        if abs(prob_sum - 1.0) > 0.05:
            print(f" ⚠ Deviation: {abs(prob_sum - 1.0):.1%}")
        else:
            print(" ✓")

        # Generate signals
        signals = strategy_manager.generate_all_signals(market)

        if signals:
            print("\n🎯 Trading Signals Generated:")
            for strategy_signal in signals:
                if strategy_signal.get('signals'):
                    strategy_name = strategy_signal['strategy']
                    print(f"\n  Strategy: {strategy_name.upper()}")

                    for signal in strategy_signal['signals']:
                        print(f"    ➜ {signal['signal']}: {signal['outcome']}")
                        print(f"      Confidence: {signal['confidence']:.1%} | Edge: {signal['edge']:.1%}")
                        print(f"      Reasoning: {signal['reasoning']}")

                        all_opportunities.append({
                            'market': market['question'],
                            'signal': signal,
                            'strategy': strategy_name
                        })
        else:
            print("\n  No high-confidence signals for this market")

    # Step 3: Position Sizing
    print_header("STEP 3: Risk Management & Position Sizing")

    if all_opportunities:
        # Get best signal
        best_signal = strategy_manager.get_best_signal([
            {'market_id': 'mock', 'question': opp['market'],
             'signals': [opp['signal']], 'strategy': opp['strategy']}
            for opp in all_opportunities
        ])

        if best_signal:
            print("🏆 Best Trading Opportunity:")
            print(f"  Market: {[o for o in all_opportunities if o['signal'] == best_signal][0]['market']}")
            print(f"  Signal: {best_signal['signal']}")
            print(f"  Outcome: {best_signal['outcome']}")
            print(f"  Strategy: {best_signal.get('strategy', 'N/A')}")
            print(f"  Confidence: {best_signal['confidence']:.1%}")
            print(f"  Edge: {best_signal['edge']:.1%}")
            print(f"\n  💡 {best_signal['reasoning']}")

            # Calculate position size
            position_size = risk_manager.calculate_position_size(best_signal)
            print(f"\n💰 Recommended Position Size: ${position_size:,.2f}")

            capital = config['risk_management']['total_capital']
            position_pct = (position_size / capital) * 100
            print(f"   ({position_pct:.1f}% of ${capital:,} total capital)")

            # Simulate opening position
            print("\n📝 Simulating position entry...")
            position = risk_manager.open_position(best_signal)
            if position:
                print(f"   ✓ Position opened: {position.side} ${position.size:.2f} @ ${position.entry_price:.4f}")
    else:
        print("No trading opportunities identified in sample markets.")

    # Step 4: Portfolio Summary
    print_header("STEP 4: Portfolio Status")

    summary = risk_manager.get_portfolio_summary()

    print("Current Portfolio:")
    print(f"  💵 Total Capital:      ${summary['total_capital']:>10,.2f}")
    print(f"  💰 Current Value:      ${summary['current_value']:>10,.2f}")
    print(f"  📊 Total Exposure:     ${summary['total_exposure']:>10,.2f}")
    print(f"  ✅ Available Capital:  ${summary['available_capital']:>10,.2f}")
    print(f"  📈 Active Positions:   {summary['active_positions']:>10}")
    print(f"  💵 Realized PnL:       ${summary['realized_pnl']:>10,.2f}")
    print(f"  💹 Unrealized PnL:     ${summary['unrealized_pnl']:>10,.2f}")
    print(f"  🎯 Total PnL:          ${summary['total_pnl']:>10,.2f}")
    print(f"  📊 Return:             {summary['return_pct']:>10.2f}%")

    # Show active positions
    if risk_manager.positions:
        print("\n📋 Active Positions:")
        for pos in risk_manager.positions:
            print(f"  • {pos.outcome}: {pos.side} ${pos.size:.2f} @ ${pos.entry_price:.4f}")
            print(f"    Unrealized PnL: ${pos.unrealized_pnl:.2f}")

    # Step 5: Strategy Analysis
    print_header("STEP 5: Strategy Performance Summary")

    print(f"Total opportunities identified: {len(all_opportunities)}\n")

    # Count by strategy
    strategy_counts = {}
    for opp in all_opportunities:
        strategy = opp['strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print("Signals by Strategy:")
    for strategy, count in strategy_counts.items():
        print(f"  • {strategy.upper()}: {count} signals")

    # Final summary
    print_header("DEMO COMPLETE")
    print("The trading bot successfully demonstrated:")
    print("  ✓ Market probability analysis")
    print("  ✓ Multi-strategy signal generation:")
    print("     • Mean Reversion: Trades against extreme probabilities")
    print("     • Arbitrage: Exploits probability sum deviations")
    print("  ✓ Risk management and position sizing")
    print("  ✓ Portfolio tracking and PnL calculation")
    print("  ✓ Intelligent opportunity ranking")

    print("\n📚 Next Steps:")
    print("  1. To run with live data: python demo.py")
    print("  2. To run bot continuously: python src/bot.py")
    print("  3. To explore interactively: jupyter notebook notebooks/trading_bot_demo.ipynb")
    print("  4. Edit config/config.yaml to customize strategies and risk parameters")

    print("\n⚠️  Note: Currently in DRY RUN mode - no real trades are executed.")
    print("   To enable live trading, set 'dry_run: false' in config.yaml")
    print("   (Requires Polymarket API credentials)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print(f"\n⚠ Error: {e}")
