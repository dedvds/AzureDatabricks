"""
Polymarket Trading Bot - Main Orchestrator

Coordinates data fetching, strategy execution, and risk management.
"""

import logging
import time
import yaml
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from market_data import PolymarketDataFetcher
from strategy import StrategyManager
from risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolymarketTradingBot:
    """Main trading bot orchestrator"""

    def __init__(self, config_path: str = None):
        """
        Initialize the trading bot

        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        self.config = self._load_config(config_path)
        self.dry_run = self.config.get('execution', {}).get('dry_run', True)

        # Initialize components
        self.data_fetcher = PolymarketDataFetcher(
            rate_limit_delay=self.config.get('execution', {}).get('rate_limit_delay', 0.6)
        )
        self.strategy_manager = StrategyManager(self.config)
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))

        self.scan_interval = self.config.get('execution', {}).get('scan_interval', 300)
        self.running = False

        logger.info(f"Trading bot initialized - Dry run: {self.dry_run}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def scan_markets(self) -> List[Dict]:
        """
        Scan Polymarket for trading opportunities

        Returns:
            List of trading opportunities
        """
        logger.info("Scanning markets for opportunities...")

        # Fetch markets
        market_filters = self.config.get('market_filters', {})
        max_markets = market_filters.get('max_markets_to_scan', 100)

        markets = self.data_fetcher.get_all_markets(
            limit=max_markets,
            active=market_filters.get('active_only', True)
        )

        logger.info(f"Fetched {len(markets)} markets")

        all_opportunities = []

        for market in markets[:20]:  # Limit to first 20 markets for demo
            market_id = market.get('id')
            if not market_id:
                continue

            # Get market sentiment and probabilities
            sentiment_data = self.data_fetcher.get_market_sentiment(market_id)
            if not sentiment_data:
                continue

            # Generate signals from all strategies
            signals = self.strategy_manager.generate_all_signals(sentiment_data)

            if signals:
                all_opportunities.extend(signals)

        logger.info(f"Found {len(all_opportunities)} potential opportunities")
        return all_opportunities

    def execute_trades(self, opportunities: List[Dict]):
        """
        Execute trades based on opportunities

        Args:
            opportunities: List of trading opportunities
        """
        if not opportunities:
            logger.info("No trading opportunities found")
            return

        # Get best signal from all opportunities
        best_signal = self.strategy_manager.get_best_signal(opportunities)

        if not best_signal:
            logger.info("No high-confidence signals to trade")
            return

        logger.info(f"Best signal: {best_signal.get('strategy')} - {best_signal.get('outcome')}")
        logger.info(f"Confidence: {best_signal.get('confidence', 0):.2%} - Edge: {best_signal.get('edge', 0):.2%}")
        logger.info(f"Reasoning: {best_signal.get('reasoning')}")

        # Check if we can open a position
        if not self.risk_manager.can_open_position():
            logger.warning("Cannot open new position - risk limits reached")
            return

        # Open position
        if self.dry_run:
            position_size = self.risk_manager.calculate_position_size(best_signal)
            logger.info(f"[DRY RUN] Would open position: ${position_size:.2f}")
            logger.info(f"[DRY RUN] Signal details: {best_signal}")
        else:
            position = self.risk_manager.open_position(best_signal)
            if position:
                logger.info(f"Opened position: {position.to_dict()}")
            else:
                logger.warning("Failed to open position")

    def update_positions(self):
        """Update all open positions with current market prices"""
        if not self.risk_manager.positions:
            return

        logger.info(f"Updating {len(self.risk_manager.positions)} positions...")

        # Collect current prices for all positions
        market_prices = {}
        for position in self.risk_manager.positions:
            price_data = self.data_fetcher.get_market_prices(position.token_id)
            if price_data:
                market_prices[position.token_id] = float(price_data.get('mid', position.entry_price))

        # Update positions and check stop loss/take profit
        self.risk_manager.update_positions(market_prices)

    def print_portfolio_summary(self):
        """Print current portfolio summary"""
        summary = self.risk_manager.get_portfolio_summary()

        logger.info("=" * 60)
        logger.info("PORTFOLIO SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Capital:      ${summary['total_capital']:,.2f}")
        logger.info(f"Current Value:      ${summary['current_value']:,.2f}")
        logger.info(f"Total Exposure:     ${summary['total_exposure']:,.2f}")
        logger.info(f"Available Capital:  ${summary['available_capital']:,.2f}")
        logger.info(f"Active Positions:   {summary['active_positions']}")
        logger.info(f"Realized PnL:       ${summary['realized_pnl']:,.2f}")
        logger.info(f"Unrealized PnL:     ${summary['unrealized_pnl']:,.2f}")
        logger.info(f"Total PnL:          ${summary['total_pnl']:,.2f}")
        logger.info(f"Return:             {summary['return_pct']:.2f}%")
        logger.info("=" * 60)

        # Print performance metrics if we have closed trades
        metrics = self.risk_manager.get_performance_metrics()
        if metrics['total_trades'] > 0:
            logger.info("PERFORMANCE METRICS")
            logger.info("=" * 60)
            logger.info(f"Total Trades:       {metrics['total_trades']}")
            logger.info(f"Win Rate:           {metrics['win_rate']:.2%}")
            logger.info(f"Average Win:        ${metrics['avg_win']:,.2f}")
            logger.info(f"Average Loss:       ${metrics['avg_loss']:,.2f}")
            logger.info(f"Profit Factor:      {metrics['profit_factor']:.2f}")
            logger.info("=" * 60)

    def run_single_cycle(self):
        """Run a single trading cycle"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting trading cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")

        # Scan for opportunities
        opportunities = self.scan_markets()

        # Update existing positions
        self.update_positions()

        # Execute new trades
        self.execute_trades(opportunities)

        # Print portfolio summary
        self.print_portfolio_summary()

    def run(self):
        """Run the trading bot continuously"""
        self.running = True
        logger.info(f"Starting Polymarket Trading Bot (Dry Run: {self.dry_run})")
        logger.info(f"Scan interval: {self.scan_interval} seconds")

        try:
            while self.running:
                self.run_single_cycle()

                logger.info(f"\nWaiting {self.scan_interval} seconds until next cycle...")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")

        # Print final summary
        logger.info("\nFinal Portfolio Summary:")
        self.print_portfolio_summary()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Polymarket Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    bot = PolymarketTradingBot(config_path=args.config)

    if args.once:
        bot.run_single_cycle()
    else:
        bot.run()


if __name__ == "__main__":
    main()
