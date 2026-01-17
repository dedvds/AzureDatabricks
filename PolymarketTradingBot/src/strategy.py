"""
Trading Strategy Module

Implements various trading strategies based on Polymarket probabilities and sentiment.
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(Enum):
    """Available trading strategies"""
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    VALUE = "value"


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, config: Dict):
        """
        Initialize trading strategy

        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.min_edge = config.get('min_edge', 0.05)  # Minimum probability edge (5%)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

    def generate_signal(self, market_data: Dict) -> Dict:
        """
        Generate trading signal based on market data

        Args:
            market_data: Market data dictionary

        Returns:
            Signal dictionary with action, confidence, and reasoning
        """
        raise NotImplementedError("Subclasses must implement generate_signal")


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy - trades against extreme probabilities
    Assumes markets overreact and will revert to mean
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.extreme_low = config.get('extreme_low', 0.05)
        self.extreme_high = config.get('extreme_high', 0.95)

    def generate_signal(self, market_data: Dict) -> Dict:
        """
        Generate mean reversion signal

        Buys low probability outcomes, sells high probability outcomes
        """
        outcomes = market_data.get('outcomes', [])

        signals = []
        for outcome in outcomes:
            prob = outcome.get('implied_probability', 0.5)
            price = outcome.get('price', 0)

            if prob < self.extreme_low:
                # Probability is extremely low - buy signal
                edge = self.extreme_low - prob
                confidence = min(1.0, edge / self.min_edge)

                if confidence >= self.confidence_threshold:
                    signals.append({
                        'outcome': outcome.get('outcome'),
                        'token_id': outcome.get('token_id'),
                        'signal': SignalType.BUY.value,
                        'current_probability': prob,
                        'target_probability': self.extreme_low * 2,
                        'edge': edge,
                        'confidence': confidence,
                        'price': price,
                        'reasoning': f"Extreme low probability {prob:.2%} - likely to revert"
                    })

            elif prob > self.extreme_high:
                # Probability is extremely high - sell signal
                edge = prob - self.extreme_high
                confidence = min(1.0, edge / self.min_edge)

                if confidence >= self.confidence_threshold:
                    signals.append({
                        'outcome': outcome.get('outcome'),
                        'token_id': outcome.get('token_id'),
                        'signal': SignalType.SELL.value,
                        'current_probability': prob,
                        'target_probability': self.extreme_high * 0.9,
                        'edge': edge,
                        'confidence': confidence,
                        'price': price,
                        'reasoning': f"Extreme high probability {prob:.2%} - likely to revert"
                    })

        return {
            'market_id': market_data.get('market_id'),
            'question': market_data.get('question'),
            'strategy': Strategy.MEAN_REVERSION.value,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }


class ArbitrageStrategy(TradingStrategy):
    """
    Arbitrage strategy - exploits probability sum deviations
    In binary markets, probabilities should sum to ~1.0
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_prob_sum = config.get('max_prob_sum', 1.05)
        self.min_prob_sum = config.get('min_prob_sum', 0.95)

    def generate_signal(self, market_data: Dict) -> Dict:
        """
        Generate arbitrage signal based on probability sum deviations
        """
        outcomes = market_data.get('outcomes', [])

        if len(outcomes) != 2:
            # Arbitrage strategy only works for binary markets
            return {
                'market_id': market_data.get('market_id'),
                'question': market_data.get('question'),
                'strategy': Strategy.ARBITRAGE.value,
                'signals': [],
                'timestamp': datetime.now().isoformat()
            }

        prob_sum = sum(o.get('implied_probability', 0) for o in outcomes)
        signals = []

        if prob_sum < self.min_prob_sum:
            # Probabilities sum to less than 1 - arbitrage opportunity
            # Buy both outcomes
            for outcome in outcomes:
                signals.append({
                    'outcome': outcome.get('outcome'),
                    'token_id': outcome.get('token_id'),
                    'signal': SignalType.BUY.value,
                    'current_probability': outcome.get('implied_probability'),
                    'probability_sum': prob_sum,
                    'edge': 1.0 - prob_sum,
                    'confidence': min(1.0, (1.0 - prob_sum) / self.min_edge),
                    'price': outcome.get('price'),
                    'reasoning': f"Arbitrage: probabilities sum to {prob_sum:.2%} < 100%"
                })

        elif prob_sum > self.max_prob_sum:
            # Probabilities sum to more than 1 - sell both
            for outcome in outcomes:
                signals.append({
                    'outcome': outcome.get('outcome'),
                    'token_id': outcome.get('token_id'),
                    'signal': SignalType.SELL.value,
                    'current_probability': outcome.get('implied_probability'),
                    'probability_sum': prob_sum,
                    'edge': prob_sum - 1.0,
                    'confidence': min(1.0, (prob_sum - 1.0) / self.min_edge),
                    'price': outcome.get('price'),
                    'reasoning': f"Arbitrage: probabilities sum to {prob_sum:.2%} > 100%"
                })

        return {
            'market_id': market_data.get('market_id'),
            'question': market_data.get('question'),
            'strategy': Strategy.ARBITRAGE.value,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }


class ValueStrategy(TradingStrategy):
    """
    Value strategy - compares market probabilities to fundamental analysis
    Requires external probability estimates (e.g., from models, polls, etc.)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.fundamental_probabilities = config.get('fundamental_probabilities', {})

    def set_fundamental_probability(self, market_id: str, outcome: str, probability: float):
        """
        Set fundamental probability estimate for a market outcome

        Args:
            market_id: Market identifier
            outcome: Outcome name
            probability: Estimated true probability (0-1)
        """
        if market_id not in self.fundamental_probabilities:
            self.fundamental_probabilities[market_id] = {}
        self.fundamental_probabilities[market_id][outcome] = probability

    def generate_signal(self, market_data: Dict) -> Dict:
        """
        Generate value signal by comparing market price to fundamental value
        """
        market_id = market_data.get('market_id')
        outcomes = market_data.get('outcomes', [])

        signals = []

        if market_id not in self.fundamental_probabilities:
            logger.warning(f"No fundamental probabilities for market {market_id}")
            return {
                'market_id': market_id,
                'question': market_data.get('question'),
                'strategy': Strategy.VALUE.value,
                'signals': [],
                'timestamp': datetime.now().isoformat()
            }

        for outcome in outcomes:
            outcome_name = outcome.get('outcome')
            market_prob = outcome.get('implied_probability', 0.5)
            fundamental_prob = self.fundamental_probabilities[market_id].get(outcome_name)

            if fundamental_prob is None:
                continue

            edge = fundamental_prob - market_prob

            if edge > self.min_edge:
                # Market underprices outcome - buy signal
                signals.append({
                    'outcome': outcome_name,
                    'token_id': outcome.get('token_id'),
                    'signal': SignalType.BUY.value,
                    'market_probability': market_prob,
                    'fundamental_probability': fundamental_prob,
                    'edge': edge,
                    'confidence': min(1.0, edge / (self.min_edge * 2)),
                    'price': outcome.get('price'),
                    'reasoning': f"Undervalued: market {market_prob:.2%} vs fundamental {fundamental_prob:.2%}"
                })

            elif edge < -self.min_edge:
                # Market overprices outcome - sell signal
                signals.append({
                    'outcome': outcome_name,
                    'token_id': outcome.get('token_id'),
                    'signal': SignalType.SELL.value,
                    'market_probability': market_prob,
                    'fundamental_probability': fundamental_prob,
                    'edge': abs(edge),
                    'confidence': min(1.0, abs(edge) / (self.min_edge * 2)),
                    'price': outcome.get('price'),
                    'reasoning': f"Overvalued: market {market_prob:.2%} vs fundamental {fundamental_prob:.2%}"
                })

        return {
            'market_id': market_id,
            'question': market_data.get('question'),
            'strategy': Strategy.VALUE.value,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }


class StrategyManager:
    """Manages multiple trading strategies"""

    def __init__(self, config: Dict):
        """
        Initialize strategy manager

        Args:
            config: Configuration for all strategies
        """
        self.strategies = []
        self._initialize_strategies(config)

    def _initialize_strategies(self, config: Dict):
        """Initialize enabled strategies from config"""
        strategy_configs = config.get('strategies', {})

        if strategy_configs.get('mean_reversion', {}).get('enabled', False):
            self.strategies.append(
                MeanReversionStrategy(strategy_configs['mean_reversion'])
            )
            logger.info("Mean reversion strategy enabled")

        if strategy_configs.get('arbitrage', {}).get('enabled', False):
            self.strategies.append(
                ArbitrageStrategy(strategy_configs['arbitrage'])
            )
            logger.info("Arbitrage strategy enabled")

        if strategy_configs.get('value', {}).get('enabled', False):
            self.strategies.append(
                ValueStrategy(strategy_configs['value'])
            )
            logger.info("Value strategy enabled")

    def generate_all_signals(self, market_data: Dict) -> List[Dict]:
        """
        Generate signals from all enabled strategies

        Args:
            market_data: Market data dictionary

        Returns:
            List of signal dictionaries from all strategies
        """
        all_signals = []

        for strategy in self.strategies:
            try:
                signals = strategy.generate_signal(market_data)
                if signals.get('signals'):
                    all_signals.append(signals)
            except Exception as e:
                logger.error(f"Error generating signals from {strategy.__class__.__name__}: {e}")

        return all_signals

    def get_best_signal(self, all_signals: List[Dict]) -> Optional[Dict]:
        """
        Select the best signal from multiple strategies

        Args:
            all_signals: List of signal dictionaries

        Returns:
            Best signal or None
        """
        if not all_signals:
            return None

        # Flatten all signals and sort by confidence
        flattened = []
        for strategy_signals in all_signals:
            for signal in strategy_signals.get('signals', []):
                signal['strategy'] = strategy_signals['strategy']
                signal['market_id'] = strategy_signals['market_id']
                signal['question'] = strategy_signals['question']
                flattened.append(signal)

        if not flattened:
            return None

        # Sort by confidence and edge
        flattened.sort(key=lambda x: (x.get('confidence', 0), x.get('edge', 0)), reverse=True)

        return flattened[0] if flattened else None
