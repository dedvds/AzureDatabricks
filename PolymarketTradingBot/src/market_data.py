"""
Market Data Fetcher for Polymarket

This module fetches market data, probabilities, and sentiment from Polymarket.
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketDataFetcher:
    """Fetches and processes market data from Polymarket API"""

    BASE_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    def __init__(self, rate_limit_delay: float = 0.6):
        """
        Initialize the data fetcher

        Args:
            rate_limit_delay: Delay between API calls to respect rate limits (default 0.6s for ~100 req/min)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def get_all_markets(self, limit: int = 100, offset: int = 0, active: bool = True) -> List[Dict]:
        """
        Fetch all available markets from Polymarket

        Args:
            limit: Number of markets to fetch per request
            offset: Offset for pagination
            active: Only fetch active markets

        Returns:
            List of market dictionaries
        """
        self._rate_limit()

        params = {
            "limit": limit,
            "offset": offset,
            "active": active,
            "closed": False
        }

        try:
            response = requests.get(f"{self.BASE_URL}/markets", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched {len(data)} markets")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_market_details(self, market_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific market

        Args:
            market_id: The ID of the market

        Returns:
            Market details dictionary or None
        """
        self._rate_limit()

        try:
            response = requests.get(f"{self.BASE_URL}/markets/{market_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    def get_market_prices(self, token_id: str) -> Optional[Dict]:
        """
        Get current price data for a market token

        Args:
            token_id: The token ID for the market outcome

        Returns:
            Price data dictionary or None
        """
        self._rate_limit()

        try:
            response = requests.get(f"{self.CLOB_URL}/prices", params={"token_id": token_id}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching prices for token {token_id}: {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """
        Get order book data for a market token

        Args:
            token_id: The token ID for the market outcome

        Returns:
            Order book dictionary with bids and asks
        """
        self._rate_limit()

        try:
            response = requests.get(f"{self.CLOB_URL}/book", params={"token_id": token_id}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching order book for token {token_id}: {e}")
            return None

    def calculate_implied_probability(self, price: float) -> float:
        """
        Calculate implied probability from market price

        Args:
            price: Market price (typically 0-1 or 0-100)

        Returns:
            Implied probability (0-1)
        """
        if price > 1:
            price = price / 100
        return max(0, min(1, price))

    def get_market_sentiment(self, market_id: str) -> Optional[Dict]:
        """
        Analyze market sentiment based on recent trading activity

        Args:
            market_id: The market ID

        Returns:
            Sentiment analysis dictionary
        """
        market_details = self.get_market_details(market_id)
        if not market_details:
            return None

        tokens = market_details.get('tokens', [])
        sentiment_data = {
            'market_id': market_id,
            'question': market_details.get('question', ''),
            'outcomes': [],
            'timestamp': datetime.now().isoformat()
        }

        for token in tokens:
            token_id = token.get('token_id')
            outcome = token.get('outcome')

            price_data = self.get_market_prices(token_id)
            if price_data:
                price = float(price_data.get('mid', 0))
                probability = self.calculate_implied_probability(price)

                sentiment_data['outcomes'].append({
                    'outcome': outcome,
                    'token_id': token_id,
                    'price': price,
                    'implied_probability': probability
                })

        return sentiment_data

    def find_trading_opportunities(self, markets: List[Dict], threshold: float = 0.1) -> List[Dict]:
        """
        Identify potential trading opportunities based on probability analysis

        Args:
            markets: List of market dictionaries
            threshold: Minimum probability edge to consider (default 0.1 = 10%)

        Returns:
            List of trading opportunities
        """
        opportunities = []

        for market in markets:
            market_id = market.get('id')
            if not market_id:
                continue

            sentiment = self.get_market_sentiment(market_id)
            if not sentiment or len(sentiment.get('outcomes', [])) < 2:
                continue

            # Look for binary markets where probabilities don't sum to ~1
            outcomes = sentiment['outcomes']
            if len(outcomes) == 2:
                prob_sum = sum(o['implied_probability'] for o in outcomes)
                prob_deviation = abs(prob_sum - 1.0)

                # Check if there's an arbitrage opportunity or mispricing
                if prob_deviation > threshold:
                    opportunities.append({
                        'market_id': market_id,
                        'question': sentiment['question'],
                        'outcomes': outcomes,
                        'probability_sum': prob_sum,
                        'deviation': prob_deviation,
                        'opportunity_type': 'arbitrage' if prob_sum < 1 else 'overpriced'
                    })

                # Check for extreme probabilities that might revert
                for outcome in outcomes:
                    prob = outcome['implied_probability']
                    if prob < 0.05 or prob > 0.95:
                        opportunities.append({
                            'market_id': market_id,
                            'question': sentiment['question'],
                            'outcome': outcome['outcome'],
                            'probability': prob,
                            'opportunity_type': 'extreme_probability'
                        })

        logger.info(f"Found {len(opportunities)} potential trading opportunities")
        return opportunities
