"""
Unit tests for market data fetcher

Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from market_data import PolymarketDataFetcher


def test_data_fetcher_initialization():
    """Test that data fetcher initializes correctly"""
    fetcher = PolymarketDataFetcher()
    assert fetcher is not None
    assert fetcher.BASE_URL == "https://gamma-api.polymarket.com"
    assert fetcher.CLOB_URL == "https://clob.polymarket.com"


def test_calculate_implied_probability():
    """Test probability calculation from prices"""
    fetcher = PolymarketDataFetcher()

    # Test with decimal prices
    assert fetcher.calculate_implied_probability(0.5) == 0.5
    assert fetcher.calculate_implied_probability(0.25) == 0.25
    assert fetcher.calculate_implied_probability(0.75) == 0.75

    # Test with percentage prices
    assert fetcher.calculate_implied_probability(50) == 0.5
    assert fetcher.calculate_implied_probability(25) == 0.25
    assert fetcher.calculate_implied_probability(75) == 0.75

    # Test bounds
    assert fetcher.calculate_implied_probability(0) == 0
    assert fetcher.calculate_implied_probability(1) == 1
    assert fetcher.calculate_implied_probability(100) == 1


def test_rate_limiting():
    """Test that rate limiting is enforced"""
    import time

    fetcher = PolymarketDataFetcher(rate_limit_delay=0.1)

    start = time.time()
    fetcher._rate_limit()
    fetcher._rate_limit()
    elapsed = time.time() - start

    # Should take at least 0.1 seconds due to rate limiting
    assert elapsed >= 0.1


if __name__ == "__main__":
    # Run basic tests
    test_data_fetcher_initialization()
    test_calculate_implied_probability()
    test_rate_limiting()
    print("All tests passed!")
