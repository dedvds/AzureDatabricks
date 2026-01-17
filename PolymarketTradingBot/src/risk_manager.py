"""
Risk Management and Position Manager

Handles position sizing, risk limits, and portfolio management.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position"""

    def __init__(self, market_id: str, token_id: str, outcome: str,
                 side: str, size: float, entry_price: float):
        """
        Initialize a position

        Args:
            market_id: Market identifier
            token_id: Token identifier
            outcome: Outcome name
            side: 'LONG' or 'SHORT'
            size: Position size in USD
            entry_price: Entry price
        """
        self.market_id = market_id
        self.token_id = token_id
        self.outcome = outcome
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.unrealized_pnl = 0.0

    def update_pnl(self, current_price: float):
        """
        Update unrealized PnL based on current price

        Args:
            current_price: Current market price
        """
        if self.side == 'LONG':
            self.unrealized_pnl = self.size * (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.unrealized_pnl = self.size * (self.entry_price - current_price) / self.entry_price

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'market_id': self.market_id,
            'token_id': self.token_id,
            'outcome': self.outcome,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'unrealized_pnl': self.unrealized_pnl
        }


class RiskManager:
    """Manages risk and position sizing"""

    def __init__(self, config: Dict):
        """
        Initialize risk manager

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.total_capital = config.get('total_capital', 10000)
        self.max_position_size = config.get('max_position_size_pct', 0.1) * self.total_capital
        self.max_portfolio_risk = config.get('max_portfolio_risk_pct', 0.2) * self.total_capital
        self.max_positions = config.get('max_positions', 10)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.15)
        self.take_profit_pct = config.get('take_profit_pct', 0.30)

        self.positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        self.total_pnl = 0.0

    def calculate_position_size(self, signal: Dict) -> float:
        """
        Calculate appropriate position size based on signal confidence and risk limits

        Args:
            signal: Trading signal dictionary

        Returns:
            Position size in USD
        """
        confidence = signal.get('confidence', 0.5)
        edge = signal.get('edge', 0)

        # Base position size on confidence
        base_size = self.max_position_size * confidence

        # Adjust for edge
        edge_multiplier = min(2.0, 1 + edge)
        adjusted_size = base_size * edge_multiplier

        # Cap at maximum position size
        position_size = min(adjusted_size, self.max_position_size)

        # Check if we have enough available capital
        current_exposure = self.get_total_exposure()
        available_capital = self.total_capital - current_exposure

        if position_size > available_capital:
            logger.warning(f"Insufficient capital. Requested: ${position_size:.2f}, Available: ${available_capital:.2f}")
            position_size = max(0, available_capital)

        return position_size

    def get_total_exposure(self) -> float:
        """
        Calculate total current exposure across all positions

        Returns:
            Total exposure in USD
        """
        return sum(pos.size for pos in self.positions)

    def can_open_position(self) -> bool:
        """
        Check if we can open a new position based on risk limits

        Returns:
            True if we can open a new position, False otherwise
        """
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False

        current_exposure = self.get_total_exposure()
        if current_exposure >= self.max_portfolio_risk:
            logger.warning(f"Portfolio risk limit reached. Exposure: ${current_exposure:.2f}")
            return False

        return True

    def open_position(self, signal: Dict) -> Optional[Position]:
        """
        Open a new position based on trading signal

        Args:
            signal: Trading signal dictionary

        Returns:
            Position object if opened, None otherwise
        """
        if not self.can_open_position():
            return None

        position_size = self.calculate_position_size(signal)

        if position_size < 10:  # Minimum position size
            logger.warning(f"Position size too small: ${position_size:.2f}")
            return None

        # Determine side based on signal
        side = 'LONG' if signal.get('signal') == 'BUY' else 'SHORT'

        position = Position(
            market_id=signal.get('market_id'),
            token_id=signal.get('token_id'),
            outcome=signal.get('outcome'),
            side=side,
            size=position_size,
            entry_price=signal.get('price', 0)
        )

        self.positions.append(position)
        logger.info(f"Opened {side} position: {position.outcome} - Size: ${position_size:.2f} @ {position.entry_price:.4f}")

        return position

    def close_position(self, position: Position, current_price: float, reason: str = "Manual"):
        """
        Close a position and record PnL

        Args:
            position: Position to close
            current_price: Current market price
            reason: Reason for closing
        """
        position.update_pnl(current_price)
        realized_pnl = position.unrealized_pnl

        self.total_pnl += realized_pnl

        closed_position_data = position.to_dict()
        closed_position_data['exit_price'] = current_price
        closed_position_data['exit_time'] = datetime.now().isoformat()
        closed_position_data['realized_pnl'] = realized_pnl
        closed_position_data['close_reason'] = reason

        self.closed_positions.append(closed_position_data)
        self.positions.remove(position)

        logger.info(f"Closed position: {position.outcome} - PnL: ${realized_pnl:.2f} - Reason: {reason}")

    def update_positions(self, market_prices: Dict[str, float]):
        """
        Update all positions with current prices and check stop loss/take profit

        Args:
            market_prices: Dictionary mapping token_id to current price
        """
        positions_to_close = []

        for position in self.positions:
            current_price = market_prices.get(position.token_id)

            if current_price is None:
                continue

            position.update_pnl(current_price)

            # Check stop loss
            pnl_pct = position.unrealized_pnl / position.size
            if pnl_pct <= -self.stop_loss_pct:
                positions_to_close.append((position, current_price, "Stop Loss"))

            # Check take profit
            elif pnl_pct >= self.take_profit_pct:
                positions_to_close.append((position, current_price, "Take Profit"))

        # Close positions that hit stop loss or take profit
        for position, price, reason in positions_to_close:
            self.close_position(position, price, reason)

    def get_portfolio_summary(self) -> Dict:
        """
        Get summary of current portfolio state

        Returns:
            Portfolio summary dictionary
        """
        total_exposure = self.get_total_exposure()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        total_value = self.total_capital + self.total_pnl + unrealized_pnl

        return {
            'total_capital': self.total_capital,
            'current_value': total_value,
            'total_exposure': total_exposure,
            'available_capital': self.total_capital - total_exposure,
            'active_positions': len(self.positions),
            'realized_pnl': self.total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_pnl + unrealized_pnl,
            'return_pct': ((total_value - self.total_capital) / self.total_capital) * 100
        }

    def get_position_details(self) -> List[Dict]:
        """
        Get details of all active positions

        Returns:
            List of position dictionaries
        """
        return [pos.to_dict() for pos in self.positions]

    def get_trade_history(self) -> List[Dict]:
        """
        Get history of closed trades

        Returns:
            List of closed position dictionaries
        """
        return self.closed_positions

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics

        Returns:
            Dictionary of performance metrics
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p['realized_pnl'] > 0]
        losing_trades = [p for p in self.closed_positions if p['realized_pnl'] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(p['realized_pnl'] for p in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(p['realized_pnl'] for p in losing_trades) / len(losing_trades) if losing_trades else 0

        total_wins = sum(p['realized_pnl'] for p in winning_trades)
        total_losses = abs(sum(p['realized_pnl'] for p in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': 0.0  # Placeholder - would need time series data
        }
