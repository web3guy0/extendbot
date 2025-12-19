"""
Kelly Criterion Position Sizing
Optimal bet sizing for maximum long-term growth while minimizing risk of ruin.

Formula: f* = (p * b - q) / b
Where:
    f* = Optimal fraction of capital to risk
    p = Win probability
    q = Loss probability (1 - p)
    b = Win/Loss ratio (average win / average loss)

This implementation uses FRACTIONAL KELLY for safety:
- Full Kelly: Maximum growth, high volatility
- Half Kelly: Good growth, moderate volatility (RECOMMENDED)
- Quarter Kelly: Steady growth, low volatility
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly calculation result"""
    full_kelly: float          # Full Kelly fraction
    recommended_fraction: float # Fractional Kelly (safer)
    position_size_pct: float   # Recommended position size %
    win_rate: float            # Historical win rate
    win_loss_ratio: float      # Average win / average loss
    confidence: float          # Confidence in estimate (0-1)
    sample_size: int           # Number of trades analyzed
    edge: float               # Mathematical edge (expected value per trade)


class KellyCriterion:
    """
    Kelly Criterion Calculator with Adaptive Position Sizing
    
    Features:
    - Calculates optimal position size based on historical performance
    - Uses fractional Kelly for safety (default: Half Kelly)
    - Adapts to recent performance with rolling window
    - Provides confidence scores based on sample size
    - Integrates with risk engine for position sizing
    """
    
    # Minimum trades required for reliable Kelly estimate
    MIN_TRADES_FOR_KELLY = 20
    
    # Default fractional Kelly multiplier (0.5 = Half Kelly)
    DEFAULT_KELLY_FRACTION = 0.5
    
    # Maximum allowed Kelly fraction (safety cap)
    MAX_KELLY_FRACTION = 0.25  # Never risk more than 25%
    
    # Minimum position size %
    MIN_POSITION_SIZE_PCT = 5.0
    
    # Rolling window for recent performance (days)
    ROLLING_WINDOW_DAYS = 30
    
    def __init__(self, 
                 kelly_fraction: float = 0.5,
                 min_trades: int = 20,
                 max_position_pct: float = 25.0,
                 min_position_pct: float = 5.0):
        """
        Initialize Kelly Criterion calculator
        
        Args:
            kelly_fraction: Fractional Kelly multiplier (0.5 = Half Kelly)
            min_trades: Minimum trades for reliable estimate
            max_position_pct: Maximum position size %
            min_position_pct: Minimum position size %
        """
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        
        # Trade history for calculations
        self.trade_history: List[Dict[str, Any]] = []
        
        # Cached Kelly result
        self._cached_result: Optional[KellyResult] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)  # Recalculate every 15 min
        
        logger.info("📊 Kelly Criterion Calculator initialized")
        logger.info(f"   Fractional Kelly: {kelly_fraction:.0%}")
        logger.info(f"   Min trades for estimate: {min_trades}")
        logger.info(f"   Position range: {min_position_pct}% - {max_position_pct}%")
    
    def add_trade(self, pnl: float, entry_price: float, exit_price: float,
                  size: float, side: str, strategy: str = None):
        """
        Record a completed trade for Kelly calculation
        
        Args:
            pnl: Realized P&L in USD
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            side: 'long' or 'short'
            strategy: Strategy name (optional)
        """
        trade = {
            'timestamp': datetime.now(timezone.utc),
            'pnl': pnl,
            'pnl_pct': (pnl / (size * entry_price)) * 100,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'side': side,
            'strategy': strategy,
            'is_winner': pnl > 0
        }
        
        self.trade_history.append(trade)
        
        # Invalidate cache
        self._cached_result = None
        
        # Keep only recent trades (rolling window)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.ROLLING_WINDOW_DAYS * 2)
        self.trade_history = [t for t in self.trade_history if t['timestamp'] > cutoff]
        
        logger.debug(f"Kelly: Added trade P&L ${pnl:+.2f}, total trades: {len(self.trade_history)}")
    
    def load_trade_history(self, trades: List[Dict[str, Any]]):
        """Load historical trades from database or file with validation"""
        loaded_count = 0
        skipped_count = 0
        
        for trade in trades:
            # CRITICAL FIX: Validate trade data before processing
            try:
                pnl = trade.get('pnl', trade.get('realized_pnl', 0))
                
                # Skip invalid PnL values
                if pnl is None:
                    skipped_count += 1
                    continue
                    
                # Convert to float with validation
                try:
                    pnl_float = float(pnl)
                except (ValueError, TypeError):
                    logger.warning(f"Kelly: Invalid PnL value: {pnl}")
                    skipped_count += 1
                    continue
                
                # Skip zero PnL trades (breakeven - no information)
                if pnl_float == 0:
                    skipped_count += 1
                    continue
                
                # Validate timestamp
                timestamp = trade.get('timestamp')
                if timestamp is None:
                    timestamp = datetime.now(timezone.utc)
                elif isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)
                
                # Validate pnl_pct if present
                pnl_pct = trade.get('pnl_pct', 0)
                try:
                    pnl_pct = float(pnl_pct)
                except (ValueError, TypeError):
                    pnl_pct = 0
                
                self.trade_history.append({
                    'timestamp': timestamp,
                    'pnl': pnl_float,
                    'pnl_pct': pnl_pct,
                    'is_winner': pnl_float > 0,
                    'strategy': trade.get('strategy')
                })
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Kelly: Error processing trade: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"📊 Loaded {loaded_count} trades for Kelly calculation (skipped {skipped_count} invalid)")
        self._cached_result = None
    
    def calculate(self, strategy: str = None) -> KellyResult:
        """
        Calculate optimal Kelly fraction
        
        Args:
            strategy: Optional strategy filter
            
        Returns:
            KellyResult with optimal position sizing
        """
        # Check cache
        if (self._cached_result and self._cache_time and 
            datetime.now(timezone.utc) - self._cache_time < self._cache_ttl):
            return self._cached_result
        
        # Filter trades
        if strategy:
            trades = [t for t in self.trade_history if t.get('strategy') == strategy]
        else:
            trades = self.trade_history
        
        # Apply rolling window
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.ROLLING_WINDOW_DAYS)
        recent_trades = [t for t in trades if t['timestamp'] > cutoff]
        
        # Use recent if enough, otherwise all trades
        if len(recent_trades) >= self.min_trades:
            trades = recent_trades
        
        sample_size = len(trades)
        
        # Not enough trades - use conservative defaults
        if sample_size < self.min_trades:
            result = KellyResult(
                full_kelly=0.0,
                recommended_fraction=0.0,
                position_size_pct=self.min_position_pct,
                win_rate=0.0,
                win_loss_ratio=0.0,
                confidence=0.0,
                sample_size=sample_size,
                edge=0.0
            )
            logger.info(f"📊 Kelly: Insufficient data ({sample_size}/{self.min_trades} trades) - using min size {self.min_position_pct}%")
            return result
        
        # Calculate statistics
        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]
        
        win_rate = len(winners) / sample_size
        loss_rate = 1 - win_rate
        
        # Calculate average wins and losses
        avg_win = sum(abs(t['pnl']) for t in winners) / len(winners) if winners else 0
        avg_loss = sum(abs(t['pnl']) for t in losers) / len(losers) if losers else 1
        
        # Win/Loss ratio (b)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly Formula: f* = (p * b - q) / b
        if win_loss_ratio > 0:
            full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        else:
            full_kelly = 0
        
        # Edge (expected value per $1 risked)
        edge = win_rate * win_loss_ratio - loss_rate
        
        # Apply fractional Kelly
        fractional_kelly = max(0, full_kelly * self.kelly_fraction)
        
        # Cap at maximum
        capped_kelly = min(fractional_kelly, self.MAX_KELLY_FRACTION)
        
        # Convert to position size %
        position_size_pct = capped_kelly * 100
        
        # Apply min/max bounds
        position_size_pct = max(self.min_position_pct, min(self.max_position_pct, position_size_pct))
        
        # Confidence based on sample size (more trades = higher confidence)
        # 100 trades = 100% confidence
        confidence = min(1.0, sample_size / 100)
        
        # Blend with minimum based on confidence
        # Low confidence = closer to minimum position size
        blended_size = (position_size_pct * confidence + 
                       self.min_position_pct * (1 - confidence))
        
        result = KellyResult(
            full_kelly=full_kelly,
            recommended_fraction=capped_kelly,
            position_size_pct=blended_size,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            confidence=confidence,
            sample_size=sample_size,
            edge=edge
        )
        
        # Cache result
        self._cached_result = result
        self._cache_time = datetime.now(timezone.utc)
        
        logger.info(f"📊 Kelly Calculation:")
        logger.info(f"   Win Rate: {win_rate:.1%} ({len(winners)}/{sample_size})")
        logger.info(f"   Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        logger.info(f"   Win/Loss Ratio: {win_loss_ratio:.2f}")
        logger.info(f"   Full Kelly: {full_kelly:.1%}")
        logger.info(f"   Recommended: {capped_kelly:.1%} ({self.kelly_fraction:.0%} fractional)")
        logger.info(f"   Position Size: {blended_size:.1f}% (confidence: {confidence:.0%})")
        logger.info(f"   Edge: {edge:.3f} (expected value per $1)")
        
        return result
    
    def get_position_size_pct(self, strategy: str = None) -> float:
        """
        Get recommended position size percentage
        
        Returns:
            Position size as percentage of equity
        """
        result = self.calculate(strategy)
        return result.position_size_pct
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics for display"""
        result = self.calculate()
        
        return {
            'trades_analyzed': result.sample_size,
            'win_rate': f"{result.win_rate:.1%}",
            'win_loss_ratio': f"{result.win_loss_ratio:.2f}",
            'full_kelly': f"{result.full_kelly:.1%}",
            'recommended_size': f"{result.position_size_pct:.1f}%",
            'confidence': f"{result.confidence:.0%}",
            'edge': f"{result.edge:.3f}",
            'status': 'Active' if result.sample_size >= self.min_trades else 'Warming Up'
        }
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if Kelly recommends trading
        
        Returns:
            (should_trade, reason)
        """
        result = self.calculate()
        
        # Not enough data - be cautious but allow trading
        if result.sample_size < self.min_trades:
            return True, f"Warming up ({result.sample_size}/{self.min_trades} trades)"
        
        # Negative edge - stop trading!
        if result.edge < 0:
            return False, f"Negative edge ({result.edge:.3f}) - strategy losing money"
        
        # Very low edge - warn but allow
        if result.edge < 0.1:
            return True, f"Low edge ({result.edge:.3f}) - consider reviewing strategy"
        
        # Good edge
        return True, f"Positive edge ({result.edge:.3f})"


# Singleton instance for global access
_kelly_instance: Optional[KellyCriterion] = None


def get_kelly_calculator() -> KellyCriterion:
    """Get or create global Kelly calculator instance"""
    global _kelly_instance
    if _kelly_instance is None:
        _kelly_instance = KellyCriterion()
    return _kelly_instance
