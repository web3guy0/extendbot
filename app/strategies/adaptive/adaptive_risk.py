"""
Adaptive Risk Manager - Dynamic TP/SL Based on Market Conditions
Volatility-adjusted position sizing and risk parameters.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveRiskManager:
    """
    Adaptive Risk Management System
    
    Replaces fixed TP/SL percentages with ATR-based dynamic levels.
    Adjusts based on:
    - Current volatility (ATR)
    - Market regime
    - Trading session
    - Recent performance
    
    Key formulas:
    - Stop Loss = ATR * SL_multiplier
    - Take Profit = ATR * TP_multiplier
    - Position Size = (Account * Risk%) / Stop Distance
    """
    
    # Regime-Based R:R Optimization
    # QUANTITATIVE INSIGHT: Trending markets = wider TP, Ranging = tighter TP
    REGIME_MULTIPLIERS = {
        'TRENDING_UP': {'tp_mult': 2.5, 'sl_mult': 0.8, 'rr': 3.1},    # Ride the trend
        'TRENDING_DOWN': {'tp_mult': 2.5, 'sl_mult': 0.8, 'rr': 3.1},  # Ride the trend
        'RANGING': {'tp_mult': 1.2, 'sl_mult': 0.8, 'rr': 1.5},        # Quick scalps
        'VOLATILE': {'tp_mult': 3.0, 'sl_mult': 1.5, 'rr': 2.0},       # Wider stops
        'BREAKOUT': {'tp_mult': 4.0, 'sl_mult': 0.5, 'rr': 8.0},       # Big move potential
        'LOW_VOL': {'tp_mult': 1.0, 'sl_mult': 0.6, 'rr': 1.7},        # Tight range
    }
    
    def __init__(self):
        """Initialize adaptive risk manager."""
        # Base risk parameters from env
        self.base_risk_per_trade = Decimal(os.getenv('RISK_PER_TRADE_PCT', '2.0'))
        self.max_risk_per_trade = Decimal(os.getenv('MAX_RISK_PER_TRADE_PCT', '3.0'))
        self.base_leverage = int(os.getenv('MAX_LEVERAGE', '5'))
        
        # ATR multipliers for TP/SL - PROFIT FOCUSED
        # With 5x leverage:
        #   - 1% price move = 5% PnL
        #   - SL at 1.2x ATR (~0.35%) = ~1.75% loss
        #   - TP at 4.5x ATR (~1.3%) = ~6.5% gain  
        #   - R:R = 3.75:1 = You can lose 3 trades, win 1, and still profit!
        self.atr_sl_multiplier = Decimal(os.getenv('ATR_SL_MULTIPLIER', '1.2'))
        self.atr_tp_multiplier = Decimal(os.getenv('ATR_TP_MULTIPLIER', '4.5'))
        
        # Minimum/Maximum bounds - REAL MONEY TARGETS
        self.min_sl_pct = Decimal('0.3')   # Minimum 0.3% SL (tight but safe)
        self.max_sl_pct = Decimal('2.0')   # Maximum 2% SL (10% account loss with 5x)
        self.min_tp_pct = Decimal('0.8')   # Minimum 0.8% TP (4% account gain with 5x)
        self.max_tp_pct = Decimal('6.0')   # Maximum 6% TP (30% account gain with 5x)
        
        # Risk reduction after losses
        self.consecutive_loss_count = 0
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
        self.loss_reduction_factor = Decimal('0.5')  # Reduce size by 50% after max losses
        
        # Performance tracking
        self.recent_trades: deque = deque(maxlen=20)
        self.win_rate_20: Decimal = Decimal('0.5')  # Rolling 20-trade win rate
        
        # ATR baseline tracking
        self.atr_history: deque = deque(maxlen=100)
        self.atr_baseline: Optional[Decimal] = None
        
        logger.info("🛡️ Adaptive Risk Manager initialized")
        logger.info(f"   Base Risk: {self.base_risk_per_trade}% per trade")
        logger.info(f"   ATR SL Multiplier: {self.atr_sl_multiplier}x")
        logger.info(f"   ATR TP Multiplier: {self.atr_tp_multiplier}x")
    
    def calculate_adaptive_levels(
        self,
        entry_price: Decimal,
        direction: str,  # 'long' or 'short'
        atr: Decimal,
        regime_params: Optional[Dict] = None,
        session_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Calculate adaptive TP/SL levels based on ATR.
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Current ATR value
            regime_params: Market regime adjustments
            session_params: Session-based adjustments
        
        Returns:
            Dict with stop_loss, take_profit, position_size_pct
        """
        # Ensure all inputs are Decimal to avoid type mismatches
        entry_price = Decimal(str(entry_price)) if not isinstance(entry_price, Decimal) else entry_price
        atr = Decimal(str(atr)) if not isinstance(atr, Decimal) else atr
        
        # Base ATR-based distances
        sl_distance = atr * self.atr_sl_multiplier
        tp_distance = atr * self.atr_tp_multiplier
        
        # Apply regime adjustments
        if regime_params:
            sl_distance *= Decimal(str(regime_params.get('sl_multiplier', 1.0)))
            tp_distance *= Decimal(str(regime_params.get('tp_multiplier', 1.0)))
        
        # Apply session adjustments
        if session_params:
            sl_distance *= Decimal(str(session_params.get('sl_multiplier', 1.0)))
            tp_distance *= Decimal(str(session_params.get('tp_multiplier', 1.0)))
        
        # Convert to percentages
        sl_pct = (sl_distance / entry_price) * 100
        tp_pct = (tp_distance / entry_price) * 100
        
        # Apply bounds
        sl_pct = max(self.min_sl_pct, min(self.max_sl_pct, sl_pct))
        tp_pct = max(self.min_tp_pct, min(self.max_tp_pct, tp_pct))
        
        # Ensure R:R is at least 2.5:1 for sustainable profits
        if tp_pct < sl_pct * Decimal('2.5'):
            tp_pct = sl_pct * Decimal('3.0')  # Force 3:1 minimum R:R
        
        # Calculate actual price levels
        if direction == 'long':
            stop_loss = entry_price * (1 - sl_pct / 100)
            take_profit = entry_price * (1 + tp_pct / 100)
        else:
            stop_loss = entry_price * (1 + sl_pct / 100)
            take_profit = entry_price * (1 - tp_pct / 100)
        
        # Calculate recommended position size
        position_size_pct = self._calculate_position_size(sl_pct)
        
        return {
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'sl_pct': float(sl_pct),
            'tp_pct': float(tp_pct),
            'rr_ratio': float(tp_pct / sl_pct),
            'position_size_pct': float(position_size_pct),
            'atr_used': float(atr),
        }
    
    def get_regime_params(self, regime: str) -> Dict[str, float]:
        """
        Get optimized TP/SL multipliers for a specific market regime.
        
        Regime-Based Optimization:
        - TRENDING: Wide TP (ride the trend), tight SL
        - RANGING: Tight TP (scalp), tight SL  
        - VOLATILE: Wide everything (big moves)
        - BREAKOUT: Very wide TP (capture the move), very tight SL
        
        Args:
            regime: Market regime name (TRENDING_UP, TRENDING_DOWN, RANGING, etc.)
            
        Returns:
            Dict with tp_multiplier, sl_multiplier, expected_rr
        """
        # Normalize regime name
        regime_key = regime.upper().replace(' ', '_')
        
        # Get regime-specific params or default
        if regime_key in self.REGIME_MULTIPLIERS:
            params = self.REGIME_MULTIPLIERS[regime_key]
            return {
                'tp_multiplier': params['tp_mult'],
                'sl_multiplier': params['sl_mult'],
                'expected_rr': params['rr'],
            }
        
        # Check for partial matches
        if 'TREND' in regime_key:
            return {
                'tp_multiplier': 2.5,
                'sl_multiplier': 0.8,
                'expected_rr': 3.1,
            }
        elif 'RANGE' in regime_key or 'SIDEWAYS' in regime_key:
            return {
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.8,
                'expected_rr': 1.5,
            }
        elif 'VOLATILE' in regime_key or 'CHOPPY' in regime_key:
            return {
                'tp_multiplier': 3.0,
                'sl_multiplier': 1.5,
                'expected_rr': 2.0,
            }
        elif 'BREAK' in regime_key:
            return {
                'tp_multiplier': 4.0,
                'sl_multiplier': 0.5,
                'expected_rr': 8.0,
            }
        
        # Default: balanced
        return {
            'tp_multiplier': 1.0,
            'sl_multiplier': 1.0,
            'expected_rr': float(self.atr_tp_multiplier / self.atr_sl_multiplier),
        }
    
    def calculate_position_size(
        self,
        account_value: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        max_position_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate position size based on risk.
        
        Position Size = (Account × Risk%) / |Entry - Stop|
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            stop_loss: Stop loss price
            max_position_pct: Maximum position size %
        
        Returns:
            Position size as percentage of account
        """
        if max_position_pct is None:
            max_position_pct = Decimal(os.getenv('MAX_POSITION_SIZE_PCT', '55'))
        
        # Calculate stop distance in %
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        
        if stop_distance_pct == 0:
            return Decimal('0')
        
        # Risk amount in USD
        risk_pct = self._get_adjusted_risk()
        
        # Position size = Risk% / Stop%
        # With leverage considered
        position_pct = (risk_pct / stop_distance_pct) * self.base_leverage
        
        # Apply consecutive loss reduction
        if self.consecutive_loss_count >= self.max_consecutive_losses:
            position_pct *= self.loss_reduction_factor
            logger.warning(f"⚠️ Position size reduced due to {self.consecutive_loss_count} consecutive losses")
        
        # Cap at maximum
        position_pct = min(position_pct, max_position_pct)
        
        return position_pct
    
    def update_atr_baseline(self, atr: Decimal):
        """Update ATR baseline for volatility comparison."""
        self.atr_history.append(float(atr))
        
        if len(self.atr_history) >= 20:
            self.atr_baseline = Decimal(str(sum(self.atr_history) / len(self.atr_history)))
    
    def get_volatility_state(self, current_atr: Decimal) -> str:
        """Determine volatility state relative to baseline."""
        if not self.atr_baseline:
            return 'normal'
        
        ratio = current_atr / self.atr_baseline
        
        if ratio < Decimal('0.7'):
            return 'low'
        elif ratio > Decimal('1.5'):
            return 'high'
        elif ratio > Decimal('2.0'):
            return 'extreme'
        return 'normal'
    
    def record_trade_result(self, won: bool, pnl_pct: Decimal):
        """Record trade result for adaptive risk adjustment."""
        self.recent_trades.append({
            'won': won,
            'pnl_pct': float(pnl_pct),
            'timestamp': datetime.now(timezone.utc),
        })
        
        if won:
            self.consecutive_loss_count = 0
        else:
            self.consecutive_loss_count += 1
            if self.consecutive_loss_count >= self.max_consecutive_losses:
                logger.warning(f"🔴 {self.consecutive_loss_count} consecutive losses - reducing risk")
        
        # Update rolling win rate
        wins = sum(1 for t in self.recent_trades if t['won'])
        self.win_rate_20 = Decimal(str(wins / len(self.recent_trades)))
    
    def get_trailing_stop_levels(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        direction: str,
        atr: Decimal,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic trailing stop levels based on profit.
        
        Trailing logic:
        - At 2x ATR profit: Move SL to breakeven
        - At 3x ATR profit: Move SL to 1x ATR profit
        - At 4x+ ATR profit: Trail by 1.5x ATR
        """
        pnl_atr = (current_price - entry_price) / atr if direction == 'long' else \
                  (entry_price - current_price) / atr
        
        trailing_stop = None
        action = 'hold'
        
        if pnl_atr >= 4:
            # Trail aggressively
            trail_distance = atr * Decimal('1.5')
            if direction == 'long':
                trailing_stop = current_price - trail_distance
            else:
                trailing_stop = current_price + trail_distance
            action = 'trail_aggressive'
        
        elif pnl_atr >= 3:
            # Move to 1x ATR profit
            if direction == 'long':
                trailing_stop = entry_price + atr
            else:
                trailing_stop = entry_price - atr
            action = 'trail_to_profit'
        
        elif pnl_atr >= 2:
            # Move to breakeven + small buffer
            buffer = atr * Decimal('0.2')
            if direction == 'long':
                trailing_stop = entry_price + buffer
            else:
                trailing_stop = entry_price - buffer
            action = 'move_to_breakeven'
        
        return {
            'trailing_stop': float(trailing_stop) if trailing_stop else None,
            'action': action,
            'pnl_in_atr': float(pnl_atr),
        }
    
    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """Check if exposure should be reduced based on recent performance."""
        if self.consecutive_loss_count >= self.max_consecutive_losses:
            return True, f"Consecutive losses: {self.consecutive_loss_count}"
        
        if len(self.recent_trades) >= 10 and self.win_rate_20 < Decimal('0.35'):
            return True, f"Low win rate: {self.win_rate_20:.1%}"
        
        return False, "Performance OK"
    
    def _get_adjusted_risk(self) -> Decimal:
        """Get risk per trade adjusted for recent performance."""
        risk = self.base_risk_per_trade
        
        # Reduce risk if win rate is low
        if len(self.recent_trades) >= 10:
            if self.win_rate_20 < Decimal('0.4'):
                risk *= Decimal('0.7')
            elif self.win_rate_20 > Decimal('0.7'):
                risk = min(risk * Decimal('1.2'), self.max_risk_per_trade)
        
        return risk
    
    def _calculate_position_size(self, sl_pct: Decimal) -> Decimal:
        """Calculate position size percentage from stop loss percentage."""
        if sl_pct == 0:
            return Decimal('0')
        
        risk_pct = self._get_adjusted_risk()
        
        # Position Size = Risk% / Stop% × Leverage
        position_pct = (risk_pct / sl_pct) * self.base_leverage * 100
        
        # Cap at maximum
        max_pct = Decimal(os.getenv('MAX_POSITION_SIZE_PCT', '55'))
        return min(position_pct, max_pct)
