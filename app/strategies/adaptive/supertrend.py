"""
Supertrend Indicator
A trend-following indicator that provides clear buy/sell signals.

Formula:
- Basic Upper Band = (High + Low) / 2 + Multiplier * ATR
- Basic Lower Band = (High + Low) / 2 - Multiplier * ATR
- Final bands adjust based on previous close position

Signal:
- Price closes above Upper Band → Bullish (buy signal)
- Price closes below Lower Band → Bearish (sell signal)
- Stays in trend until opposite signal

Advantages:
- Clear trend direction
- Built-in volatility adjustment (uses ATR)
- Reduces whipsaw in trending markets
- Great for swing trading

Default: Period=10, Multiplier=2 (same as TradingView default)
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SupertrendDirection(Enum):
    """Supertrend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SupertrendResult:
    """Supertrend calculation result."""
    direction: SupertrendDirection
    value: Decimal  # Current supertrend line value
    upper_band: Decimal
    lower_band: Decimal
    changed: bool  # True if direction just changed
    strength: float  # Distance from price as % (larger = stronger trend)


class SupertrendIndicator:
    """
    Supertrend Indicator Calculator.
    
    A volatility-based trend indicator that's excellent for:
    - Identifying trend direction
    - Setting trailing stops
    - Filtering out noise/whipsaw
    
    Usage:
        st = SupertrendIndicator(period=10, multiplier=2.0)
        result = st.calculate(candles)
        if result.direction == SupertrendDirection.BULLISH:
            # Consider long positions
    """
    
    def __init__(self, period: int = 10, multiplier: float = 2.0):
        """
        Initialize Supertrend indicator.
        
        Args:
            period: ATR period (default 10)
            multiplier: ATR multiplier for bands (default 2.0)
        """
        self.period = period
        self.multiplier = Decimal(str(multiplier))
        
        # State tracking
        self._prev_upper_band: Optional[Decimal] = None
        self._prev_lower_band: Optional[Decimal] = None
        self._prev_supertrend: Optional[Decimal] = None
        self._prev_direction: SupertrendDirection = SupertrendDirection.NEUTRAL
        
        logger.info(f"📈 Supertrend initialized: period={period}, multiplier={multiplier}")
    
    def calculate(self, candles: List[Dict]) -> Optional[SupertrendResult]:
        """
        Calculate Supertrend for given candles.
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            SupertrendResult or None if insufficient data
        """
        if len(candles) < self.period + 1:
            return None
        
        # Calculate ATR
        atr = self._calculate_atr(candles)
        if atr is None or atr <= 0:
            return None
        
        # Get current candle data
        current = candles[-1]
        high = Decimal(str(current.get('high', current.get('h', 0))))
        low = Decimal(str(current.get('low', current.get('l', 0))))
        close = Decimal(str(current.get('close', current.get('c', 0))))
        
        # Calculate HL2 (typical price for bands)
        hl2 = (high + low) / 2
        
        # Calculate basic bands
        basic_upper = hl2 + (self.multiplier * atr)
        basic_lower = hl2 - (self.multiplier * atr)
        
        # Get previous close
        prev_close = Decimal(str(candles[-2].get('close', candles[-2].get('c', 0))))
        
        # Calculate final upper band
        if self._prev_upper_band is None:
            final_upper = basic_upper
        else:
            if basic_upper < self._prev_upper_band or prev_close > self._prev_upper_band:
                final_upper = basic_upper
            else:
                final_upper = self._prev_upper_band
        
        # Calculate final lower band
        if self._prev_lower_band is None:
            final_lower = basic_lower
        else:
            if basic_lower > self._prev_lower_band or prev_close < self._prev_lower_band:
                final_lower = basic_lower
            else:
                final_lower = self._prev_lower_band
        
        # Determine supertrend value and direction
        if self._prev_supertrend is None:
            # Initial state - use close position to determine
            if close > final_upper:
                supertrend = final_lower
                direction = SupertrendDirection.BULLISH
            else:
                supertrend = final_upper
                direction = SupertrendDirection.BEARISH
        else:
            # Check for direction change
            if self._prev_supertrend == self._prev_upper_band:
                # Was bearish
                if close > final_upper:
                    # Flip to bullish
                    supertrend = final_lower
                    direction = SupertrendDirection.BULLISH
                else:
                    supertrend = final_upper
                    direction = SupertrendDirection.BEARISH
            else:
                # Was bullish
                if close < final_lower:
                    # Flip to bearish
                    supertrend = final_upper
                    direction = SupertrendDirection.BEARISH
                else:
                    supertrend = final_lower
                    direction = SupertrendDirection.BULLISH
        
        # Check if direction changed
        changed = direction != self._prev_direction and self._prev_direction != SupertrendDirection.NEUTRAL
        
        # Calculate strength (distance from price as %)
        if close > 0:
            strength = float(abs(close - supertrend) / close * 100)
        else:
            strength = 0.0
        
        # Update state
        self._prev_upper_band = final_upper
        self._prev_lower_band = final_lower
        self._prev_supertrend = supertrend
        self._prev_direction = direction
        
        result = SupertrendResult(
            direction=direction,
            value=supertrend,
            upper_band=final_upper,
            lower_band=final_lower,
            changed=changed,
            strength=strength,
        )
        
        if changed:
            logger.info(f"🔄 Supertrend flipped to {direction.value.upper()} @ {float(supertrend):.2f}")
        
        return result
    
    def _calculate_atr(self, candles: List[Dict]) -> Optional[Decimal]:
        """Calculate ATR using True Range."""
        if len(candles) < self.period + 1:
            return None
        
        tr_list = []
        for i in range(1, len(candles)):
            high = Decimal(str(candles[i].get('high', candles[i].get('h', 0))))
            low = Decimal(str(candles[i].get('low', candles[i].get('l', 0))))
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            
            # True Range = max(H-L, |H-prev_close|, |L-prev_close|)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        if len(tr_list) < self.period:
            return None
        
        # Simple average of last 'period' true ranges
        return sum(tr_list[-self.period:]) / self.period
    
    def get_signal(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trading signal from Supertrend.
        
        Returns:
            Tuple of (direction, confidence)
            direction: 'long', 'short', or 'neutral'
            confidence: 0.0 to 1.0
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        # Base confidence from direction
        if result.direction == SupertrendDirection.BULLISH:
            direction = 'long'
        elif result.direction == SupertrendDirection.BEARISH:
            direction = 'short'
        else:
            return ('neutral', 0.0)
        
        # Confidence based on strength (distance from line)
        # 0.5% = 50% confidence, 1% = 75%, 2%+ = 90%+
        confidence = min(0.95, 0.3 + result.strength * 0.3)
        
        # Boost confidence if direction just changed (fresh signal)
        if result.changed:
            confidence = min(0.95, confidence + 0.15)
        
        return (direction, confidence)
    
    def reset(self):
        """Reset indicator state."""
        self._prev_upper_band = None
        self._prev_lower_band = None
        self._prev_supertrend = None
        self._prev_direction = SupertrendDirection.NEUTRAL
