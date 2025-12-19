"""
Donchian Channel Indicator
A volatility channel showing the highest high and lowest low over N periods.

Formula:
- Upper Band = Highest High over N periods
- Lower Band = Lowest Low over N periods
- Middle Band = (Upper + Lower) / 2

Signals:
- Price breaks above Upper → Strong bullish breakout
- Price breaks below Lower → Strong bearish breakdown
- Price near Middle → Equilibrium/ranging

Trading Uses:
1. Breakout Trading: Enter when price breaks channel
2. Trend Following: Stay long above middle, short below
3. Support/Resistance: Bands act as dynamic S/R
4. Volatility: Channel width shows volatility

Default: Period=50 (same as your TradingView chart "DC 50 0")
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DonchianPosition(Enum):
    """Price position relative to Donchian Channel."""
    ABOVE_UPPER = "above_upper"  # Breakout bullish
    UPPER_ZONE = "upper_zone"    # Upper half
    MIDDLE = "middle"            # Near middle
    LOWER_ZONE = "lower_zone"    # Lower half
    BELOW_LOWER = "below_lower"  # Breakdown bearish


@dataclass
class DonchianResult:
    """Donchian Channel calculation result."""
    upper: Decimal      # Highest high
    lower: Decimal      # Lowest low
    middle: Decimal     # (Upper + Lower) / 2
    width: Decimal      # Channel width (upper - lower)
    width_pct: float    # Width as % of middle
    position: DonchianPosition  # Where price is
    breakout: Optional[str]  # 'bullish', 'bearish', or None
    squeeze: bool       # True if channel is narrow (low volatility)


class DonchianChannel:
    """
    Donchian Channel Indicator Calculator.
    
    Excellent for:
    - Breakout detection
    - Trend direction (price above/below middle)
    - Volatility measurement (channel width)
    - Setting stops at channel boundaries
    
    Usage:
        dc = DonchianChannel(period=50)
        result = dc.calculate(candles)
        if result.breakout == 'bullish':
            # Strong buy signal
    """
    
    def __init__(self, period: int = 50, offset: int = 0):
        """
        Initialize Donchian Channel.
        
        Args:
            period: Lookback period for high/low (default 50)
            offset: Offset for calculation (default 0)
        """
        self.period = period
        self.offset = offset
        
        # State for breakout detection
        self._prev_upper: Optional[Decimal] = None
        self._prev_lower: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None
        
        # Track channel width for squeeze detection
        self._width_history: List[float] = []
        self._max_history = 20
        
        logger.info(f"📊 Donchian Channel initialized: period={period}, offset={offset}")
    
    def calculate(self, candles: List[Dict]) -> Optional[DonchianResult]:
        """
        Calculate Donchian Channel for given candles.
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            DonchianResult or None if insufficient data
        """
        if len(candles) < self.period:
            return None
        
        # Get the range of candles to analyze
        start_idx = len(candles) - self.period - self.offset
        end_idx = len(candles) - self.offset
        
        if start_idx < 0:
            start_idx = 0
        
        analysis_candles = candles[start_idx:end_idx]
        
        if not analysis_candles:
            return None
        
        # Calculate highest high and lowest low
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in analysis_candles]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in analysis_candles]
        
        upper = max(highs)
        lower = min(lows)
        middle = (upper + lower) / 2
        width = upper - lower
        
        # Width as percentage
        width_pct = float(width / middle * 100) if middle > 0 else 0.0
        
        # Track width history for squeeze detection
        self._width_history.append(width_pct)
        if len(self._width_history) > self._max_history:
            self._width_history.pop(0)
        
        # Detect squeeze (narrow channel = low volatility, often precedes big move)
        squeeze = False
        if len(self._width_history) >= 5:
            avg_width = sum(self._width_history) / len(self._width_history)
            squeeze = width_pct < avg_width * 0.7  # 30% below average
        
        # Get current price
        current = candles[-1]
        close = Decimal(str(current.get('close', current.get('c', 0))))
        high = Decimal(str(current.get('high', current.get('h', 0))))
        low = Decimal(str(current.get('low', current.get('l', 0))))
        
        # Determine position within channel
        if close > upper:
            position = DonchianPosition.ABOVE_UPPER
        elif close < lower:
            position = DonchianPosition.BELOW_LOWER
        elif close > middle + (width * Decimal('0.25')):
            position = DonchianPosition.UPPER_ZONE
        elif close < middle - (width * Decimal('0.25')):
            position = DonchianPosition.LOWER_ZONE
        else:
            position = DonchianPosition.MIDDLE
        
        # Detect breakouts
        breakout = None
        if self._prev_upper is not None and self._prev_close is not None:
            # Bullish breakout: price was below upper, now closes above
            if self._prev_close <= self._prev_upper and close > upper:
                breakout = 'bullish'
                logger.info(f"🚀 Donchian BULLISH breakout @ {float(close):.2f} (above {float(upper):.2f})")
            # Bearish breakdown: price was above lower, now closes below
            elif self._prev_close >= self._prev_lower and close < lower:
                breakout = 'bearish'
                logger.info(f"📉 Donchian BEARISH breakdown @ {float(close):.2f} (below {float(lower):.2f})")
        
        # Update state
        self._prev_upper = upper
        self._prev_lower = lower
        self._prev_close = close
        
        return DonchianResult(
            upper=upper,
            lower=lower,
            middle=middle,
            width=width,
            width_pct=width_pct,
            position=position,
            breakout=breakout,
            squeeze=squeeze,
        )
    
    def get_signal(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trading signal from Donchian Channel.
        
        Returns:
            Tuple of (direction, confidence)
            direction: 'long', 'short', or 'neutral'
            confidence: 0.0 to 1.0
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        # Strong signals from breakouts
        if result.breakout == 'bullish':
            return ('long', 0.85)
        elif result.breakout == 'bearish':
            return ('short', 0.85)
        
        # Position-based signals
        if result.position == DonchianPosition.ABOVE_UPPER:
            return ('long', 0.7)
        elif result.position == DonchianPosition.BELOW_LOWER:
            return ('short', 0.7)
        elif result.position == DonchianPosition.UPPER_ZONE:
            return ('long', 0.5)
        elif result.position == DonchianPosition.LOWER_ZONE:
            return ('short', 0.5)
        
        # Middle = neutral
        return ('neutral', 0.3)
    
    def get_trend_bias(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trend bias from price position relative to middle.
        
        Returns:
            Tuple of (bias, strength)
            bias: 'bullish', 'bearish', or 'neutral'
            strength: 0.0 to 1.0
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        close = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
        
        # Distance from middle as % of half-channel
        half_width = result.width / 2
        if half_width > 0:
            distance_from_middle = close - result.middle
            normalized_distance = float(distance_from_middle / half_width)
        else:
            normalized_distance = 0.0
        
        if normalized_distance > 0.2:
            bias = 'bullish'
            strength = min(1.0, abs(normalized_distance))
        elif normalized_distance < -0.2:
            bias = 'bearish'
            strength = min(1.0, abs(normalized_distance))
        else:
            bias = 'neutral'
            strength = 0.0
        
        return (bias, strength)
    
    def is_squeeze(self, candles: List[Dict]) -> bool:
        """Check if channel is in a squeeze (low volatility)."""
        result = self.calculate(candles)
        return result.squeeze if result else False
    
    def reset(self):
        """Reset indicator state."""
        self._prev_upper = None
        self._prev_lower = None
        self._prev_close = None
        self._width_history = []
