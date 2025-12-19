"""
On-Balance Volume (OBV) Indicator
Measures buying and selling pressure using volume flow.

Formula:
- If close > prev_close: OBV = prev_OBV + volume
- If close < prev_close: OBV = prev_OBV - volume
- If close = prev_close: OBV = prev_OBV

Signals:
- OBV rising with price rising = Confirmed uptrend
- OBV falling with price falling = Confirmed downtrend
- OBV rising but price flat/falling = Bullish divergence (accumulation)
- OBV falling but price flat/rising = Bearish divergence (distribution)

Why It Matters:
- Volume precedes price - smart money moves before the crowd
- Divergences often signal major reversals
- Used by institutions to detect accumulation/distribution

Default: Uses EMA smoothing for cleaner signals
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass 
class OBVResult:
    """OBV calculation result."""
    obv: float              # Current OBV value
    obv_ema: float          # Smoothed OBV (EMA)
    trend: str              # 'rising', 'falling', 'flat'
    divergence: Optional[str]  # 'bullish', 'bearish', or None
    strength: float         # Trend strength (0-1)


class OBVCalculator:
    """
    On-Balance Volume (OBV) Indicator.
    
    Volume-based indicator that shows accumulation/distribution.
    Professional traders use OBV divergences to catch major reversals.
    
    Usage:
        obv = OnBalanceVolume()
        result = obv.calculate(candles)
        if result.divergence == 'bullish':
            # Smart money accumulating - potential reversal up
    """
    
    def __init__(self, ema_period: int = 20, divergence_lookback: int = 10):
        """
        Initialize OBV.
        
        Args:
            ema_period: Period for OBV smoothing
            divergence_lookback: Candles to look back for divergence
        """
        self.ema_period = ema_period
        self.divergence_lookback = divergence_lookback
        
        # State
        self._obv_history: deque = deque(maxlen=100)
        self._price_history: deque = deque(maxlen=100)
        self._obv_ema: Optional[float] = None
        self._prev_obv_ema: Optional[float] = None
        
        logger.info(f"📊 OBV initialized: EMA={ema_period}, Divergence lookback={divergence_lookback}")
    
    def calculate(self, candles: List[Dict]) -> Optional[OBVResult]:
        """
        Calculate OBV for given candles.
        
        Args:
            candles: OHLCV candles
            
        Returns:
            OBVResult or None
        """
        if len(candles) < 10:
            return None
        
        # Calculate OBV
        obv = 0.0
        obv_values = []
        
        for i in range(1, len(candles)):
            close = float(candles[i].get('close', candles[i].get('c', 0)))
            prev_close = float(candles[i-1].get('close', candles[i-1].get('c', 0)))
            volume = float(candles[i].get('volume', candles[i].get('v', 0)))
            
            if close > prev_close:
                obv += volume
            elif close < prev_close:
                obv -= volume
            # If equal, OBV stays same
            
            obv_values.append(obv)
        
        if not obv_values:
            return None
        
        current_obv = obv_values[-1]
        
        # Calculate EMA of OBV
        if self._obv_ema is None:
            # Initialize with SMA
            if len(obv_values) >= self.ema_period:
                self._obv_ema = sum(obv_values[-self.ema_period:]) / self.ema_period
            else:
                self._obv_ema = current_obv
        else:
            # EMA calculation
            multiplier = 2 / (self.ema_period + 1)
            self._obv_ema = (current_obv * multiplier) + (self._obv_ema * (1 - multiplier))
        
        obv_ema = self._obv_ema
        
        # Determine trend
        if len(obv_values) >= 5:
            recent_obv = obv_values[-5:]
            obv_change = recent_obv[-1] - recent_obv[0]
            avg_obv = sum(abs(v) for v in recent_obv) / len(recent_obv) if recent_obv else 1
            
            if avg_obv > 0:
                change_pct = obv_change / avg_obv
            else:
                change_pct = 0
            
            if change_pct > 0.05:
                trend = 'rising'
                strength = min(1.0, abs(change_pct))
            elif change_pct < -0.05:
                trend = 'falling'
                strength = min(1.0, abs(change_pct))
            else:
                trend = 'flat'
                strength = 0.0
        else:
            trend = 'flat'
            strength = 0.0
        
        # Detect divergence
        divergence = self._detect_divergence(candles, obv_values)
        
        # Store for history
        self._obv_history.append(current_obv)
        self._price_history.append(float(candles[-1].get('close', candles[-1].get('c', 0))))
        self._prev_obv_ema = obv_ema
        
        return OBVResult(
            obv=current_obv,
            obv_ema=obv_ema,
            trend=trend,
            divergence=divergence,
            strength=strength
        )
    
    def _detect_divergence(self, candles: List[Dict], obv_values: List[float]) -> Optional[str]:
        """
        Detect price/OBV divergence.
        
        Bullish divergence: Price making lower lows, OBV making higher lows
        Bearish divergence: Price making higher highs, OBV making lower highs
        """
        if len(candles) < self.divergence_lookback or len(obv_values) < self.divergence_lookback:
            return None
        
        lookback = self.divergence_lookback
        
        # Get recent prices and OBV
        recent_closes = [float(c.get('close', c.get('c', 0))) for c in candles[-lookback:]]
        recent_obv = obv_values[-lookback:]
        
        # Find local extremes
        price_min_idx = recent_closes.index(min(recent_closes))
        price_max_idx = recent_closes.index(max(recent_closes))
        obv_min_idx = recent_obv.index(min(recent_obv))
        obv_max_idx = recent_obv.index(max(recent_obv))
        
        # Check for bullish divergence (price lower low, OBV higher low)
        # Recent price is near low, but OBV is not
        if price_min_idx > lookback * 0.5:  # Recent low in price
            # Compare with earlier low
            early_low = min(recent_closes[:lookback//2])
            late_low = min(recent_closes[lookback//2:])
            early_obv_low = min(recent_obv[:lookback//2])
            late_obv_low = min(recent_obv[lookback//2:])
            
            if late_low < early_low and late_obv_low > early_obv_low:
                return 'bullish'
        
        # Check for bearish divergence (price higher high, OBV lower high)
        if price_max_idx > lookback * 0.5:  # Recent high in price
            early_high = max(recent_closes[:lookback//2])
            late_high = max(recent_closes[lookback//2:])
            early_obv_high = max(recent_obv[:lookback//2])
            late_obv_high = max(recent_obv[lookback//2:])
            
            if late_high > early_high and late_obv_high < early_obv_high:
                return 'bearish'
        
        return None
    
    def get_signal(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trading signal from OBV.
        
        Returns:
            Tuple of (direction, confidence)
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        # Divergence signals are strongest
        if result.divergence == 'bullish':
            return ('long', 0.75)
        elif result.divergence == 'bearish':
            return ('short', 0.75)
        
        # Trend confirmation
        if result.trend == 'rising':
            return ('long', 0.4 + result.strength * 0.3)
        elif result.trend == 'falling':
            return ('short', 0.4 + result.strength * 0.3)
        
        return ('neutral', 0.0)
    
    def reset(self):
        """Reset indicator state."""
        self._obv_history.clear()
        self._price_history.clear()
        self._obv_ema = None
        self._prev_obv_ema = None
