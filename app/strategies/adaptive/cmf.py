"""
Chaikin Money Flow (CMF) Indicator
Measures the amount of money flow volume over a period.

Formula:
1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
2. Money Flow Volume = Multiplier * Volume
3. CMF = Sum(MF Volume, N) / Sum(Volume, N)

Range: -1 to +1
- CMF > 0 = Buying pressure (accumulation)
- CMF < 0 = Selling pressure (distribution)
- CMF > 0.25 = Strong buying
- CMF < -0.25 = Strong selling

Why It's Used:
- Shows if price moves are backed by institutional buying/selling
- Divergences between CMF and price indicate potential reversals
- More reliable than simple volume analysis
- Used by hedge funds to detect "smart money" moves

Default: Period=20 (industry standard)
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CMFResult:
    """CMF calculation result."""
    cmf: float              # CMF value (-1 to +1)
    zone: str               # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    trend: str              # 'rising', 'falling', 'flat'
    divergence: Optional[str]  # 'bullish', 'bearish', or None
    strength: float         # Absolute CMF value (0-1)


class ChaikinMoneyFlow:
    """
    Chaikin Money Flow (CMF) Indicator.
    
    Measures buying and selling pressure based on where price closes
    within its range, weighted by volume.
    
    Professional traders use CMF to:
    - Confirm breakouts (CMF should be positive for bullish breaks)
    - Detect accumulation before price rises
    - Detect distribution before price falls
    - Spot divergences for reversals
    
    Usage:
        cmf = ChaikinMoneyFlow()
        result = cmf.calculate(candles)
        if result.zone == 'strong_buy':
            # Heavy institutional buying
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize CMF.
        
        Args:
            period: Lookback period (default 20)
        """
        self.period = period
        
        # State
        self._cmf_history: deque = deque(maxlen=30)
        self._price_history: deque = deque(maxlen=30)
        
        logger.info(f"📊 Chaikin Money Flow initialized: period={period}")
    
    def calculate(self, candles: List[Dict]) -> Optional[CMFResult]:
        """
        Calculate CMF for given candles.
        
        Args:
            candles: OHLCV candles
            
        Returns:
            CMFResult or None
        """
        if len(candles) < self.period:
            return None
        
        # Calculate Money Flow Volume for each candle in period
        mf_volumes = []
        volumes = []
        
        for candle in candles[-self.period:]:
            high = float(candle.get('high', candle.get('h', 0)))
            low = float(candle.get('low', candle.get('l', 0)))
            close = float(candle.get('close', candle.get('c', 0)))
            volume = float(candle.get('volume', candle.get('v', 0)))
            
            # Money Flow Multiplier
            if high - low > 0:
                mf_multiplier = ((close - low) - (high - close)) / (high - low)
            else:
                mf_multiplier = 0
            
            # Money Flow Volume
            mf_volume = mf_multiplier * volume
            
            mf_volumes.append(mf_volume)
            volumes.append(volume)
        
        # CMF = Sum(MF Volume) / Sum(Volume)
        total_volume = sum(volumes)
        if total_volume > 0:
            cmf = sum(mf_volumes) / total_volume
        else:
            cmf = 0
        
        # Determine zone
        if cmf > 0.25:
            zone = 'strong_buy'
        elif cmf > 0.05:
            zone = 'buy'
        elif cmf < -0.25:
            zone = 'strong_sell'
        elif cmf < -0.05:
            zone = 'sell'
        else:
            zone = 'neutral'
        
        # Determine trend
        self._cmf_history.append(cmf)
        if len(self._cmf_history) >= 5:
            recent = list(self._cmf_history)[-5:]
            cmf_change = recent[-1] - recent[0]
            
            if cmf_change > 0.05:
                trend = 'rising'
            elif cmf_change < -0.05:
                trend = 'falling'
            else:
                trend = 'flat'
        else:
            trend = 'flat'
        
        # Store price for divergence detection
        current_price = float(candles[-1].get('close', candles[-1].get('c', 0)))
        self._price_history.append(current_price)
        
        # Detect divergence
        divergence = self._detect_divergence()
        
        # Strength = absolute CMF value
        strength = min(1.0, abs(cmf))
        
        return CMFResult(
            cmf=cmf,
            zone=zone,
            trend=trend,
            divergence=divergence,
            strength=strength
        )
    
    def _detect_divergence(self) -> Optional[str]:
        """Detect price/CMF divergence."""
        if len(self._cmf_history) < 10 or len(self._price_history) < 10:
            return None
        
        recent_cmf = list(self._cmf_history)[-10:]
        recent_price = list(self._price_history)[-10:]
        
        # Split into first half and second half
        first_half_price = recent_price[:5]
        second_half_price = recent_price[5:]
        first_half_cmf = recent_cmf[:5]
        second_half_cmf = recent_cmf[5:]
        
        # Bullish divergence: Price lower, CMF higher
        if max(second_half_price) < max(first_half_price):
            # Price making lower highs
            if max(second_half_cmf) > max(first_half_cmf):
                # CMF making higher highs
                return 'bullish'
        
        if min(second_half_price) < min(first_half_price):
            # Price making lower lows
            if min(second_half_cmf) > min(first_half_cmf):
                # CMF making higher lows
                return 'bullish'
        
        # Bearish divergence: Price higher, CMF lower
        if min(second_half_price) > min(first_half_price):
            # Price making higher lows
            if min(second_half_cmf) < min(first_half_cmf):
                # CMF making lower lows
                return 'bearish'
        
        if max(second_half_price) > max(first_half_price):
            # Price making higher highs
            if max(second_half_cmf) < max(first_half_cmf):
                # CMF making lower highs
                return 'bearish'
        
        return None
    
    def get_signal(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trading signal from CMF.
        
        Returns:
            Tuple of (direction, confidence)
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        # Divergence signals
        if result.divergence == 'bullish':
            return ('long', 0.8)
        elif result.divergence == 'bearish':
            return ('short', 0.8)
        
        # Zone-based signals
        if result.zone == 'strong_buy':
            return ('long', 0.7 + result.strength * 0.2)
        elif result.zone == 'buy':
            return ('long', 0.5)
        elif result.zone == 'strong_sell':
            return ('short', 0.7 + result.strength * 0.2)
        elif result.zone == 'sell':
            return ('short', 0.5)
        
        return ('neutral', 0.0)
    
    def confirms_breakout(self, candles: List[Dict], direction: str) -> bool:
        """
        Check if CMF confirms a breakout.
        
        Args:
            candles: OHLCV candles
            direction: 'long' or 'short'
            
        Returns:
            True if CMF confirms the breakout
        """
        result = self.calculate(candles)
        if result is None:
            return False
        
        if direction == 'long':
            return result.cmf > 0.1 and result.trend in ['rising', 'flat']
        else:
            return result.cmf < -0.1 and result.trend in ['falling', 'flat']
    
    def reset(self):
        """Reset indicator state."""
        self._cmf_history.clear()
        self._price_history.clear()
