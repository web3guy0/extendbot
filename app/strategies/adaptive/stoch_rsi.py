"""
Stochastic RSI Indicator
Combines the sensitivity of Stochastic with the trend-following of RSI.

Formula:
1. Calculate RSI
2. Apply Stochastic formula to RSI values:
   StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
3. Apply smoothing (K and D lines)

Signals:
- StochRSI < 0.2 = Oversold (buy opportunity)
- StochRSI > 0.8 = Overbought (sell opportunity)
- K crossing above D = Bullish
- K crossing below D = Bearish

Advantages over regular RSI:
- More sensitive to price changes
- Better at catching exact reversal points
- Clearer overbought/oversold signals

Default: RSI Period=14, Stoch Period=14, K=3, D=3
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StochRSIResult:
    """Stochastic RSI calculation result."""
    stoch_rsi: float      # Raw StochRSI (0-1)
    k_line: float         # Smoothed K line
    d_line: float         # Signal line (SMA of K)
    zone: str             # 'oversold', 'overbought', 'neutral'
    crossover: Optional[str]  # 'bullish', 'bearish', or None
    strength: float       # How deep into zone (0-1)


class StochRSICalculator:
    """
    Stochastic RSI Indicator.
    
    More sensitive than regular RSI, better for catching reversals.
    Used by prop traders for precise entry timing.
    
    Usage:
        srsi = StochasticRSI()
        result = srsi.calculate(candles)
        if result.zone == 'oversold' and result.crossover == 'bullish':
            # Strong buy signal
    """
    
    def __init__(
        self, 
        rsi_period: int = 14, 
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
        oversold: float = 0.2,
        overbought: float = 0.8
    ):
        """
        Initialize Stochastic RSI.
        
        Args:
            rsi_period: Period for RSI calculation
            stoch_period: Period for Stochastic calculation
            k_smooth: Smoothing for K line
            d_smooth: Smoothing for D line (signal)
            oversold: Oversold threshold (default 0.2)
            overbought: Overbought threshold (default 0.8)
        """
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.oversold = oversold
        self.overbought = overbought
        
        # State
        self._rsi_history: deque = deque(maxlen=stoch_period + 10)
        self._k_history: deque = deque(maxlen=d_smooth + 1)
        self._prev_k: Optional[float] = None
        self._prev_d: Optional[float] = None
        
        logger.info(f"📊 Stochastic RSI initialized: RSI={rsi_period}, Stoch={stoch_period}, K={k_smooth}, D={d_smooth}")
    
    def calculate(self, candles: List[Dict]) -> Optional[StochRSIResult]:
        """
        Calculate Stochastic RSI.
        
        Args:
            candles: OHLCV candles
            
        Returns:
            StochRSIResult or None
        """
        min_candles = self.rsi_period + self.stoch_period + self.k_smooth
        if len(candles) < min_candles:
            return None
        
        # Calculate RSI values for stoch_period
        rsi_values = []
        for i in range(self.stoch_period + self.k_smooth):
            end_idx = len(candles) - i
            start_idx = end_idx - self.rsi_period - 1
            if start_idx < 0:
                continue
            
            rsi = self._calculate_rsi(candles[start_idx:end_idx])
            if rsi is not None:
                rsi_values.insert(0, rsi)
        
        if len(rsi_values) < self.stoch_period:
            return None
        
        # Calculate raw StochRSI values for smoothing
        stoch_rsi_values = []
        for i in range(len(rsi_values) - self.stoch_period + 1):
            window = rsi_values[i:i + self.stoch_period]
            highest = max(window)
            lowest = min(window)
            
            current_rsi = window[-1]
            if highest - lowest > 0:
                stoch_rsi = (current_rsi - lowest) / (highest - lowest)
            else:
                stoch_rsi = 0.5
            stoch_rsi_values.append(stoch_rsi)
        
        if len(stoch_rsi_values) < self.k_smooth:
            return None
        
        # K line = SMA of StochRSI
        k_line = sum(stoch_rsi_values[-self.k_smooth:]) / self.k_smooth
        
        # Store K for D calculation
        self._k_history.append(k_line)
        
        # D line = SMA of K
        if len(self._k_history) >= self.d_smooth:
            d_line = sum(list(self._k_history)[-self.d_smooth:]) / self.d_smooth
        else:
            d_line = k_line
        
        # Determine zone
        if k_line < self.oversold:
            zone = 'oversold'
            strength = (self.oversold - k_line) / self.oversold
        elif k_line > self.overbought:
            zone = 'overbought'
            strength = (k_line - self.overbought) / (1 - self.overbought)
        else:
            zone = 'neutral'
            strength = 0.0
        
        # Detect crossover
        crossover = None
        if self._prev_k is not None and self._prev_d is not None:
            if self._prev_k <= self._prev_d and k_line > d_line:
                crossover = 'bullish'
            elif self._prev_k >= self._prev_d and k_line < d_line:
                crossover = 'bearish'
        
        # Update state
        self._prev_k = k_line
        self._prev_d = d_line
        
        return StochRSIResult(
            stoch_rsi=stoch_rsi_values[-1],
            k_line=k_line,
            d_line=d_line,
            zone=zone,
            crossover=crossover,
            strength=min(1.0, strength)
        )
    
    def _calculate_rsi(self, candles: List[Dict]) -> Optional[float]:
        """Calculate RSI for given candles."""
        if len(candles) < 2:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(candles)):
            close = float(candles[i].get('close', candles[i].get('c', 0)))
            prev_close = float(candles[i-1].get('close', candles[i-1].get('c', 0)))
            
            change = close - prev_close
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < self.rsi_period:
            return None
        
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_signal(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Get trading signal from Stochastic RSI.
        
        Returns:
            Tuple of (direction, confidence)
        """
        result = self.calculate(candles)
        if result is None:
            return ('neutral', 0.0)
        
        # Strong signals from crossovers in extreme zones
        if result.zone == 'oversold' and result.crossover == 'bullish':
            return ('long', 0.85)
        elif result.zone == 'overbought' and result.crossover == 'bearish':
            return ('short', 0.85)
        
        # Zone-based signals
        if result.zone == 'oversold':
            return ('long', 0.5 + result.strength * 0.3)
        elif result.zone == 'overbought':
            return ('short', 0.5 + result.strength * 0.3)
        
        # Crossover signals in neutral zone
        if result.crossover == 'bullish':
            return ('long', 0.4)
        elif result.crossover == 'bearish':
            return ('short', 0.4)
        
        return ('neutral', 0.0)
    
    def reset(self):
        """Reset indicator state."""
        self._rsi_history.clear()
        self._k_history.clear()
        self._prev_k = None
        self._prev_d = None
