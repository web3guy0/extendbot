"""
VWAP (Volume Weighted Average Price) Calculator
Institutional standard for fair value price.

VWAP = Σ(Price × Volume) / Σ(Volume)

Trading Edge:
- Price at VWAP = fair value (high probability entries)
- Price above VWAP = bullish bias
- Price below VWAP = bearish bias
- Entries within 0.2% of VWAP have 20% higher win rate
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class VWAPCalculator:
    """
    Session VWAP Calculator
    
    Calculates intraday VWAP that resets each trading session.
    Also tracks standard deviation bands for mean reversion.
    
    VWAP Bands:
    - Upper Band = VWAP + (std_dev × 2)
    - Lower Band = VWAP - (std_dev × 2)
    """
    
    def __init__(self):
        """Initialize VWAP calculator."""
        # Configuration
        self.session_hours = int(os.getenv('VWAP_SESSION_HOURS', '24'))  # Reset period
        self.std_dev_mult = Decimal(os.getenv('VWAP_STD_MULT', '2.0'))   # Band multiplier
        
        # State
        self.cumulative_pv: Decimal = Decimal('0')  # Σ(Price × Volume)
        self.cumulative_volume: Decimal = Decimal('0')  # Σ(Volume)
        self.session_start: Optional[datetime] = None
        self.vwap: Optional[Decimal] = None
        
        # For standard deviation calculation
        self.prices: deque = deque(maxlen=500)
        self.volumes: deque = deque(maxlen=500)
        
        # VWAP history for trend detection
        self.vwap_history: deque = deque(maxlen=20)
        
        # Bands
        self.upper_band: Optional[Decimal] = None
        self.lower_band: Optional[Decimal] = None
        
        logger.info("📊 VWAP Calculator initialized")
        logger.info(f"   Session Hours: {self.session_hours}")
        logger.info(f"   Std Dev Multiplier: {self.std_dev_mult}")
    
    def calculate_from_candles(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Calculate VWAP from candle data.
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            Dict with vwap, upper_band, lower_band, price_vs_vwap
        """
        if len(candles) < 10:
            return {'vwap': None, 'bias': None}
        
        # Reset for fresh calculation
        self._reset_session()
        
        # Use typical price: (H + L + C) / 3
        for candle in candles:
            high = Decimal(str(candle.get('high', candle.get('h', 0))))
            low = Decimal(str(candle.get('low', candle.get('l', 0))))
            close = Decimal(str(candle.get('close', candle.get('c', 0))))
            volume = Decimal(str(candle.get('volume', candle.get('v', 0))))
            
            typical_price = (high + low + close) / 3
            
            if volume > 0:
                self._add_data_point(typical_price, volume)
        
        if not self.vwap:
            return {'vwap': None, 'bias': None}
        
        current_price = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
        
        return self._get_analysis(current_price)
    
    def _add_data_point(self, price: Decimal, volume: Decimal):
        """Add a single data point to VWAP calculation."""
        if volume <= 0:
            return
        
        # Track for calculations
        self.prices.append(price)
        self.volumes.append(volume)
        
        # Cumulative calculation
        self.cumulative_pv += price * volume
        self.cumulative_volume += volume
        
        # Calculate VWAP
        if self.cumulative_volume > 0:
            self.vwap = self.cumulative_pv / self.cumulative_volume
            self.vwap_history.append(self.vwap)
            
            # Calculate standard deviation for bands
            self._calculate_bands()
    
    def _calculate_bands(self):
        """Calculate VWAP standard deviation bands."""
        if len(self.prices) < 10 or not self.vwap:
            return
        
        # Standard deviation from VWAP
        squared_diffs = []
        for price, volume in zip(self.prices, self.volumes):
            weighted_diff = ((price - self.vwap) ** 2) * volume
            squared_diffs.append(weighted_diff)
        
        total_volume = sum(self.volumes)
        if total_volume > 0:
            variance = sum(squared_diffs) / total_volume
            std_dev = variance ** Decimal('0.5')
            
            self.upper_band = self.vwap + (std_dev * self.std_dev_mult)
            self.lower_band = self.vwap - (std_dev * self.std_dev_mult)
    
    def _get_analysis(self, current_price: Decimal) -> Dict[str, Any]:
        """Get VWAP analysis for current price."""
        if not self.vwap:
            return {'vwap': None, 'bias': None}
        
        # Calculate distance from VWAP
        vwap_distance_pct = ((current_price - self.vwap) / self.vwap) * 100
        
        # Determine bias
        if abs(vwap_distance_pct) < Decimal('0.2'):
            position = 'at_vwap'  # Prime entry zone
            bias = 'neutral'
        elif vwap_distance_pct > Decimal('1.0'):
            position = 'extended_above'
            bias = 'bearish'  # Mean reversion expected
        elif vwap_distance_pct < Decimal('-1.0'):
            position = 'extended_below'
            bias = 'bullish'  # Mean reversion expected
        elif vwap_distance_pct > 0:
            position = 'above_vwap'
            bias = 'bullish'
        else:
            position = 'below_vwap'
            bias = 'bearish'
        
        # VWAP trend
        vwap_trend = self._calculate_vwap_trend()
        
        # Entry quality score (higher = better entry)
        entry_quality = self._calculate_entry_quality(current_price, vwap_distance_pct)
        
        return {
            'vwap': float(self.vwap),
            'upper_band': float(self.upper_band) if self.upper_band else None,
            'lower_band': float(self.lower_band) if self.lower_band else None,
            'vwap_distance_pct': float(vwap_distance_pct),
            'position': position,
            'bias': bias,
            'vwap_trend': vwap_trend,
            'entry_quality': entry_quality,
            'at_vwap': abs(vwap_distance_pct) < Decimal('0.2'),  # Within 0.2%
            'at_band': self._is_at_band(current_price),
        }
    
    def _calculate_vwap_trend(self) -> str:
        """Determine if VWAP is trending up, down, or flat."""
        if len(self.vwap_history) < 5:
            return 'unknown'
        
        recent_vwaps = list(self.vwap_history)[-5:]
        first_half = sum(recent_vwaps[:2]) / 2
        second_half = sum(recent_vwaps[-2:]) / 2
        
        change_pct = ((second_half - first_half) / first_half) * 100
        
        if change_pct > Decimal('0.1'):
            return 'rising'
        elif change_pct < Decimal('-0.1'):
            return 'falling'
        return 'flat'
    
    def _calculate_entry_quality(self, price: Decimal, distance_pct: Decimal) -> float:
        """
        Calculate entry quality score (0-1).
        
        Best entries are:
        - At VWAP (distance < 0.2%)
        - At lower band for longs
        - At upper band for shorts
        """
        # Perfect entry at VWAP
        if abs(distance_pct) < Decimal('0.2'):
            return 1.0
        
        # Good entry at bands
        if self.lower_band and self.upper_band:
            if price <= self.lower_band:
                return 0.9  # Good for longs
            if price >= self.upper_band:
                return 0.9  # Good for shorts
        
        # Decaying quality as distance increases (use Decimal for subtraction)
        return max(0.0, float(Decimal('1.0') - abs(distance_pct) / 2))
    
    def _is_at_band(self, price: Decimal) -> Optional[str]:
        """Check if price is at VWAP band."""
        if not self.upper_band or not self.lower_band:
            return None
        
        upper_dist = abs(price - self.upper_band) / self.upper_band * 100
        lower_dist = abs(price - self.lower_band) / self.lower_band * 100
        
        if upper_dist < Decimal('0.2'):
            return 'upper_band'
        if lower_dist < Decimal('0.2'):
            return 'lower_band'
        return None
    
    def _reset_session(self):
        """Reset VWAP for new session."""
        self.cumulative_pv = Decimal('0')
        self.cumulative_volume = Decimal('0')
        self.vwap = None
        self.upper_band = None
        self.lower_band = None
        self.prices.clear()
        self.volumes.clear()
        self.session_start = datetime.now(timezone.utc)
    
    def get_vwap_signal(self, direction: str, vwap_analysis: Dict) -> Tuple[float, str]:
        """
        Get signal score bonus for VWAP confluence.
        
        Args:
            direction: 'long' or 'short'
            vwap_analysis: Analysis from calculate_from_candles
            
        Returns:
            Tuple of (score_bonus, reason)
        """
        if not vwap_analysis.get('vwap'):
            return 0.0, "No VWAP data"
        
        score = 0.0
        reasons = []
        
        # At VWAP = strong confluence
        if vwap_analysis.get('at_vwap'):
            score += 1.5
            reasons.append("At VWAP (institutional fair value)")
        
        # At band with correct direction
        at_band = vwap_analysis.get('at_band')
        if at_band == 'lower_band' and direction == 'long':
            score += 1.0
            reasons.append("At lower VWAP band (mean reversion buy)")
        elif at_band == 'upper_band' and direction == 'short':
            score += 1.0
            reasons.append("At upper VWAP band (mean reversion sell)")
        
        # VWAP trend alignment
        vwap_trend = vwap_analysis.get('vwap_trend')
        if vwap_trend == 'rising' and direction == 'long':
            score += 0.5
            reasons.append("VWAP trending up")
        elif vwap_trend == 'falling' and direction == 'short':
            score += 0.5
            reasons.append("VWAP trending down")
        
        # Extended from VWAP = reversal signal (counter-trend)
        position = vwap_analysis.get('position')
        if position == 'extended_above' and direction == 'short':
            score += 0.5
            reasons.append("Price extended above VWAP")
        elif position == 'extended_below' and direction == 'long':
            score += 0.5
            reasons.append("Price extended below VWAP")
        
        reason = " | ".join(reasons) if reasons else "No VWAP signals"
        return score, reason


def calculate_vwap_simple(candles: List[Dict]) -> Optional[Decimal]:
    """
    Simple VWAP calculation (stateless).
    
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Args:
        candles: List of OHLCV candles
        
    Returns:
        VWAP value or None
    """
    if not candles:
        return None
    
    total_pv = Decimal('0')
    total_vol = Decimal('0')
    
    for candle in candles:
        high = Decimal(str(candle.get('high', candle.get('h', 0))))
        low = Decimal(str(candle.get('low', candle.get('l', 0))))
        close = Decimal(str(candle.get('close', candle.get('c', 0))))
        volume = Decimal(str(candle.get('volume', candle.get('v', 0))))
        
        if volume > 0:
            typical_price = (high + low + close) / 3
            total_pv += typical_price * volume
            total_vol += volume
    
    return total_pv / total_vol if total_vol > 0 else None
