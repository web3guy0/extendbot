"""
Multi-Timeframe Analysis Module
Higher timeframe confirmation for lower timeframe entries.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classification."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Analysis (MTF)
    
    Core principle: Trade in direction of higher timeframe trend.
    - 1m entries aligned with 15m trend = higher win rate
    - 15m trend aligned with 1h trend = even better
    
    HTF provides: Direction and key levels
    LTF provides: Precise entries
    
    The goal: Only take 1m trades that align with 15m and 1h trends.
    """
    
    def __init__(self):
        """Initialize MTF analyzer."""
        # Timeframe configuration
        self.htf_intervals = os.getenv('HTF_INTERVALS', '15m,1h,4h').split(',')
        self.ltf_interval = os.getenv('LTF_INTERVAL', '1m')
        
        # Trend calculation settings
        self.ema_fast_period = int(os.getenv('MTF_EMA_FAST', '9'))
        self.ema_slow_period = int(os.getenv('MTF_EMA_SLOW', '21'))
        self.adx_threshold = float(os.getenv('MTF_ADX_THRESHOLD', '20'))
        
        # Cache for HTF data
        self.htf_cache: Dict[str, Dict] = {}
        self.last_htf_update: Dict[str, datetime] = {}
        
        # Alignment scoring
        self.alignment_weights = {
            '15m': 0.4,  # 40% weight
            '1h': 0.35,  # 35% weight
            '4h': 0.25,  # 25% weight
        }
        
        logger.info("📊 Multi-Timeframe Analyzer initialized")
        logger.info(f"   HTF Intervals: {self.htf_intervals}")
        logger.info(f"   LTF Interval: {self.ltf_interval}")
        logger.info(f"   EMA: {self.ema_fast_period}/{self.ema_slow_period}")
    
    def analyze_timeframe(
        self,
        candles: List[Dict],
        interval: str,
    ) -> Dict[str, Any]:
        """
        Analyze a single timeframe for trend and structure.
        
        Args:
            candles: Candle data for the timeframe
            interval: Timeframe interval (1m, 15m, 1h, etc)
        
        Returns:
            Analysis dict with trend, structure, key levels
        """
        if len(candles) < 50:
            return {
                'interval': interval,
                'trend': TrendDirection.NEUTRAL,
                'trend_strength': 0,
                'bias': None,
            }
        
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles]
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, self.ema_fast_period)
        ema_slow = self._calculate_ema(prices, self.ema_slow_period)
        
        # Calculate ADX for trend strength
        adx = self._calculate_adx(candles)
        
        # Determine trend direction
        current_price = prices[-1]
        trend = TrendDirection.NEUTRAL
        trend_strength = 0.0
        
        if ema_fast and ema_slow and adx:
            adx_val = float(adx)
            
            if ema_fast > ema_slow:
                if adx_val > self.adx_threshold * 1.5:
                    trend = TrendDirection.STRONG_UP
                    trend_strength = min(1.0, adx_val / 50)
                elif adx_val > self.adx_threshold:
                    trend = TrendDirection.UP
                    trend_strength = min(0.7, adx_val / 50)
                else:
                    trend = TrendDirection.NEUTRAL
                    trend_strength = 0.3
            else:
                if adx_val > self.adx_threshold * 1.5:
                    trend = TrendDirection.STRONG_DOWN
                    trend_strength = min(1.0, adx_val / 50)
                elif adx_val > self.adx_threshold:
                    trend = TrendDirection.DOWN
                    trend_strength = min(0.7, adx_val / 50)
                else:
                    trend = TrendDirection.NEUTRAL
                    trend_strength = 0.3
        
        # Calculate key levels
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in candles[-20:]]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in candles[-20:]]
        
        analysis = {
            'interval': interval,
            'trend': trend,
            'trend_strength': trend_strength,
            'ema_fast': float(ema_fast) if ema_fast else None,
            'ema_slow': float(ema_slow) if ema_slow else None,
            'adx': float(adx) if adx else None,
            'current_price': float(current_price),
            'swing_high': float(max(highs)),
            'swing_low': float(min(lows)),
            'bias': 'bullish' if trend in [TrendDirection.UP, TrendDirection.STRONG_UP] else
                    'bearish' if trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN] else 'neutral',
        }
        
        # Cache HTF analysis
        self.htf_cache[interval] = analysis
        self.last_htf_update[interval] = datetime.now(timezone.utc)
        
        return analysis
    
    def get_alignment_score(
        self,
        ltf_direction: str,  # 'long' or 'short'
        htf_analyses: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[float, str]:
        """
        Calculate alignment score between LTF signal and HTF trends.
        
        Args:
            ltf_direction: Direction of LTF signal ('long' or 'short')
            htf_analyses: Dict of HTF analyses by interval
        
        Returns:
            Tuple of (alignment_score 0-1, reason string)
        """
        if htf_analyses is None:
            htf_analyses = self.htf_cache
        
        if not htf_analyses:
            return 0.5, "No HTF data available"
        
        aligned_weight = 0.0
        total_weight = 0.0
        reasons = []
        
        for interval, analysis in htf_analyses.items():
            weight = self.alignment_weights.get(interval, 0.2)
            total_weight += weight
            
            trend = analysis.get('trend', TrendDirection.NEUTRAL)
            strength = analysis.get('trend_strength', 0)
            
            if ltf_direction == 'long':
                if trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    aligned_weight += weight * strength
                    reasons.append(f"{interval}:✅UP")
                elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    # Against HTF trend - reduce score
                    aligned_weight -= weight * strength * 0.5
                    reasons.append(f"{interval}:❌DOWN")
                else:
                    aligned_weight += weight * 0.3  # Neutral doesn't hurt much
                    reasons.append(f"{interval}:➖NEUTRAL")
            
            elif ltf_direction == 'short':
                if trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    aligned_weight += weight * strength
                    reasons.append(f"{interval}:✅DOWN")
                elif trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    aligned_weight -= weight * strength * 0.5
                    reasons.append(f"{interval}:❌UP")
                else:
                    aligned_weight += weight * 0.3
                    reasons.append(f"{interval}:➖NEUTRAL")
        
        # Normalize score to 0-1
        if total_weight > 0:
            score = max(0, min(1, (aligned_weight / total_weight + 1) / 2))
        else:
            score = 0.5
        
        return score, " | ".join(reasons)
    
    def should_take_trade(
        self,
        ltf_direction: str,
        min_alignment: float = 0.6,
    ) -> Tuple[bool, float, str]:
        """
        Determine if trade aligns with higher timeframes.
        
        Args:
            ltf_direction: 'long' or 'short'
            min_alignment: Minimum alignment score required (0-1)
        
        Returns:
            Tuple of (should_trade, alignment_score, reason)
        """
        score, reason = self.get_alignment_score(ltf_direction)
        should_trade = score >= min_alignment
        
        if should_trade:
            logger.info(f"✅ HTF Alignment OK for {ltf_direction}: {score:.1%} | {reason}")
        else:
            logger.debug(f"⚠️ HTF Alignment WEAK for {ltf_direction}: {score:.1%} | {reason}")
        
        return should_trade, score, reason
    
    def get_htf_bias(self) -> Optional[str]:
        """Get overall bias from all HTF analyses."""
        if not self.htf_cache:
            return None
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        for interval, analysis in self.htf_cache.items():
            weight = self.alignment_weights.get(interval, 0.2)
            trend = analysis.get('trend', TrendDirection.NEUTRAL)
            strength = analysis.get('trend_strength', 0)
            
            if trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                bullish_score += weight * strength
            elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                bearish_score += weight * strength
        
        if bullish_score > bearish_score * 1.3:
            return 'bullish'
        elif bearish_score > bullish_score * 1.3:
            return 'bearish'
        return 'neutral'
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate EMA."""
        if len(prices) < period:
            return None
        
        multiplier = Decimal('2') / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_adx(self, candles: List[Dict], period: int = 14) -> Optional[Decimal]:
        """Calculate ADX using proper True Range (H/L/C)."""
        if len(candles) < period * 2:
            return None
        
        # Proper True Range calculation using High/Low/Close
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            high = Decimal(str(candles[i].get('high', candles[i].get('h', 0))))
            low = Decimal(str(candles[i].get('low', candles[i].get('l', 0))))
            prev_high = Decimal(str(candles[i-1].get('high', candles[i-1].get('h', 0))))
            prev_low = Decimal(str(candles[i-1].get('low', candles[i-1].get('l', 0))))
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            
            # True Range = max(H-L, |H-prev_close|, |L-prev_close|)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low
            
            plus_dm = up_move if (up_move > down_move and up_move > 0) else Decimal('0')
            minus_dm = down_move if (down_move > up_move and down_move > 0) else Decimal('0')
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        if len(tr_list) < period:
            return None
        
        atr = sum(tr_list[-period:]) / period
        plus_dm_avg = sum(plus_dm_list[-period:]) / period
        minus_dm_avg = sum(minus_dm_list[-period:]) / period
        
        plus_di = (plus_dm_avg / atr * 100) if atr > 0 else Decimal('0')
        minus_di = (minus_dm_avg / atr * 100) if atr > 0 else Decimal('0')
        
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = (di_diff / di_sum * 100) if di_sum > 0 else Decimal('0')
        
        return dx
