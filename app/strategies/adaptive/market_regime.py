"""
Market Regime Detector - ML-Lite Classification
Detects: TRENDING, RANGING, VOLATILE, BREAKOUT regimes
Auto-adjusts strategy parameters based on regime.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    ML-Lite Market Regime Classifier
    
    Uses statistical features to classify market state:
    - ADX for trend strength
    - ATR ratio for volatility
    - Bollinger Band width for ranging
    - Volume profile for breakout detection
    
    Auto-adjusts strategy parameters:
    - TRENDING: Wider TP, tighter SL, follow momentum
    - RANGING: Mean reversion, tighter bands
    - VOLATILE: Wider stops, smaller positions
    - BREAKOUT: Aggressive entries, trailing stops
    """
    
    def __init__(self):
        """Initialize regime detector with dynamic config."""
        # Thresholds from env or defaults
        self.adx_trending_threshold = float(os.getenv('ADX_TRENDING_THRESHOLD', '25'))
        self.adx_strong_threshold = float(os.getenv('ADX_STRONG_THRESHOLD', '35'))
        self.volatility_high_ratio = float(os.getenv('VOLATILITY_HIGH_RATIO', '1.5'))
        self.bb_squeeze_threshold = float(os.getenv('BB_SQUEEZE_THRESHOLD', '2.0'))
        
        # State tracking
        self.regime_history: deque = deque(maxlen=20)
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_start_time: Optional[datetime] = None
        
        # ATR baseline for volatility comparison
        self._atr_baseline: Optional[Decimal] = None
        self._atr_history: deque = deque(maxlen=100)
        
        # Strategy parameter adjustments per regime
        self.regime_params = {
            MarketRegime.TRENDING_UP: {
                'tp_multiplier': 1.5,      # Wider TP in trends
                'sl_multiplier': 0.8,      # Tighter SL 
                'position_size_mult': 1.2, # Slightly larger positions
                'signal_threshold': 4,     # More aggressive entry
                'prefer_direction': 'long',
            },
            MarketRegime.TRENDING_DOWN: {
                'tp_multiplier': 1.5,
                'sl_multiplier': 0.8,
                'position_size_mult': 1.2,
                'signal_threshold': 4,
                'prefer_direction': 'short',
            },
            MarketRegime.RANGING: {
                'tp_multiplier': 0.7,      # Tighter TP in ranges
                'sl_multiplier': 0.6,      # Tighter SL
                'position_size_mult': 0.8, # Smaller positions
                'signal_threshold': 5,     # Need more confirmation
                'prefer_direction': 'mean_reversion',
            },
            MarketRegime.VOLATILE: {
                'tp_multiplier': 2.0,      # Wider TP for big moves
                'sl_multiplier': 1.5,      # Wider SL to avoid noise
                'position_size_mult': 0.5, # Smaller positions (higher risk)
                'signal_threshold': 6,     # Need strong confirmation
                'prefer_direction': None,
            },
            MarketRegime.BREAKOUT: {
                'tp_multiplier': 2.0,      # Let winners run
                'sl_multiplier': 0.5,      # Tight SL on breakouts
                'position_size_mult': 1.0,
                'signal_threshold': 4,     # Quick entry
                'prefer_direction': 'breakout',
            },
            MarketRegime.UNKNOWN: {
                'tp_multiplier': 1.0,
                'sl_multiplier': 1.0,
                'position_size_mult': 0.7, # Conservative when unsure
                'signal_threshold': 5,
                'prefer_direction': None,
            },
        }
        
        logger.info("🧠 Market Regime Detector initialized")
        logger.info(f"   ADX Trending: >{self.adx_trending_threshold}")
        logger.info(f"   ADX Strong: >{self.adx_strong_threshold}")
        logger.info(f"   Volatility High: >{self.volatility_high_ratio}x avg")
    
    def detect_regime(
        self,
        candles: List[Dict],
        adx: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        bb_bandwidth: Optional[Decimal] = None,
        volume_ratio: Optional[Decimal] = None,
        ema_fast: Optional[Decimal] = None,
        ema_slow: Optional[Decimal] = None,
    ) -> Tuple[MarketRegime, float, Dict[str, Any]]:
        """
        Detect current market regime.
        
        Args:
            candles: Recent candle data
            adx: Average Directional Index value
            atr: Average True Range
            bb_bandwidth: Bollinger Band bandwidth %
            volume_ratio: Current volume / average volume
            ema_fast: Fast EMA value
            ema_slow: Slow EMA value
        
        Returns:
            Tuple of (regime, confidence, params_adjustment)
        """
        if not candles or len(candles) < 20:
            return MarketRegime.UNKNOWN, 0.0, self.regime_params[MarketRegime.UNKNOWN]
        
        # Calculate features if not provided
        if adx is None:
            adx = self._calculate_adx(candles)
        if atr is None:
            atr = self._calculate_atr(candles)
        if bb_bandwidth is None:
            bb_bandwidth = self._calculate_bb_bandwidth(candles)
        
        # Update ATR baseline
        if atr:
            self._atr_history.append(float(atr))
            if len(self._atr_history) >= 50:
                self._atr_baseline = Decimal(str(sum(self._atr_history) / len(self._atr_history)))
        
        # Classification logic
        regime = MarketRegime.UNKNOWN
        confidence = 0.5
        
        adx_val = float(adx) if adx else 0
        bb_val = float(bb_bandwidth) if bb_bandwidth else 0
        vol_ratio = float(volume_ratio) if volume_ratio else 1.0
        
        # Check for VOLATILE regime first (takes priority)
        if self._atr_baseline and atr:
            atr_ratio = float(atr) / float(self._atr_baseline)
            if atr_ratio > self.volatility_high_ratio:
                regime = MarketRegime.VOLATILE
                confidence = min(0.9, 0.5 + (atr_ratio - self.volatility_high_ratio) * 0.2)
        
        # Check for TRENDING regime
        if regime == MarketRegime.UNKNOWN and adx_val > self.adx_trending_threshold:
            # Determine trend direction with HYSTERESIS to prevent flip-flopping
            if ema_fast and ema_slow:
                ema_diff_pct = (float(ema_fast) - float(ema_slow)) / float(ema_slow) * 100 if ema_slow else 0
                
                # Require 0.1% difference to confirm trend direction (hysteresis)
                # This prevents flip-flopping when EMAs are very close
                if ema_diff_pct > 0.1:  # Clear uptrend
                    regime = MarketRegime.TRENDING_UP
                elif ema_diff_pct < -0.1:  # Clear downtrend
                    regime = MarketRegime.TRENDING_DOWN
                else:
                    # EMAs too close - stay in previous direction or use price action
                    if self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                        regime = self.current_regime  # Keep current direction
                    else:
                        # Use price action as tiebreaker
                        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles[-20:]]
                        if prices[-1] > prices[0]:
                            regime = MarketRegime.TRENDING_UP
                        else:
                            regime = MarketRegime.TRENDING_DOWN
            else:
                # Use price action
                prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles[-20:]]
                if prices[-1] > prices[0]:
                    regime = MarketRegime.TRENDING_UP
                else:
                    regime = MarketRegime.TRENDING_DOWN
            
            confidence = min(0.95, 0.5 + (adx_val - self.adx_trending_threshold) / 50)
        
        # Check for BREAKOUT regime
        if regime == MarketRegime.UNKNOWN and bb_val < self.bb_squeeze_threshold and vol_ratio > 1.5:
            regime = MarketRegime.BREAKOUT
            confidence = min(0.85, 0.5 + vol_ratio * 0.1)
        
        # Check for RANGING regime
        if regime == MarketRegime.UNKNOWN:
            if adx_val < self.adx_trending_threshold * 0.8 and bb_val < 4.0:
                regime = MarketRegime.RANGING
                confidence = min(0.8, 0.6 + (self.adx_trending_threshold - adx_val) / 50)
        
        # FALLBACK: If still UNKNOWN, default to RANGING with lower confidence
        # This prevents the bot from being stuck in UNKNOWN which blocks all trading
        if regime == MarketRegime.UNKNOWN:
            # Use a sensible fallback based on available data
            if adx_val > 0:
                if adx_val > self.adx_trending_threshold * 0.7:
                    # Borderline trending - pick direction based on EMAs/price
                    if ema_fast and ema_slow:
                        if float(ema_fast) > float(ema_slow):
                            regime = MarketRegime.TRENDING_UP
                        else:
                            regime = MarketRegime.TRENDING_DOWN
                    else:
                        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles[-20:]]
                        if prices[-1] > prices[0]:
                            regime = MarketRegime.TRENDING_UP
                        else:
                            regime = MarketRegime.TRENDING_DOWN
                    confidence = 0.4  # Lower confidence for borderline
                else:
                    # Default to ranging in uncertain conditions
                    regime = MarketRegime.RANGING
                    confidence = 0.4
            else:
                # No ADX data - default to ranging
                regime = MarketRegime.RANGING
                confidence = 0.3
        
        # Update state with CONFIRMATION to prevent rapid flip-flopping
        self.regime_history.append(regime)
        
        # Only change regime if:
        # 1. It's a different regime, AND
        # 2. We've seen the same new regime at least 2 times in last 3 checks (confirmation)
        # EXCEPTION: First regime detection (when current is UNKNOWN) doesn't need confirmation
        if regime != self.current_regime:
            # First detection - accept immediately if not UNKNOWN
            if self.current_regime == MarketRegime.UNKNOWN and regime != MarketRegime.UNKNOWN:
                old_regime = self.current_regime
                self.current_regime = regime
                self.regime_start_time = datetime.now(timezone.utc)
                logger.info(f"📊 Initial regime detected: {regime.value} (confidence: {confidence:.1%})")
            else:
                # Subsequent changes need confirmation
                recent = list(self.regime_history)[-3:] if len(self.regime_history) >= 3 else list(self.regime_history)
                regime_count = sum(1 for r in recent if r == regime)
                
                if regime_count >= 2:  # Confirmed - regime seen at least 2x in last 3 checks
                    old_regime = self.current_regime
                    self.current_regime = regime
                    self.regime_start_time = datetime.now(timezone.utc)
                    logger.info(f"🔄 Regime changed: {old_regime.value} → {regime.value} (confidence: {confidence:.1%})")
                else:
                    # Not confirmed yet - keep current regime but log at debug level
                    logger.debug(f"📊 Regime candidate: {regime.value} (awaiting confirmation, count={regime_count}/2)")
                    regime = self.current_regime  # Return current regime, not the candidate
        
        self.regime_confidence = confidence
        
        return regime, confidence, self.regime_params.get(regime, self.regime_params[MarketRegime.UNKNOWN])
    
    def get_regime_duration(self) -> float:
        """Get how long current regime has been active (seconds)."""
        if self.regime_start_time:
            return (datetime.now(timezone.utc) - self.regime_start_time).total_seconds()
        return 0.0
    
    def is_regime_stable(self, min_confirmations: int = 3) -> bool:
        """Check if regime has been consistent (stable for trading)."""
        if len(self.regime_history) < min_confirmations:
            return False
        recent = list(self.regime_history)[-min_confirmations:]
        return len(set(recent)) == 1
    
    def _calculate_adx(self, candles: List[Dict], period: int = 14) -> Optional[Decimal]:
        """Calculate ADX from candles using proper True Range (H/L/C)."""
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
    
    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> Optional[Decimal]:
        """Calculate ATR from candles."""
        if len(candles) < period:
            return None
        
        tr_list = []
        for i in range(1, len(candles)):
            high = Decimal(str(candles[i].get('high', candles[i].get('h', 0))))
            low = Decimal(str(candles[i].get('low', candles[i].get('l', 0))))
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        return sum(tr_list[-period:]) / period if tr_list else None
    
    def _calculate_bb_bandwidth(self, candles: List[Dict], period: int = 20) -> Optional[Decimal]:
        """Calculate Bollinger Band bandwidth percentage."""
        if len(candles) < period:
            return None
        
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles[-period:]]
        sma = sum(prices) / period
        
        variance = sum((p - sma) ** 2 for p in prices) / period
        std = variance ** Decimal('0.5')
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        bandwidth = ((upper - lower) / sma) * 100
        return bandwidth
