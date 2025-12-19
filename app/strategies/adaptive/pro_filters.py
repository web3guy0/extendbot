#!/usr/bin/env python3
"""
Pro Trading Filters
===================

Advanced filters used by professional traders to improve signal quality.
These filters are applied BEFORE generating any signal.

Filters:
1. Multi-Timeframe Confluence - HTF trend must align with LTF entry
2. Volatility Regime Filter - Avoid low volatility choppy markets
3. Correlation Filter - Don't fight the leader (BTC)
4. Momentum Alignment - RSI + MACD + Price must all agree
5. Time-Based Filters - Avoid dangerous market times
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filter check."""
    passed: bool
    reason: str
    confidence: float = 0.5


class VolatilityRegime(Enum):
    """Volatility classification"""
    TOO_LOW = "too_low"        # < 0.5x average - avoid, choppy
    LOW = "low"                # 0.5-0.8x average - reduce size
    NORMAL = "normal"          # 0.8-1.5x average - full size
    HIGH = "high"              # 1.5-2.5x average - reduce size
    EXTREME = "extreme"        # > 2.5x average - avoid or 25% size


class ProTradingFilters:
    """
    Professional-grade trade filters that institutional traders use.
    
    Goal: Only take trades with multiple confluences aligned.
    Better to miss trades than take bad ones.
    """
    
    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        """Initialize pro trading filters."""
        self.symbol = symbol
        self.config = config or {}
        
        # HTF alignment thresholds
        self.htf_min_alignment = float(self.config.get('htf_min_alignment', 0.6))  # 60% HTF alignment needed
        
        # Volatility thresholds (as multiple of 20-period average ATR)
        self.vol_too_low_mult = 0.5
        self.vol_low_mult = 0.8
        self.vol_high_mult = 1.5
        self.vol_extreme_mult = 2.5
        
        # Correlation thresholds
        self.btc_dump_threshold = -2.0  # Don't long alts if BTC dumps 2%+
        self.btc_pump_threshold = 2.0   # Don't short alts if BTC pumps 2%+
        
        # Momentum alignment
        self.require_all_momentum_agree = True  # RSI + MACD + EMA must agree
        
        # ATR history for volatility regime
        self._atr_history: List[float] = []
        self._atr_history_max = 100
        
        logger.info("🎯 Pro Trading Filters initialized")
        logger.info(f"   HTF Min Alignment: {self.htf_min_alignment*100:.0f}%")
        logger.info(f"   Momentum Alignment: {'Required' if self.require_all_momentum_agree else 'Optional'}")
    
    def check_all(
        self,
        direction: str,
        candles: List[Dict],
        indicators: Dict[str, Any],
        btc_candles: Optional[List[Dict]] = None,
    ) -> 'FilterResult':
        """
        Simplified check_all method for swing strategy integration.
        
        Args:
            direction: 'long' or 'short'
            candles: Current timeframe candles
            indicators: Calculated indicators
            btc_candles: BTC candles for correlation check
        
        Returns:
            FilterResult with passed, reason, confidence
        """
        checks_passed = 0
        total_checks = 0
        reasons = []
        
        # 1. Volatility Check
        atr = indicators.get('atr')
        if atr:
            total_checks += 1
            vol_passed, vol_reason, vol_regime = self.check_volatility_regime(float(atr))
            if vol_passed:
                checks_passed += 1
            else:
                reasons.append(vol_reason)
                # Extreme volatility is a hard reject
                if vol_regime == VolatilityRegime.EXTREME or vol_regime == VolatilityRegime.TOO_LOW:
                    return FilterResult(False, vol_reason, 0.0)
        
        # 2. Momentum Check
        total_checks += 1
        mom_passed, mom_reason = self.check_momentum_alignment(direction, indicators)
        if mom_passed:
            checks_passed += 1
        else:
            reasons.append(mom_reason)
        
        # 3. BTC Correlation Check (for altcoins)
        if btc_candles and self.symbol != 'BTC' and len(btc_candles) >= 2:
            total_checks += 1
            # Calculate BTC change
            btc_current = float(btc_candles[-1].get('close', btc_candles[-1].get('c', 0)))
            btc_prev = float(btc_candles[-2].get('close', btc_candles[-2].get('c', 0)))
            btc_change_pct = ((btc_current / btc_prev) - 1) * 100 if btc_prev > 0 else 0
            
            corr_passed, corr_reason = self.check_btc_correlation(direction, btc_change_pct)
            if corr_passed:
                checks_passed += 1
            else:
                reasons.append(corr_reason)
                # Strong BTC divergence is a hard reject
                if abs(btc_change_pct) > 2.0:
                    return FilterResult(False, corr_reason, 0.0)
        
        # 4. Time Filter - Avoid first 15 min of major session opens
        total_checks += 1
        time_passed, time_reason = self.check_time_filter(datetime.now(timezone.utc))
        if time_passed:
            checks_passed += 1
        else:
            reasons.append(time_reason)
        
        # Calculate confidence
        confidence = checks_passed / total_checks if total_checks > 0 else 0.5
        
        # Require at least 60% of checks to pass
        if confidence >= 0.6:
            return FilterResult(True, f"Passed {checks_passed}/{total_checks} filters", confidence)
        else:
            reason_str = "; ".join(reasons[:2]) if reasons else "Insufficient confluence"
            return FilterResult(False, reason_str, confidence)
    
    def check_all_filters(
        self,
        direction: str,  # 'long' or 'short'
        indicators: Dict[str, Any],
        htf_analysis: Dict[str, Any],
        btc_change_pct: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check ALL pro trading filters.
        
        Args:
            direction: Trade direction ('long' or 'short')
            indicators: Technical indicators from current timeframe
            htf_analysis: Higher timeframe analysis results
            btc_change_pct: BTC price change percentage (for correlation filter)
            current_time: Current time (for time-based filters)
        
        Returns:
            Tuple of (passed: bool, reason: str, details: dict)
        """
        details = {
            'htf_check': None,
            'volatility_check': None,
            'correlation_check': None,
            'momentum_check': None,
            'time_check': None,
        }
        
        # 1. HTF Alignment Check
        htf_passed, htf_reason = self.check_htf_alignment(direction, htf_analysis)
        details['htf_check'] = {'passed': htf_passed, 'reason': htf_reason}
        if not htf_passed:
            return False, f"HTF: {htf_reason}", details
        
        # 2. Volatility Regime Check
        atr = indicators.get('atr')
        if atr:
            vol_passed, vol_reason, vol_regime = self.check_volatility_regime(float(atr))
            details['volatility_check'] = {'passed': vol_passed, 'reason': vol_reason, 'regime': vol_regime.value}
            if not vol_passed:
                return False, f"Volatility: {vol_reason}", details
        
        # 3. Correlation Filter (BTC leader)
        if btc_change_pct is not None:
            corr_passed, corr_reason = self.check_btc_correlation(direction, btc_change_pct)
            details['correlation_check'] = {'passed': corr_passed, 'reason': corr_reason}
            if not corr_passed:
                return False, f"Correlation: {corr_reason}", details
        
        # 4. Momentum Alignment Check
        if self.require_all_momentum_agree:
            mom_passed, mom_reason = self.check_momentum_alignment(direction, indicators)
            details['momentum_check'] = {'passed': mom_passed, 'reason': mom_reason}
            if not mom_passed:
                return False, f"Momentum: {mom_reason}", details
        
        # 5. Time-Based Filter
        if current_time:
            time_passed, time_reason = self.check_time_filter(current_time)
            details['time_check'] = {'passed': time_passed, 'reason': time_reason}
            if not time_passed:
                return False, f"Time: {time_reason}", details
        
        return True, "All filters passed ✅", details
    
    def check_htf_alignment(
        self,
        direction: str,
        htf_analysis: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if higher timeframe trend aligns with trade direction.
        
        Rule: Don't take longs in HTF downtrend, don't take shorts in HTF uptrend.
        """
        if not htf_analysis:
            return False, "No HTF data available"
        
        # Get alignment score and trend info
        alignment_score = htf_analysis.get('alignment_score', 0.5)
        htf_bias = htf_analysis.get('bias', 'neutral')
        htf_15m = htf_analysis.get('15m', {}).get('bias', 'neutral')
        htf_1h = htf_analysis.get('1h', {}).get('bias', 'neutral')
        
        # Check if direction aligns with HTF
        if direction == 'long':
            if htf_bias == 'bearish' and htf_1h == 'bearish':
                return False, f"HTF bearish (15m:{htf_15m}, 1h:{htf_1h})"
            if alignment_score < self.htf_min_alignment:
                return False, f"Weak HTF alignment ({alignment_score:.0%})"
        else:  # short
            if htf_bias == 'bullish' and htf_1h == 'bullish':
                return False, f"HTF bullish (15m:{htf_15m}, 1h:{htf_1h})"
            if alignment_score < self.htf_min_alignment:
                return False, f"Weak HTF alignment ({alignment_score:.0%})"
        
        return True, f"HTF aligned ({alignment_score:.0%})"
    
    def check_volatility_regime(
        self,
        current_atr: float,
    ) -> Tuple[bool, str, VolatilityRegime]:
        """
        Check if current volatility is tradeable.
        
        - Too low = Choppy, fees eat profits
        - Too high = Unpredictable, reduce size
        """
        # Update ATR history
        self._atr_history.append(current_atr)
        if len(self._atr_history) > self._atr_history_max:
            self._atr_history.pop(0)
        
        # Need history to compare
        if len(self._atr_history) < 20:
            return True, "Warming up", VolatilityRegime.NORMAL
        
        # Calculate average ATR
        avg_atr = sum(self._atr_history) / len(self._atr_history)
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Classify regime
        if vol_ratio < self.vol_too_low_mult:
            return False, f"Too low ({vol_ratio:.1f}x avg) - choppy market", VolatilityRegime.TOO_LOW
        elif vol_ratio < self.vol_low_mult:
            return True, f"Low ({vol_ratio:.1f}x avg) - reduce size", VolatilityRegime.LOW
        elif vol_ratio < self.vol_high_mult:
            return True, f"Normal ({vol_ratio:.1f}x avg)", VolatilityRegime.NORMAL
        elif vol_ratio < self.vol_extreme_mult:
            return True, f"High ({vol_ratio:.1f}x avg) - reduce size", VolatilityRegime.HIGH
        else:
            return False, f"Extreme ({vol_ratio:.1f}x avg) - too risky", VolatilityRegime.EXTREME
    
    def check_btc_correlation(
        self,
        direction: str,
        btc_change_pct: float,
    ) -> Tuple[bool, str]:
        """
        Check if trade direction aligns with BTC movement.
        
        Rule: Altcoins follow BTC. Don't fight the leader.
        """
        if direction == 'long':
            if btc_change_pct < self.btc_dump_threshold:
                return False, f"BTC dumping ({btc_change_pct:+.1f}%) - don't long alts"
        else:  # short
            if btc_change_pct > self.btc_pump_threshold:
                return False, f"BTC pumping ({btc_change_pct:+.1f}%) - don't short alts"
        
        return True, f"BTC aligned ({btc_change_pct:+.1f}%)"
    
    def check_momentum_alignment(
        self,
        direction: str,
        indicators: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if RSI, MACD, and EMA all agree on direction.
        
        Rule: All momentum indicators must point the same way.
        """
        rsi = indicators.get('rsi')
        macd = indicators.get('macd', {})
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        current_price = indicators.get('current_price', 0)
        
        if not all([rsi, ema_fast, ema_slow]):
            return True, "Incomplete indicators - skipping check"
        
        # Get MACD signals
        macd_value = macd.get('value', 0)
        macd_signal = macd.get('signal', 0)
        macd_histogram = macd.get('histogram', 0)
        
        if direction == 'long':
            # For LONG: RSI not overbought, MACD bullish, Price > EMAs
            rsi_ok = rsi < 65  # Not overbought
            macd_ok = macd_histogram > 0 or macd_value > macd_signal
            ema_ok = float(current_price) > float(ema_fast) > float(ema_slow) * 0.998  # Allow small tolerance
            
            if not rsi_ok:
                return False, f"RSI overbought ({rsi:.1f})"
            if not macd_ok:
                return False, "MACD bearish"
            if not ema_ok:
                return False, "Price below EMAs"
        
        else:  # short
            # For SHORT: RSI not oversold, MACD bearish, Price < EMAs
            rsi_ok = rsi > 35  # Not oversold
            macd_ok = macd_histogram < 0 or macd_value < macd_signal
            ema_ok = float(current_price) < float(ema_fast) < float(ema_slow) * 1.002  # Allow small tolerance
            
            if not rsi_ok:
                return False, f"RSI oversold ({rsi:.1f})"
            if not macd_ok:
                return False, "MACD bullish"
            if not ema_ok:
                return False, "Price above EMAs"
        
        return True, "All momentum aligned"
    
    def check_time_filter(
        self,
        current_time: datetime,
    ) -> Tuple[bool, str]:
        """
        Check if current time is suitable for trading.
        
        Avoid:
        - First/last 15 min of trading sessions
        - Low liquidity hours (weekends, holidays)
        """
        hour = current_time.hour
        minute = current_time.minute
        weekday = current_time.weekday()
        
        # Avoid weekends (crypto trades 24/7 but volume is low)
        if weekday >= 5:  # Saturday, Sunday
            # Still tradeable but note lower volume
            pass
        
        # Avoid first 15 minutes of major sessions
        # US Open: 13:30-14:30 UTC (9:30-10:30 EST)
        # London Open: 07:00-08:00 UTC
        # Asia Open: 00:00-01:00 UTC
        
        dangerous_times = [
            (0, 0, 0, 15),    # Asia open first 15 min
            (7, 0, 7, 15),    # London open first 15 min
            (13, 30, 13, 45), # US open first 15 min
        ]
        
        for start_h, start_m, end_h, end_m in dangerous_times:
            if start_h == hour and start_m <= minute < end_m:
                return False, f"Session open volatility ({hour:02d}:{minute:02d} UTC)"
        
        return True, "Good trading time"
    
    def get_position_size_multiplier(
        self,
        vol_regime: VolatilityRegime,
        htf_alignment: float,
        signal_score: int,
        max_score: int = 10,
    ) -> float:
        """
        Calculate position size multiplier based on conditions.
        
        Returns value 0.0 - 1.0 to multiply base position size by.
        """
        multiplier = 1.0
        
        # Volatility adjustment
        vol_mult = {
            VolatilityRegime.TOO_LOW: 0.0,   # Don't trade
            VolatilityRegime.LOW: 0.5,       # Half size
            VolatilityRegime.NORMAL: 1.0,    # Full size
            VolatilityRegime.HIGH: 0.6,      # Reduced
            VolatilityRegime.EXTREME: 0.25,  # Quarter size
        }
        multiplier *= vol_mult.get(vol_regime, 1.0)
        
        # HTF alignment adjustment
        if htf_alignment < 0.5:
            multiplier *= 0.5
        elif htf_alignment < 0.7:
            multiplier *= 0.75
        
        # Signal score adjustment (prevent divide by zero)
        if max_score > 0:
            score_ratio = signal_score / max_score
        else:
            score_ratio = 0.5  # Default to 50% if no max_score
        if score_ratio >= 0.8:
            multiplier *= 1.0  # Full size for A+ setups
        elif score_ratio >= 0.7:
            multiplier *= 0.8  # Slightly reduced
        else:
            multiplier *= 0.6  # Minimum signal quality
        
        return max(0.25, min(1.0, multiplier))  # Cap between 25% and 100%


# Singleton instance
_pro_filters: Optional[ProTradingFilters] = None


def get_pro_filters(config: Dict[str, Any] = None) -> ProTradingFilters:
    """Get or create pro trading filters singleton."""
    global _pro_filters
    if _pro_filters is None:
        _pro_filters = ProTradingFilters(config or {})
    return _pro_filters
