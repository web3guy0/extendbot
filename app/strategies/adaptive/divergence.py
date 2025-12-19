"""
Divergence Detection Module
Detects RSI/MACD divergences that predict reversals.

PRO TRADER SECRET: Divergence precedes price by 1-3 candles.

Types:
- Regular Bullish: Lower price lows, higher indicator lows (reversal up)
- Regular Bearish: Higher price highs, lower indicator highs (reversal down)
- Hidden Bullish: Higher price lows, lower indicator lows (continuation up)
- Hidden Bearish: Lower price highs, higher indicator highs (continuation down)
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    """Types of divergence."""
    REGULAR_BULLISH = "regular_bullish"     # Reversal up
    REGULAR_BEARISH = "regular_bearish"     # Reversal down
    HIDDEN_BULLISH = "hidden_bullish"       # Continuation up
    HIDDEN_BEARISH = "hidden_bearish"       # Continuation down


@dataclass
class Divergence:
    """Represents a detected divergence."""
    divergence_type: DivergenceType
    indicator: str  # 'rsi' or 'macd'
    price_point1: float
    price_point2: float
    indicator_point1: float
    indicator_point2: float
    bar_distance: int
    strength: float  # 0-1
    detected_at: datetime


class DivergenceDetector:
    """
    RSI/MACD Divergence Detection
    
    Identifies divergences between price and momentum indicators.
    Divergences are leading indicators that signal reversals.
    
    Signal Strength:
    - RSI divergence: Strong (reliable)
    - MACD divergence: Very strong (institutional)
    - Combined: Extremely strong (A+ setup)
    """
    
    def __init__(self):
        """Initialize divergence detector."""
        # Configuration
        self.lookback = int(os.getenv('DIVERGENCE_LOOKBACK', '20'))
        self.min_bar_distance = int(os.getenv('DIVERGENCE_MIN_BARS', '5'))
        self.max_bar_distance = int(os.getenv('DIVERGENCE_MAX_BARS', '20'))
        
        # RSI thresholds
        self.rsi_oversold = Decimal(os.getenv('RSI_OVERSOLD', '30'))
        self.rsi_overbought = Decimal(os.getenv('RSI_OVERBOUGHT', '70'))
        
        # Recent divergences
        self.recent_divergences: List[Divergence] = []
        
        logger.info("📈 Divergence Detector initialized")
        logger.info(f"   Lookback: {self.lookback} bars")
        logger.info(f"   Bar Distance: {self.min_bar_distance}-{self.max_bar_distance}")
    
    def detect_all(
        self,
        candles: List[Dict],
        rsi_values: List[Decimal],
        macd_values: List[Dict],
    ) -> Dict[str, Any]:
        """
        Detect all divergences in current data.
        
        Args:
            candles: OHLCV candles
            rsi_values: Historical RSI values
            macd_values: Historical MACD dicts with 'macd', 'signal', 'histogram'
            
        Returns:
            Dict with divergences and signals
        """
        self.recent_divergences = []
        
        if len(candles) < self.lookback:
            return {'divergences': [], 'signal': None}
        
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles]
        
        # Detect RSI divergences
        if len(rsi_values) >= self.lookback:
            rsi_divs = self._detect_divergences(prices, rsi_values, 'rsi')
            self.recent_divergences.extend(rsi_divs)
        
        # Detect MACD divergences (using MACD line, not histogram)
        if len(macd_values) >= self.lookback:
            macd_line = [Decimal(str(m.get('macd', 0))) for m in macd_values]
            macd_divs = self._detect_divergences(prices, macd_line, 'macd')
            self.recent_divergences.extend(macd_divs)
        
        # Detect MACD histogram divergences (often earlier signal)
        if len(macd_values) >= self.lookback:
            histogram = [Decimal(str(m.get('histogram', 0))) for m in macd_values]
            hist_divs = self._detect_divergences(prices, histogram, 'macd_histogram')
            self.recent_divergences.extend(hist_divs)
        
        # Generate combined signal
        signal = self._generate_signal()
        
        return {
            'divergences': [self._div_to_dict(d) for d in self.recent_divergences],
            'signal': signal,
            'rsi_divergence': self._get_latest_by_indicator('rsi'),
            'macd_divergence': self._get_latest_by_indicator('macd'),
        }
    
    def _detect_divergences(
        self,
        prices: List[Decimal],
        indicator: List[Decimal],
        indicator_name: str,
    ) -> List[Divergence]:
        """Detect divergences between price and indicator."""
        divergences = []
        
        # Need enough data
        if len(prices) < self.lookback or len(indicator) < self.lookback:
            return divergences
        
        # Use last N bars
        prices = prices[-self.lookback:]
        indicator = indicator[-self.lookback:]
        
        # Find swing highs and lows in price
        price_highs = self._find_swing_highs(prices)
        price_lows = self._find_swing_lows(prices)
        
        # Find swing highs and lows in indicator
        ind_highs = self._find_swing_highs(indicator)
        ind_lows = self._find_swing_lows(indicator)
        
        # Check for REGULAR BEARISH divergence (Higher price highs, lower indicator highs)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            ph1, pi1 = price_highs[-2]  # Earlier high
            ph2, pi2 = price_highs[-1]  # Recent high
            ih1 = self._get_indicator_at_bar(indicator, ind_highs, pi1)
            ih2 = self._get_indicator_at_bar(indicator, ind_highs, pi2)
            
            if ih1 is not None and ih2 is not None:
                bar_dist = pi2 - pi1
                if self.min_bar_distance <= bar_dist <= self.max_bar_distance:
                    # Price makes higher high, indicator makes lower high
                    if ph2 > ph1 and ih2 < ih1:
                        strength = self._calculate_strength(ph1, ph2, ih1, ih2)
                        divergences.append(Divergence(
                            divergence_type=DivergenceType.REGULAR_BEARISH,
                            indicator=indicator_name,
                            price_point1=float(ph1),
                            price_point2=float(ph2),
                            indicator_point1=float(ih1),
                            indicator_point2=float(ih2),
                            bar_distance=bar_dist,
                            strength=strength,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        # Check for REGULAR BULLISH divergence (Lower price lows, higher indicator lows)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            pl1, pi1 = price_lows[-2]  # Earlier low
            pl2, pi2 = price_lows[-1]  # Recent low
            il1 = self._get_indicator_at_bar(indicator, ind_lows, pi1)
            il2 = self._get_indicator_at_bar(indicator, ind_lows, pi2)
            
            if il1 is not None and il2 is not None:
                bar_dist = pi2 - pi1
                if self.min_bar_distance <= bar_dist <= self.max_bar_distance:
                    # Price makes lower low, indicator makes higher low
                    if pl2 < pl1 and il2 > il1:
                        strength = self._calculate_strength(pl1, pl2, il1, il2)
                        divergences.append(Divergence(
                            divergence_type=DivergenceType.REGULAR_BULLISH,
                            indicator=indicator_name,
                            price_point1=float(pl1),
                            price_point2=float(pl2),
                            indicator_point1=float(il1),
                            indicator_point2=float(il2),
                            bar_distance=bar_dist,
                            strength=strength,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        # Check for HIDDEN BULLISH divergence (Higher price lows, lower indicator lows)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            pl1, pi1 = price_lows[-2]
            pl2, pi2 = price_lows[-1]
            il1 = self._get_indicator_at_bar(indicator, ind_lows, pi1)
            il2 = self._get_indicator_at_bar(indicator, ind_lows, pi2)
            
            if il1 is not None and il2 is not None:
                bar_dist = pi2 - pi1
                if self.min_bar_distance <= bar_dist <= self.max_bar_distance:
                    # Price makes higher low, indicator makes lower low (continuation)
                    if pl2 > pl1 and il2 < il1:
                        strength = self._calculate_strength(pl1, pl2, il1, il2) * 0.8
                        divergences.append(Divergence(
                            divergence_type=DivergenceType.HIDDEN_BULLISH,
                            indicator=indicator_name,
                            price_point1=float(pl1),
                            price_point2=float(pl2),
                            indicator_point1=float(il1),
                            indicator_point2=float(il2),
                            bar_distance=bar_dist,
                            strength=strength,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        # Check for HIDDEN BEARISH divergence (Lower price highs, higher indicator highs)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            ph1, pi1 = price_highs[-2]
            ph2, pi2 = price_highs[-1]
            ih1 = self._get_indicator_at_bar(indicator, ind_highs, pi1)
            ih2 = self._get_indicator_at_bar(indicator, ind_highs, pi2)
            
            if ih1 is not None and ih2 is not None:
                bar_dist = pi2 - pi1
                if self.min_bar_distance <= bar_dist <= self.max_bar_distance:
                    # Price makes lower high, indicator makes higher high (continuation)
                    if ph2 < ph1 and ih2 > ih1:
                        strength = self._calculate_strength(ph1, ph2, ih1, ih2) * 0.8
                        divergences.append(Divergence(
                            divergence_type=DivergenceType.HIDDEN_BEARISH,
                            indicator=indicator_name,
                            price_point1=float(ph1),
                            price_point2=float(ph2),
                            indicator_point1=float(ih1),
                            indicator_point2=float(ih2),
                            bar_distance=bar_dist,
                            strength=strength,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        return divergences
    
    def _find_swing_highs(
        self,
        data: List[Decimal],
        window: int = 3,
    ) -> List[Tuple[Decimal, int]]:
        """Find swing high points (local maxima)."""
        highs = []
        for i in range(window, len(data) - window):
            is_high = True
            for j in range(1, window + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_high = False
                    break
            if is_high:
                highs.append((data[i], i))
        return highs
    
    def _find_swing_lows(
        self,
        data: List[Decimal],
        window: int = 3,
    ) -> List[Tuple[Decimal, int]]:
        """Find swing low points (local minima)."""
        lows = []
        for i in range(window, len(data) - window):
            is_low = True
            for j in range(1, window + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_low = False
                    break
            if is_low:
                lows.append((data[i], i))
        return lows
    
    def _get_indicator_at_bar(
        self,
        indicator: List[Decimal],
        swing_points: List[Tuple[Decimal, int]],
        target_bar: int,
    ) -> Optional[Decimal]:
        """Get indicator value closest to target bar from swing points."""
        # Find swing point closest to target bar
        closest = None
        min_dist = float('inf')
        
        for value, bar_idx in swing_points:
            dist = abs(bar_idx - target_bar)
            if dist < min_dist:
                min_dist = dist
                closest = value
        
        # If no swing point within 3 bars, use direct value
        if min_dist > 3 and 0 <= target_bar < len(indicator):
            return indicator[target_bar]
        
        return closest
    
    def _calculate_strength(
        self,
        price1: Decimal,
        price2: Decimal,
        ind1: Decimal,
        ind2: Decimal,
    ) -> float:
        """
        Calculate divergence strength (0-1).
        
        Stronger divergence = larger difference between price and indicator trends.
        """
        if price1 == 0 or ind1 == 0:
            return 0.5
        
        price_change_pct = abs((price2 - price1) / price1 * 100)
        ind_change_pct = abs((ind2 - ind1) / abs(ind1) * 100) if ind1 != 0 else Decimal('0')
        
        # Divergence = opposite directions
        # Stronger when both changes are significant
        # Ensure all values are float to avoid Decimal/float mixing
        combined_move = (float(price_change_pct) + float(ind_change_pct)) / 2
        
        # Normalize to 0-1 (5% move = 1.0 strength)
        strength = min(1.0, combined_move / 5.0)
        
        return strength
    
    def _generate_signal(self) -> Optional[Dict[str, Any]]:
        """Generate trading signal from detected divergences."""
        if not self.recent_divergences:
            return None
        
        # Prioritize recent divergences
        bullish_divs = [d for d in self.recent_divergences 
                       if d.divergence_type in [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]]
        bearish_divs = [d for d in self.recent_divergences 
                       if d.divergence_type in [DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH]]
        
        # Calculate combined strength
        bullish_strength = sum(d.strength for d in bullish_divs)
        bearish_strength = sum(d.strength for d in bearish_divs)
        
        if bullish_strength > bearish_strength and bullish_strength > 0.3:
            # Check for multi-indicator confluence
            rsi_bullish = any(d.indicator == 'rsi' for d in bullish_divs)
            macd_bullish = any(d.indicator in ['macd', 'macd_histogram'] for d in bullish_divs)
            
            confidence = 'high' if (rsi_bullish and macd_bullish) else 'medium'
            
            return {
                'direction': 'long',
                'type': 'divergence',
                'strength': min(1.0, bullish_strength),
                'confidence': confidence,
                'indicators': list(set(d.indicator for d in bullish_divs)),
                'divergence_types': [d.divergence_type.value for d in bullish_divs],
                'reason': f"Bullish divergence detected ({len(bullish_divs)} signals)",
            }
        
        elif bearish_strength > bullish_strength and bearish_strength > 0.3:
            rsi_bearish = any(d.indicator == 'rsi' for d in bearish_divs)
            macd_bearish = any(d.indicator in ['macd', 'macd_histogram'] for d in bearish_divs)
            
            confidence = 'high' if (rsi_bearish and macd_bearish) else 'medium'
            
            return {
                'direction': 'short',
                'type': 'divergence',
                'strength': min(1.0, bearish_strength),
                'confidence': confidence,
                'indicators': list(set(d.indicator for d in bearish_divs)),
                'divergence_types': [d.divergence_type.value for d in bearish_divs],
                'reason': f"Bearish divergence detected ({len(bearish_divs)} signals)",
            }
        
        return None
    
    def _get_latest_by_indicator(self, indicator: str) -> Optional[Dict]:
        """Get the most recent divergence for a specific indicator."""
        for div in reversed(self.recent_divergences):
            if div.indicator == indicator:
                return self._div_to_dict(div)
        return None
    
    def _div_to_dict(self, div: Divergence) -> Dict:
        """Convert Divergence to dict."""
        return {
            'type': div.divergence_type.value,
            'indicator': div.indicator,
            'price_points': [div.price_point1, div.price_point2],
            'indicator_points': [div.indicator_point1, div.indicator_point2],
            'bar_distance': div.bar_distance,
            'strength': div.strength,
        }
    
    def get_divergence_score(self, direction: str) -> Tuple[float, str]:
        """
        Get signal score bonus for divergence.
        
        Args:
            direction: 'long' or 'short'
            
        Returns:
            Tuple of (score_bonus, reason)
        """
        if not self.recent_divergences:
            return 0.0, "No divergence"
        
        score = 0.0
        reasons = []
        
        for div in self.recent_divergences:
            is_bullish = div.divergence_type in [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]
            is_bearish = div.divergence_type in [DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH]
            
            if (direction == 'long' and is_bullish) or (direction == 'short' and is_bearish):
                # Regular divergence = stronger signal
                if div.divergence_type in [DivergenceType.REGULAR_BULLISH, DivergenceType.REGULAR_BEARISH]:
                    score += 2.0 * div.strength
                    reasons.append(f"Regular {div.indicator.upper()} divergence")
                else:
                    # Hidden divergence = continuation
                    score += 1.0 * div.strength
                    reasons.append(f"Hidden {div.indicator.upper()} divergence")
            
            elif (direction == 'long' and is_bearish) or (direction == 'short' and is_bullish):
                # Divergence against our direction = penalty
                score -= 1.0 * div.strength
                reasons.append(f"Counter-divergence on {div.indicator.upper()}")
        
        reason = " | ".join(reasons) if reasons else "No matching divergence"
        return score, reason
