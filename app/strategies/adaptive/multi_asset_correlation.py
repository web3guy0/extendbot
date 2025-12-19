"""
Multi-Asset Correlation Analyzer
Checks BTC correlation and relative strength before entries.

Features:
- BTC price correlation check
- Relative strength analysis
- Sector beta calculation
- Correlation-based position sizing
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CorrelationState(Enum):
    """Asset correlation state with BTC."""
    HIGHLY_CORRELATED = "highly_correlated"  # > 0.7
    MODERATELY_CORRELATED = "moderately_correlated"  # 0.4 - 0.7
    LOW_CORRELATION = "low_correlation"  # 0.1 - 0.4
    UNCORRELATED = "uncorrelated"  # < 0.1
    INVERSELY_CORRELATED = "inversely_correlated"  # < -0.4


class RelativeStrength(Enum):
    """Asset strength relative to BTC."""
    OUTPERFORMING = "outperforming"
    MATCHING = "matching"
    UNDERPERFORMING = "underperforming"


@dataclass
class CorrelationAnalysis:
    """Correlation analysis result."""
    correlation: Decimal
    state: CorrelationState
    relative_strength: RelativeStrength
    rs_ratio: Decimal  # Relative Strength Ratio
    btc_trend: str  # 'bullish', 'bearish', 'neutral'
    should_trade: bool
    confidence: Decimal
    notes: List[str]


class MultiAssetCorrelationAnalyzer:
    """
    Analyzes correlation between assets and BTC.
    
    Critical for:
    1. Avoiding trades against BTC trend
    2. Finding alpha opportunities
    3. Risk-adjusted position sizing
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        # Thresholds from env
        self.high_corr_threshold = Decimal(os.getenv('HIGH_CORRELATION', '0.7'))
        self.mod_corr_threshold = Decimal(os.getenv('MODERATE_CORRELATION', '0.4'))
        self.low_corr_threshold = Decimal(os.getenv('LOW_CORRELATION', '0.1'))
        
        # RS thresholds
        self.rs_outperform_threshold = Decimal(os.getenv('RS_OUTPERFORM', '1.05'))
        self.rs_underperform_threshold = Decimal(os.getenv('RS_UNDERPERFORM', '0.95'))
        
        # Lookback periods
        self.correlation_period = int(os.getenv('CORRELATION_PERIOD', '20'))
        self.rs_period = int(os.getenv('RS_PERIOD', '14'))
        
        # BTC price history
        self.btc_prices: deque = deque(maxlen=100)
        
        # Cache
        self.last_btc_analysis: Optional[Dict] = None
        
        logger.info("📊 Multi-Asset Correlation Analyzer initialized")
        logger.info(f"   High Correlation: >{self.high_corr_threshold}")
        logger.info(f"   Correlation Period: {self.correlation_period} candles")
    
    def analyze(
        self,
        symbol: str,
        asset_candles: List[Dict],
        btc_candles: List[Dict],
        direction: str,
    ) -> CorrelationAnalysis:
        """
        Analyze correlation and relative strength.
        
        Args:
            symbol: Asset symbol
            asset_candles: Asset price candles
            btc_candles: BTC price candles
            direction: Proposed trade direction ('long' or 'short')
        """
        notes = []
        
        # Extract prices
        asset_prices = self._extract_returns(asset_candles)
        btc_prices = self._extract_returns(btc_candles)
        
        if len(asset_prices) < self.correlation_period or len(btc_prices) < self.correlation_period:
            return CorrelationAnalysis(
                correlation=Decimal('0'),
                state=CorrelationState.UNCORRELATED,
                relative_strength=RelativeStrength.MATCHING,
                rs_ratio=Decimal('1'),
                btc_trend='neutral',
                should_trade=True,
                confidence=Decimal('0.5'),
                notes=['Insufficient data for correlation analysis'],
            )
        
        # Calculate correlation
        correlation = self._calculate_correlation(
            asset_prices[-self.correlation_period:],
            btc_prices[-self.correlation_period:]
        )
        
        # Determine correlation state
        if correlation >= self.high_corr_threshold:
            corr_state = CorrelationState.HIGHLY_CORRELATED
            notes.append(f"High BTC correlation ({correlation:.2f})")
        elif correlation >= self.mod_corr_threshold:
            corr_state = CorrelationState.MODERATELY_CORRELATED
            notes.append(f"Moderate BTC correlation ({correlation:.2f})")
        elif correlation >= self.low_corr_threshold:
            corr_state = CorrelationState.LOW_CORRELATION
            notes.append(f"Low BTC correlation ({correlation:.2f})")
        elif correlation >= Decimal('-0.4'):
            corr_state = CorrelationState.UNCORRELATED
            notes.append(f"Uncorrelated with BTC ({correlation:.2f})")
        else:
            corr_state = CorrelationState.INVERSELY_CORRELATED
            notes.append(f"Inversely correlated with BTC ({correlation:.2f})")
        
        # Calculate relative strength
        rs_ratio = self._calculate_relative_strength(asset_candles, btc_candles)
        
        if rs_ratio >= self.rs_outperform_threshold:
            rs = RelativeStrength.OUTPERFORMING
            notes.append(f"Outperforming BTC (RS: {rs_ratio:.3f})")
        elif rs_ratio <= self.rs_underperform_threshold:
            rs = RelativeStrength.UNDERPERFORMING
            notes.append(f"Underperforming BTC (RS: {rs_ratio:.3f})")
        else:
            rs = RelativeStrength.MATCHING
            notes.append(f"Matching BTC (RS: {rs_ratio:.3f})")
        
        # Determine BTC trend
        btc_trend = self._get_btc_trend(btc_candles)
        notes.append(f"BTC trend: {btc_trend}")
        
        # Decision logic
        should_trade, confidence = self._should_trade(
            direction, correlation, rs, btc_trend, corr_state
        )
        
        if not should_trade:
            notes.append("⚠️ Trade filtered by correlation analysis")
        
        return CorrelationAnalysis(
            correlation=correlation,
            state=corr_state,
            relative_strength=rs,
            rs_ratio=rs_ratio,
            btc_trend=btc_trend,
            should_trade=should_trade,
            confidence=confidence,
            notes=notes,
        )
    
    def _extract_returns(self, candles: List[Dict]) -> List[Decimal]:
        """Extract percentage returns from candles."""
        returns = []
        for i in range(1, len(candles)):
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            curr_close = Decimal(str(candles[i].get('close', candles[i].get('c', 0))))
            
            if prev_close > 0:
                ret = ((curr_close - prev_close) / prev_close) * 100
                returns.append(ret)
        
        return returns
    
    def _calculate_correlation(
        self,
        returns_a: List[Decimal],
        returns_b: List[Decimal],
    ) -> Decimal:
        """Calculate Pearson correlation between two return series."""
        n = min(len(returns_a), len(returns_b))
        if n < 2:
            return Decimal('0')
        
        # Align series
        a = returns_a[-n:]
        b = returns_b[-n:]
        
        # Calculate means
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        
        # Calculate covariance and std devs
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = (sum((x - mean_a) ** 2 for x in a) / n) ** Decimal('0.5')
        std_b = (sum((x - mean_b) ** 2 for x in b) / n) ** Decimal('0.5')
        
        if std_a == 0 or std_b == 0:
            return Decimal('0')
        
        correlation = cov / (std_a * std_b)
        
        # Clamp to [-1, 1]
        return max(min(correlation, Decimal('1')), Decimal('-1'))
    
    def _calculate_relative_strength(
        self,
        asset_candles: List[Dict],
        btc_candles: List[Dict],
    ) -> Decimal:
        """
        Calculate Relative Strength Ratio.
        
        RS > 1 = Asset outperforming BTC
        RS < 1 = Asset underperforming BTC
        """
        if len(asset_candles) < self.rs_period or len(btc_candles) < self.rs_period:
            return Decimal('1')
        
        # Calculate asset performance
        asset_start = Decimal(str(asset_candles[-self.rs_period].get('close', 
                             asset_candles[-self.rs_period].get('c', 0))))
        asset_end = Decimal(str(asset_candles[-1].get('close', 
                           asset_candles[-1].get('c', 0))))
        
        # Calculate BTC performance
        btc_start = Decimal(str(btc_candles[-self.rs_period].get('close',
                           btc_candles[-self.rs_period].get('c', 0))))
        btc_end = Decimal(str(btc_candles[-1].get('close',
                         btc_candles[-1].get('c', 0))))
        
        if asset_start == 0 or btc_start == 0 or btc_end == 0:
            return Decimal('1')
        
        asset_perf = asset_end / asset_start
        btc_perf = btc_end / btc_start
        
        if btc_perf == 0:
            return Decimal('1')
        
        return asset_perf / btc_perf
    
    def _get_btc_trend(self, btc_candles: List[Dict]) -> str:
        """Determine BTC's current trend."""
        if len(btc_candles) < 20:
            return 'neutral'
        
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in btc_candles[-20:]]
        
        # Simple EMA comparison
        ema_10 = self._calculate_ema(prices, 10)
        ema_20 = self._calculate_ema(prices, 20)
        
        if ema_10 is None or ema_20 is None:
            return 'neutral'
        
        diff_pct = ((ema_10 - ema_20) / ema_20) * 100
        
        if diff_pct > Decimal('0.5'):
            return 'bullish'
        elif diff_pct < Decimal('-0.5'):
            return 'bearish'
        else:
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
    
    def _should_trade(
        self,
        direction: str,
        correlation: Decimal,
        rs: RelativeStrength,
        btc_trend: str,
        corr_state: CorrelationState,
    ) -> tuple[bool, Decimal]:
        """
        Decide if trade should be taken based on correlation analysis.
        
        Returns (should_trade, confidence)
        """
        confidence = Decimal('0.5')
        
        # Case 1: Highly correlated asset going against BTC trend
        if corr_state == CorrelationState.HIGHLY_CORRELATED:
            if direction == 'long' and btc_trend == 'bearish':
                return False, Decimal('0.3')
            if direction == 'short' and btc_trend == 'bullish':
                return False, Decimal('0.3')
            
            # Going with BTC trend - boost confidence
            if (direction == 'long' and btc_trend == 'bullish') or \
               (direction == 'short' and btc_trend == 'bearish'):
                confidence = Decimal('0.8')
        
        # Case 2: Inversely correlated
        if corr_state == CorrelationState.INVERSELY_CORRELATED:
            if direction == 'long' and btc_trend == 'bullish':
                return False, Decimal('0.3')  # Expect asset to drop
            if direction == 'short' and btc_trend == 'bearish':
                return False, Decimal('0.3')  # Expect asset to rise
        
        # Case 3: Relative strength alignment
        if direction == 'long':
            if rs == RelativeStrength.OUTPERFORMING:
                confidence = min(confidence + Decimal('0.2'), Decimal('1'))
            elif rs == RelativeStrength.UNDERPERFORMING:
                confidence = max(confidence - Decimal('0.15'), Decimal('0.3'))
        else:  # short
            if rs == RelativeStrength.UNDERPERFORMING:
                confidence = min(confidence + Decimal('0.2'), Decimal('1'))
            elif rs == RelativeStrength.OUTPERFORMING:
                confidence = max(confidence - Decimal('0.15'), Decimal('0.3'))
        
        return True, confidence
    
    def get_position_size_adjustment(self, analysis: CorrelationAnalysis) -> Decimal:
        """
        Get position size multiplier based on correlation.
        
        High confidence = larger position
        Low confidence = smaller position
        """
        # Base adjustment from confidence
        adjustment = analysis.confidence
        
        # Boost for outperformers going long
        if analysis.relative_strength == RelativeStrength.OUTPERFORMING:
            adjustment = min(adjustment * Decimal('1.1'), Decimal('1'))
        
        # Reduce for underperformers going long
        if analysis.relative_strength == RelativeStrength.UNDERPERFORMING:
            adjustment = adjustment * Decimal('0.9')
        
        # Minimum adjustment
        return max(adjustment, Decimal('0.5'))
