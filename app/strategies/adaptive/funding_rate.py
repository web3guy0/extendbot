"""
Funding Rate Filter
Filter trades against extreme funding rates to avoid squeezes.

CRYPTO-SPECIFIC EDGE:
- High funding = crowd is long, shorts get paid
- Low/negative funding = crowd is short, longs get paid
- Extreme funding = reversal imminent (squeeze risk)

Rule: Don't go with the crowd when funding is extreme.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class FundingRateFilter:
    """
    Funding Rate Analysis for Trade Filtering
    
    HyperLiquid funding rates:
    - Positive: Longs pay shorts (bullish crowd)
    - Negative: Shorts pay longs (bearish crowd)
    - Extreme: >0.1% or <-0.1% = potential squeeze
    
    Trading Edge:
    - Avoid longs when funding > 0.1% (crowd already long)
    - Avoid shorts when funding < -0.1% (crowd already short)
    - Fade extreme funding for mean reversion
    """
    
    def __init__(self, hl_client=None):
        """
        Initialize funding rate filter.
        
        Args:
            hl_client: HyperLiquid client for fetching funding rates
        """
        self.client = hl_client
        
        # Configuration from env
        self.extreme_high = Decimal(os.getenv('FUNDING_EXTREME_HIGH', '0.001'))   # 0.1%
        self.extreme_low = Decimal(os.getenv('FUNDING_EXTREME_LOW', '-0.001'))    # -0.1%
        self.warning_high = Decimal(os.getenv('FUNDING_WARNING_HIGH', '0.0005'))  # 0.05%
        self.warning_low = Decimal(os.getenv('FUNDING_WARNING_LOW', '-0.0005'))   # -0.05%
        
        # Funding rate history
        self.funding_history: Dict[str, list] = {}  # symbol -> list of rates
        
        # Cache
        self._cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._cache_ttl = 60  # seconds
        
        logger.info("💰 Funding Rate Filter initialized")
        logger.info(f"   Extreme thresholds: {float(self.extreme_low):.4%} to {float(self.extreme_high):.4%}")
        logger.info(f"   Warning thresholds: {float(self.warning_low):.4%} to {float(self.warning_high):.4%}")
    
    async def get_funding_rate(self, symbol: str) -> Optional[Decimal]:
        """
        Get current funding rate for a symbol using SDK method.
        
        Uses hl_client.get_funding_rate() which calls info.meta_and_asset_ctxs()
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Funding rate as Decimal (e.g., 0.0001 = 0.01%)
        """
        # Check cache
        if symbol in self._cache:
            rate, cached_at = self._cache[symbol]
            if (datetime.now(timezone.utc) - cached_at).total_seconds() < self._cache_ttl:
                return rate
        
        try:
            if self.client:
                # Use SDK method from hl_client
                funding = self.client.get_funding_rate(symbol)
                if funding is not None:
                    rate = Decimal(str(funding))
                    
                    # Cache it
                    self._cache[symbol] = (rate, datetime.now(timezone.utc))
                    
                    # Update history
                    if symbol not in self.funding_history:
                        self.funding_history[symbol] = []
                    self.funding_history[symbol].append(float(rate))
                    if len(self.funding_history[symbol]) > 100:
                        self.funding_history[symbol].pop(0)
                    
                    return rate
        except Exception as e:
            logger.warning(f"Failed to get funding rate for {symbol}: {e}")
        
        return None
    
    def check_funding(
        self,
        symbol: str,
        direction: str,
        funding_rate: Optional[Decimal] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if funding rate allows the trade.
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            funding_rate: Pre-fetched funding rate (optional)
            
        Returns:
            Tuple of (allowed, reason, details)
        """
        if funding_rate is None:
            return True, "No funding data available", {'funding': None}
        
        details = {
            'funding': float(funding_rate),
            'funding_pct': f"{float(funding_rate):.4%}",
            'direction': direction,
        }
        
        # Extreme high funding
        if funding_rate > self.extreme_high:
            if direction == 'long':
                return False, f"Funding too high ({float(funding_rate):.4%}) - crowd already long, squeeze risk", details
            else:
                # Short is okay (you get paid)
                details['edge'] = 'receiving_funding'
                return True, f"High funding favors shorts ({float(funding_rate):.4%})", details
        
        # Extreme low funding
        if funding_rate < self.extreme_low:
            if direction == 'short':
                return False, f"Funding too negative ({float(funding_rate):.4%}) - crowd already short, squeeze risk", details
            else:
                # Long is okay (you get paid)
                details['edge'] = 'receiving_funding'
                return True, f"Negative funding favors longs ({float(funding_rate):.4%})", details
        
        # Warning zone
        if funding_rate > self.warning_high and direction == 'long':
            details['warning'] = 'elevated_funding'
            return True, f"Elevated funding ({float(funding_rate):.4%}) - proceed with caution", details
        
        if funding_rate < self.warning_low and direction == 'short':
            details['warning'] = 'negative_funding'
            return True, f"Negative funding ({float(funding_rate):.4%}) - proceed with caution", details
        
        return True, f"Funding normal ({float(funding_rate):.4%})", details
    
    def get_funding_bias(self, funding_rate: Optional[Decimal]) -> str:
        """
        Get market bias from funding rate.
        
        Args:
            funding_rate: Current funding rate
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if funding_rate is None:
            return 'neutral'
        
        # High funding = crowd is bullish (contrarian: bearish)
        if funding_rate > self.extreme_high:
            return 'bearish'  # Contrarian
        
        # Low funding = crowd is bearish (contrarian: bullish)
        if funding_rate < self.extreme_low:
            return 'bullish'  # Contrarian
        
        # Moderate positive = slight bullish trend
        if funding_rate > self.warning_high:
            return 'bullish'
        
        # Moderate negative = slight bearish trend
        if funding_rate < self.warning_low:
            return 'bearish'
        
        return 'neutral'
    
    def get_signal_score(
        self,
        direction: str,
        funding_rate: Optional[Decimal],
    ) -> Tuple[float, str]:
        """
        Get signal score modifier based on funding.
        
        Args:
            direction: 'long' or 'short'
            funding_rate: Current funding rate
            
        Returns:
            Tuple of (score_modifier, reason)
        """
        if funding_rate is None:
            return 0.0, "No funding data"
        
        # Perfect setup: trade against extreme funding
        if funding_rate > self.extreme_high and direction == 'short':
            return 1.5, f"Fading extreme bullish funding ({float(funding_rate):.4%})"
        
        if funding_rate < self.extreme_low and direction == 'long':
            return 1.5, f"Fading extreme bearish funding ({float(funding_rate):.4%})"
        
        # Bad setup: going with extreme funding
        if funding_rate > self.extreme_high and direction == 'long':
            return -2.0, f"Going with crowded longs ({float(funding_rate):.4%})"
        
        if funding_rate < self.extreme_low and direction == 'short':
            return -2.0, f"Going with crowded shorts ({float(funding_rate):.4%})"
        
        # Slight bonus for favorable funding
        if funding_rate > 0 and direction == 'short':
            return 0.3, "Receiving positive funding"
        
        if funding_rate < 0 and direction == 'long':
            return 0.3, "Receiving negative funding"
        
        return 0.0, "Neutral funding"
    
    def get_funding_stats(self, symbol: str) -> Dict[str, Any]:
        """Get funding statistics for a symbol."""
        if symbol not in self.funding_history or not self.funding_history[symbol]:
            return {'error': 'No funding history'}
        
        history = self.funding_history[symbol]
        
        return {
            'current': history[-1] if history else None,
            'avg_1h': sum(history[-8:]) / len(history[-8:]) if len(history) >= 8 else None,  # 8x 8hr = 1 day approx
            'min': min(history),
            'max': max(history),
            'trend': 'rising' if len(history) >= 2 and history[-1] > history[-2] else 'falling',
        }


# Singleton for easy access
_funding_filter: Optional[FundingRateFilter] = None


def get_funding_filter(hl_client=None) -> FundingRateFilter:
    """Get or create the funding rate filter singleton."""
    global _funding_filter
    if _funding_filter is None:
        _funding_filter = FundingRateFilter(hl_client)
    elif hl_client and _funding_filter.client is None:
        _funding_filter.client = hl_client
    return _funding_filter
